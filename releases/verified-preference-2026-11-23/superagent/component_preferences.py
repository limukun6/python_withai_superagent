"""Opt-in component accounting, worker isolation and delivery through the normal A/B CLI."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.pipeline import Pipeline

from . import ab_run
from .ab_sentence import verified_directory
from .contracts import TaskSpec
from .data import load_development_task
from .metrics import MetricSpec
from .splits import sample_groups


ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'scripts/b_component_worker.py'
HELPERS=('tools/preference_adapter.py','tools/milestone_validation.py','tools/token_pooling_study.py',
         'tools/token_value_head.py','superagent/pair_attention.py','superagent/pretrained_choice.py',
         'superagent/minilm_features.py','superagent/models.py')


def sources():
    legacy=Path(os.environ['SUPERAGENT_B_COMPONENT_RUNTIME'])
    return {name:ab_run.sha(legacy/name) for name in HELPERS} | {'worker':ab_run.sha(WORKER)}


def worker_env():
    env=dict(os.environ,PYTHONPATH=str(ROOT/'scripts/ab_acceptance_guard'),AB_B1_OFFLINE='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SUPERAGENT_OFFLINE='1')
    return env


def save(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2)


class ComponentModel(ClassifierMixin,BaseEstimator):
    classes_=np.array([0,1],dtype=int)

    def __init__(self,payload,source_hashes):
        self.payload=payload
        self.source_hashes=source_hashes

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        raise RuntimeError('Component fitting must go through the metered training worker')

    def predict_proba(self,rows):
        if sources()!=self.source_hashes:
            raise ValueError('Worker or frozen preference helper changed')
        if not len(rows):
            return np.empty((0,2))
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            (d/'component.joblib').write_bytes(self.payload)
            save(d/'rows.json',list(rows))
            subprocess.run([sys.executable,str(WORKER),'predict','--data',str(d/'component.joblib'),
                            '--rows',str(d/'rows.json'),'--out',str(d/'positive.npy')],
                           check=True,env=worker_env(),timeout=1200)
            p=np.load(d/'positive.npy',allow_pickle=False)
        if p.shape!=(len(rows),) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid classification probabilities')
        return np.column_stack((1-p,p))

    def predict(self,rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]


def run(args):
    if args.planner!='local' or args.scheduler!='ab-v1' or args.catalog_scope!='full':
        raise ValueError('Component research requires local ab-v1 with the full catalog')
    limit=ab_run.validate_options(args)
    archive=Path(args.task_package).resolve()
    task=load_development_task(archive)
    contract=TaskSpec.from_task(task).record()
    schema=task.spec.get('input_schema',{})
    eligible=(task.kind=='pair_text' and schema.get('pair_semantics')=='preference' and
              schema.get('pair_label_one') in ('left','right') and set(task.y_train.tolist())=={0,1} and
              MetricSpec.for_task(task).name=='accuracy' and limit>=3 and verified_directory() is not None and
              all(os.environ.get(k) and Path(os.environ[k]).is_dir() for k in
                  ('SUPERAGENT_DEEP_MINILM_WEIGHTS','SUPERAGENT_B_COMPONENT_RUNTIME')))
    if not eligible:
        return ab_run.command(args)
    began=time.monotonic()
    folder=Path(args.run_dir or Path(args.output_dir or 'runs')/f'components_{time.time_ns()}').resolve()
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'llm_usage.jsonl').touch(exist_ok=False)
    frozen=sources()
    outcome=dict(requested_planner_completed=False,formal_test_accessed=False,new_api_requests=0,known_course_cost_rmb=0)
    try:
        metric=MetricSpec.for_task(task)
        ab_run.freeze_partitions(task,folder,args.seed,metric)
        primary=json.loads((folder/'primary_split.json').read_text())
        groups=sample_groups(task,'shared_text')
        path=folder/'training_payload.joblib'
        joblib.dump(dict(rows=list(task.X_train),labels=task.y_train,groups=groups,
                         fit_ids=primary['train_indices'],held_ids=primary['search_indices']),path,compress=3)
        save(folder/'run_manifest.json',dict(entrypoint='superagent run',scheduler='ab-v1',
             policy='local_conditional_components',task_contract=contract,task_archive_sha256=ab_run.sha(archive),
             source_files=ab_run.sources(),worker_runtime=frozen,fit_limit=limit,seed=args.seed,
             time_limit_seconds=args.time_limit,trial_timeout_seconds=args.trial_timeout,llm_request_limit=0,
             training_payload_sha256=ab_run.sha(path),primary=ab_run.sha(folder/'primary.joblib'),
             catalog=list(ab_run.registered_capabilities(contract,len(task.y_train),'full')),
             component_portfolio=['word','base','deep','plain_equal','stream_adapter','adapter','stream_hybrid','full_hybrid']))
        seconds=args.time_limit-(time.monotonic()-began)
        subprocess.run([sys.executable,str(WORKER),'train','--data',str(path),'--out',str(folder),
                        '--limit',str(limit),'--wall',str(seconds),'--timeout',str(args.trial_timeout)],
                       check=True,env=worker_env(),timeout=max(.001,seconds))
        res=json.loads((folder/'worker_result.json').read_text())
        log=[json.loads(v) for v in (folder/'component_fits.jsonl').read_text().splitlines()]
        expected=list(range(1,res['actual_fits']+1))
        if res['actual_fits']>limit or any([v['number'] for v in log if v['event']==kind]!=expected
                     for kind in ('attempt','physical_fit','completed')):
            raise ValueError('Physical fit observations disagree with declared component fits')
        data=folder/'component_payload.joblib'
        if res['payload_sha256']!=ab_run.sha(data):
            raise ValueError('Final trained components changed')
        model=Pipeline([('preference',ComponentModel(data.read_bytes(),frozen))])
        family='component_'+res['selected']
        joblib.dump(dict(model=model,kind='pair_text',candidate=family),folder/'selected_model.joblib',compress=3)
        spec=dict(task.spec,model_kind='pair_text',task_kind='pair_text_binary',metric='accuracy',metric_spec=metric.record())
        save(folder/'task_spec.json',spec)
        save(folder/'final_selection.json',dict(selected_experiment=res['selected'],family=family,
             model_file='selected_model.joblib',model_format='candidate_bundle_v1',
             model_sha256=ab_run.sha(folder/'selected_model.joblib'),task_spec_sha256=ab_run.sha(folder/'task_spec.json'),
             refit=True,training_rows=len(task.y_train),test_accessed=False,selection_metric='pooled_primary_accuracy'))
        outcome.update(status='completed',model_available=True,requested_planner_completed=True,full_refit_completed=True,
                       selected_candidate=family,classifier_fits=res['actual_fits'])
    except BaseException as exc:
        outcome.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        save(folder/'run_outcome.json',outcome)
    print(json.dumps(outcome,ensure_ascii=False))
    return 0
