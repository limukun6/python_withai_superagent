"""Small offline integration fixture. Run as python -m scripts.run_b1_ab_acceptance NEW_DIR."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import joblib
import numpy as np
from sklearn.datasets import make_classification

from superagent.ab_catalog import common_catalog
from superagent.ab_methods import FIELDS, IMAGE, TEXT
from superagent.ab_scheduler import CONFIRMATIONS, PRIMARY, ROW_SCOPE, next_action, selection_evidence
from superagent.benchmark import train_candidate
from superagent.confirmation import decide
from superagent.contracts import TaskSpec
from superagent.data import DevelopmentTask, load_training_task
from superagent.events import RunLogger
from superagent.finalization import finalize
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model, probability_with_model
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition

ROOT = Path(__file__).resolve().parents[1]
SEEDS = dict(zip((PRIMARY,*CONFIRMATIONS),(41,142,243),strict=True))
BASE = '0b112d823a1991686281fdc88602e758e29a4202'
WALL_SECONDS = 240
TRIAL_SECONDS = 65


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False) + chr(10),encoding='utf-8')


def read(path):
    return json.loads(path.read_text())


def trace(run):
    path = run/'physical_fits.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def sources():
    files = [*sorted((ROOT/'superagent').glob('*.py')),Path(__file__).resolve(),
             ROOT/'scripts/ab_acceptance_guard/sitecustomize.py',ROOT/'docs/batches/B1_01_plan.md',
             ROOT/'docs/batches/B1_02_plan.md']
    return digest({str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files})


def offline(run):
    guard = ROOT/'scripts/ab_acceptance_guard'
    os.environ.update(AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(run/'physical_fits.jsonl'),HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(guard),str(ROOT)]),
                      TOKENIZERS_PARALLELISM='false')
    def reject(event,args):
        if event in ('socket.connect','socket.sendto','socket.getaddrinfo'):
            raise PermissionError('Network disabled in the acceptance controller')
    sys.addaudithook(reject)


def packages(run):
    x,y = make_classification(n_samples=256,n_features=8,n_informative=4,n_redundant=2,
                              class_sep=.7,flip_y=.12,random_state=932)
    query = np.random.default_rng(975).normal(size=(31,x.shape[1]))
    for split, arrays in [('train',dict(X=x,y=y)),('test',dict(X=query))]:
        data = io.BytesIO()
        np.savez(data,**arrays)
        manifest = dict(schema_version=1,objective='classification',modality='tabular',metric='accuracy',
                        splits={split:dict(format='npz',path='rows.npz')})
        with zipfile.ZipFile(run/(split+'.zip'),'w') as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            archive.writestr('rows.npz',data.getvalue())
    train = load_training_task(run/'train.zip')
    return DevelopmentTask(train.kind,'anonymous',train.X_train,train.y_train,None,None,train.spec),query


def freeze_partitions(task,run):
    metric = MetricSpec.for_task(task)
    partitions = {}
    for key,seed in SEEDS.items():
        train,held,_ = grouped_partition(task,seed)
        path = run/(key+'.joblib')
        payload = dict(kind=task.kind,X_train=task.X_train[train],y_train=task.y_train[train],
                       X_search=task.X_train[held],y_search=task.y_train[held],metric_spec=metric.record())
        joblib.dump(payload,path)
        partitions[key] = dict(row_ids=held.tolist(),labels=payload['y_search'],
                               evaluation_payload_sha256=sha(path.read_bytes()))
    return partitions


def checked_worker(run, payload, name, destination, timeout):
    before = len(trace(run))
    model, metrics = train_candidate(payload,name,destination,timeout)
    observed = trace(run)[before:]
    if model is None:
        raise RuntimeError(name+' worker failed: '+str(metrics.get('error')))
    if (metrics.get('classifier_fits') != len(observed) or not observed
            or any(r['candidate'] != name for r in observed)):
        raise RuntimeError('Worker fit accounting disagrees with observed fit entries: '+name)
    return model,metrics


def observe(run,task,partitions,menu):
    history,evidence,actions = [],[],[]
    limit = 11
    began = time.monotonic()
    while True:
        used = len(trace(run))
        budget = dict(fit_limit=limit,completed_classifier_fits=used,attempted_classifier_fits=used,
                      remaining_fit_allowance=limit-used,unknown_failed_attempts=0,reserved_final_refit=1,
                      used_llm_requests=0,llm_request_limit=0,elapsed_seconds=time.monotonic()-began,
                      time_limit_seconds=WALL_SECONDS)
        state = dict(ab_schema_version=1,task=TaskSpec.from_task(task).record(),history=history,
                     available_candidates=menu,prediction_evidence=evidence,budget=budget,
                     registered_confirmation_splits=list(CONFIRMATIONS))
        action = next_action(state,trusted_partitions=partitions)
        descriptor = dict(number=len(actions)+1,state_sha256=digest(state),fits_before=used,**action)
        actions.append(descriptor)
        with (run/'inputs.jsonl').open('a') as file:
            file.write(json.dumps(state,allow_nan=False)+chr(10))
        with (run/'actions.jsonl').open('a') as file:
            file.write(json.dumps(descriptor)+chr(10))
        if action['action'] == 'stop':
            break
        name,split = action['candidate_id'],action['split_id']
        trial = len(history)+1
        if action['estimated_classifier_fits'] + used + 1 > limit:
            raise RuntimeError('Proposed action would exhaust final refit allowance')
        destination = run/f'trial_{trial}.joblib'
        _,m = checked_worker(run,run/(split+'.joblib'),name,destination,
                             min(TRIAL_SECONDS,WALL_SECONDS - (time.monotonic()-began)))
        if m['classifier_fits'] != action['estimated_classifier_fits']:
            raise ValueError('Declared and observed fit counts differ')
        candidate = next(c for c in menu if c['id'] == name)
        if m['estimator']['family'] != candidate['estimator']:
            raise ValueError('Declared and fitted classifiers differ')
        row = dict(trial=trial,candidate=name,params=candidate['params'],split_id=split,
                   **{k:m[k] for k in ('status','primary_metric','search_score','search_accuracy','classifier_fits','estimator')})
        history.append(row)
        raw = joblib.load(destination)['prediction_record']
        evidence.append(dict(trial=trial,candidate_id=name,split_id=split,row_ids=partitions[split]['row_ids'],
                             row_id_scope=ROW_SCOPE,predictions=raw['predictions'].tolist(),
                             **{k:raw[k] for k in ('label_sha256','prediction_sha256','evaluation_payload_sha256')}))
    return history,actions,began,selection_evidence(state,trusted_partitions=partitions)


def finish(run,task,history,began,review):
    primary = [r for r in history if r['split_id']==PRIMARY]
    by_candidate = {(r['candidate'],r['split_id']):r for r in history}
    chosen,aligned = review['incumbent'],review['basis']
    last = review['promotions'][-1] if review['promotions'] else None
    names = [last['previous'],chosen] if last else [chosen]
    ranked = [by_candidate[name,PRIMARY] for name in names]
    measurements = {name:[dict(seed=SEEDS[s],**by_candidate[name,s]) for s in aligned] for name in names}
    selection = decide(ranked,measurements,[SEEDS[s] for s in aligned],'exact_rows')
    if not selection['valid_confirmation'] or selection['selected_candidate'] != chosen:
        raise ValueError('A selection and supported paired evidence disagree')
    selection.update(b_scheduler_review=review,
                     selection_rule='Best of initial three; replace on two paired wins and mean gain >= 0.002')
    row = by_candidate[chosen,PRIMARY]
    frozen = joblib.load(run/f"trial_{row['trial']}.joblib")['model']
    before = len(trace(run))
    def worker(payload,name,dest,timeout):
        return checked_worker(run,payload,name,dest,timeout)
    metrics = finalize(task,primary,frozen,chosen,row['search_score'],joblib.load(run/'primary.joblib'),
                       run,RunLogger(run),worker,began,WALL_SECONDS,TRIAL_SECONDS,True,False,
                       confirmation=selection)
    if not metrics['refit_succeeded'] or len(trace(run)) != before+1 or len(trace(run)) > 11:
        raise RuntimeError('Full refit unsuccessful or budget violated')
    return dict(family=chosen,selection_splits=aligned,selection_rule=read(run/'selection.json')['selection_rule'],
                incumbent_evidence=review,
                selection_score=read(run/'final_selection.json')['selection_score'],search_fits=before,
                full_refits=len(trace(run))-before,task_classifier_fits=len(trace(run)))


def predict_twice(run,query,*,output_dir=None,task_package=None):
    original = read(run/'final_selection.json')
    workspace = run if output_dir is None else output_dir
    query_package = run/'test.zip' if task_package is None else task_package
    relocated = workspace/'relocated'
    relocated.mkdir()
    for fname in ('task_spec.json','final_selection.json',original['model_file']):
        shutil.copy2(run/fname,relocated/fname)
    model = joblib.load(run/original['model_file'])['model']
    expected = predict_with_model('numeric_tabular',model,query).astype(str).tolist()
    outputs = []
    for source,export in ((run,workspace),(relocated,relocated)):
        dest = export/'predict'/'rows.csv'
        subprocess.run([sys.executable,'-m','superagent','predict',str(query_package),'--run-dir',str(source),
                        '--output',str(dest)],check=True,cwd=ROOT,capture_output=True,text=True,timeout=TRIAL_SECONDS)
        with dest.open(newline='') as file:
            rows = list(csv.DictReader(file))
        if [r['row_id'] for r in rows] != list(map(str,range(len(expected)))) or [r['prediction'] for r in rows]!=expected:
            raise ValueError('Fresh interpreter did not reproduce the frozen in-process prediction')
        outputs.append(sha(dest.read_bytes()))
    if outputs[0] != outputs[1]:
        raise ValueError('Relocated CSV bytes differ')
    return dict(fresh_interpreters=2,rows_per_prediction=len(query),matching_row_ids_and_labels=True,
                prediction_csv_sha256=outputs[0],model_sha256=original['model_sha256'],
                task_spec_sha256=original['task_spec_sha256'],prediction_fixture_contains_labels=False)


def probes(run):
    rng = np.random.default_rng(577)
    x = rng.integers(0,4,(300,5)).astype(str)
    targets = (x[:,0] == x[:,1]).astype(int)
    pictures = rng.normal(size=(60,12,12))
    stories = ['study reports clear results','analysis notes a steady pattern','research finds new answers',
               'report asks an open question','investigation explains a local change','study reviews earlier examples']
    pairs = [(stories[i%6]+' subject '+str(i),stories[(i+1)%6]+' report '+str(i)) for i in range(40)]
    cases = [('categorical_tabular',FIELDS,x,targets),('image',IMAGE,pictures,np.arange(60)%2),
             ('single_text',TEXT,[p[0] for p in pairs],np.arange(40)%2),
             ('pair_text',TEXT,pairs,np.arange(40)%2)]
    results = []
    for kind,name,rows,labels in cases:
        schema = {'pair_semantics':'preference'} if kind=='pair_text' else {}
        metric = MetricSpec('roc_auc',(0,1))
        task = dict(contract_version=1,objective='classification',kind=kind,input_schema=schema,metric=metric.record())
        menu = common_catalog(task,train_rows=len(rows))
        entry = next((r for r in menu if r['id']==name),None)
        if entry is None:
            results.append(dict(kind=kind,id=name,status='dependency_unavailable',classifier_fits=0))
            continue
        path = run/(kind+'_probe.joblib')
        dest = run/(kind+'_fit.joblib')
        joblib.dump(dict(kind=kind,metric_spec=metric.record(),pair_semantics=schema.get('pair_semantics'),
                         X_train=rows[:-8],y_train=labels[:-8],X_search=rows[-8:],y_search=labels[-8:]),path)
        model,out = checked_worker(run,path,name,dest,TRIAL_SECONDS)
        cls,p = probability_with_model(kind,model,rows[-8:])
        if (out['classifier_fits']!=entry['expected_classifier_fits'] or out['estimator']['family']!=entry['estimator']
                or list(cls)!=[0,1] or p.shape!=(8,2) or not np.allclose(p.sum(1),1)):
            raise ValueError('Initial-menu capability failed: '+name)
        result = dict(kind=kind,id=name,status='passed',visible_before_training=True,estimator=out['estimator']['class'],
                      classifier_fits=out['classifier_fits'],probability_shape=list(p.shape),
                      worker_seconds=round(out['worker_wall_seconds'],3))
        if kind=='pair_text':
            _,reverse = probability_with_model(kind,model,[(b,a) for a,b in rows[-8:]])
            if not np.allclose(reverse[:,1],1-p[:,1],atol=1.e-6):
                raise ValueError('Preference probability fails swap rule')
            result['swap_probabilities_complementary'] = True
        if name==TEXT:
            result.update(checkpoint=entry['params']['revision'],device='cpu',frozen_encoder=True,
                          training_worker_tokens=out['training_metadata']['usage']['input_tokens'])
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    run = parser.parse_args().directory.resolve()
    run.mkdir(parents=True,exist_ok=False)
    offline(run)
    result = {}
    try:
        task,query = packages(run)
        partitions = freeze_partitions(task,run)
        menu = common_catalog(TaskSpec.from_task(task).record(),train_rows=len(task.y_train))
        tools = {k:version(k) for k in ('numpy','scipy','scikit-learn','joblib')}
        protocol = dict(batch='B1-02',base=BASE,source_sha256=sources(),library_versions=tools,
                        python=sys.version.split()[0],frozen_utc=datetime.now(timezone.utc).isoformat(),
                        dataset='anonymous generated numeric classification',data_seed=932,query_seed=975,
                        auxiliary_fixture_seed=577,training_rows=len(task.y_train),training_classes=[0,1],seeds=SEEDS,
                        grouped_splits={s:dict(row_ids_sha256=fingerprint(v['row_ids']),held_rows=len(v['row_ids']),
                        evaluation_payload_sha256=v['evaluation_payload_sha256']) for s,v in partitions.items()},
                        candidate_ids=[c['id'] for c in menu],catalog_sha256=digest(menu),fit_limit=11,
                        task_time_limit_s=WALL_SECONDS,worker_time_limit_s=TRIAL_SECONDS,
                        paid_requests_allowed=0,course_flash_cap=3,prediction_rows=len(query),
                        initial_assets='Only locally verified optional dependencies',selection='Protect initial best of three; two matched wins and mean gain >= 0.002; A decide on common evidence, A finalize')
        write(run/'protocol.json',protocol)
        history,actions,began,review = observe(run,task,partitions,menu)
        result.update(actions=actions,observations=history)
        result.update(finish(run,task,history,began,review))
        result['prediction'] = predict_twice(run,query)
        result['post_reload_task_elapsed_seconds'] = round(time.monotonic()-began,3)
        result['task_physical_fit_trace'] = trace(run)
        result['capability_probes'] = probes(run)
        result.update(total_observed_fits=len(trace(run)),paid_requests=0,input_api_tokens=0,
                      output_api_tokens=0,api_rmb=0,formal_data_accessed=False)
        action_types = sorted(set(r['action'] for r in actions))
        result['observed_action_types'] = action_types
        if action_types!=['confirm','explore','stop']:
            raise RuntimeError('This predeclared task did not exercise all three actions')
        write(run/'results.json',result)
        write(run/'completion.json',dict(status='passed',all_three_actions=True,cross_process_prediction=True,
             b_driver_with_a_components=True,production_cli_dynamic_dispatch_tested=False,
             task_classifier_fits=result['task_classifier_fits'],probe_classifier_fits=sum(r['classifier_fits'] for r in result['capability_probes']),
             fit_limit_including_refit=11,paid_requests=0,api_rmb=0))
    except Exception as error:
        write(run/'results.json',result)
        write(run/'completion.json',dict(status='failed',error_type=type(error).__name__,reason=str(error),
              observed_fit_entries=len(trace(run))))
        raise


if __name__ == '__main__':
    main()
