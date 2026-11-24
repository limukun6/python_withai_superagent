"""Run the frozen local planner unchanged on a training-only anonymous categorical table."""
import argparse
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

from scripts.audit_b_varied_decisions import audit_arm, verify_portable
from scripts.run_b_varied_preferences import ROOT, bucket, cli, read, read_predictions, require, save
from superagent.ab_catalog import untried_candidates
from superagent.ab_method_search import rank
from superagent.ab_run import freeze_partitions, registered_capabilities, sha, sources
from superagent.ab_scheduler import CONFIRMATIONS, next_action
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.scheduler_bridge import validate_action
from superagent.splits import sample_groups
from superagent.task_profile import training_facts


PARENT_SOURCE='101a2c6d59d5a109a667804f642c933cf39932f2a7e1bd9987ff319d1339633b'
PLAN=ROOT/'docs/batches/B_autonomous_table_transfer_20261124_zh.md'
SALT='b-anonymous-table-varied-20261124/v1/'
ARM='path'


def pack(path,features,labels=None):
    names=[f'x{i}' for i in range(features.shape[1])]
    has_labels=labels is not None
    split='train' if has_labels else 'test'
    manifest=dict(schema_version=1,objective='classification',modality='tabular',metric='accuracy',
                  columns=[dict(name=n,type='categorical') for n in names],target='outcome',
                  splits={split:dict(path='rows.csv',format='csv')})
    stream=io.StringIO(newline='')
    writer=csv.writer(stream)
    writer.writerow(names+(['outcome'] if has_labels else []))
    for i,row in enumerate(features):
        writer.writerow([*row,*([int(labels[i])] if has_labels else [])])
    with zipfile.ZipFile(path,'x') as z:
        for filename,body in (('manifest.json',json.dumps(manifest)),('rows.csv',stream.getvalue())):
            info=zipfile.ZipInfo(filename,(2020,1,1,0,0,0))
            info.compress_type,info.external_attr,info.create_system=zipfile.ZIP_DEFLATED,0o600<<16,3
            z.writestr(info,body.encode())


def smoke():
    with TemporaryDirectory() as scratch:
        folder=Path(scratch)
        x=np.array([[str(i),str(i%7)] for i in range(100)])
        pack(folder/'packet.zip',x,np.arange(100)%2)
        task=load_development_task(folder/'packet.zip')
        spec=TaskSpec.from_task(task)
        caps=registered_capabilities(spec.record(),len(x))
        trusted,allowed=freeze_partitions(task,folder,51803,spec.metric)
        state=dict(ab_schema_version=1,task=spec.record(),history=[],available_candidates=untried_candidates(caps,[]),
                   prediction_evidence=[],registered_confirmation_splits=list(CONFIRMATIONS),training_profile=training_facts(task),
                   budget=dict(fit_limit=11,attempted_classifier_fits=0,completed_classifier_fits=0,reserved_final_refit=1,
                               remaining_fit_allowance=11,unknown_failed_attempts=0,llm_request_limit=0,used_llm_requests=0,
                               time_limit_seconds=1200,elapsed_seconds=0))
        action=next_action(state,capabilities=caps,trusted_partitions=trusted,on_local_method=lambda ids,history:rank(state,caps,ids,history))
        admission=validate_action(state,action,caps,allowed)
        require(action['action']=='explore' and admission['expected_classifier_fits']==1,'Synthetic action not admitted')
        return dict(registered=len(caps),fitted=0,action=action['action'],proposed=action['candidate_id'])


def run(a):
    require(sha(a.training)==PARENT_SOURCE,'Different training archive')
    task=load_development_task(a.training)
    require(task.X_train.shape==(6490,21) and task.y_val is None and task.kind=='categorical_tabular','Unexpected training data')
    groups=sample_groups(task)
    mask=np.array([bucket(g,SALT,5)==0 for g in groups])
    require(int(mask.sum())==1269 and not set(groups[mask]).intersection(groups[~mask]),'Invalid outer grouping')
    folder=a.out/'inputs/k3'
    native=a.out/ARM/'agent'
    arm=native.parent
    meter=arm/'physical.jsonl'
    if not (a.out/'protocol.json').exists():
        folder.mkdir(parents=True,exist_ok=False)
        arm.mkdir()
        check=smoke()
        pack(folder/'train.zip',task.X_train[~mask],task.y_train[~mask])
        pack(folder/'features.zip',task.X_train[mask])
        tagged=load_development_task(folder/'train.zip')
        native_labels=tagged.y_train.astype(str)
        require(set(native_labels)=={'0','1'} and np.array_equal(tagged.X_train,task.X_train[~mask].astype(str)) and
                np.array_equal(native_labels,task.y_train[~mask].astype(str)),
                'Repackaging changed a category or outcome')
        plan=dict(source=sources(),input=PARENT_SOURCE,plan=sha(PLAN),runner=sha(Path(__file__)),
                  physical_guard=sha(ROOT/'scripts/ab_acceptance_guard/sitecustomize.py'),seed=51803,fit_limit=11,api_limit=0,
                  train=sha(folder/'train.zip'),features=sha(folder/'features.zip'),training_rows=int(sum(~mask)),
                  holdout_rows=int(sum(mask)),unique_holdout_groups=len(set(groups[mask])),shared_feature_groups=0,
                  majority=int(np.bincount(native_labels.astype(int)).argmax()),held_positions=fingerprint(np.flatnonzero(mask).tolist()),
                  label_roundtrip='CSV category labels 0/1 match every original integer after exact string conversion',
                  fixture_check=check,course_validation_accessed=False,formal_test_accessed=False)
        save(a.out/'protocol.json',plan)
    p=read(a.out/'protocol.json')
    require(p['source']==sources() and p['plan']==sha(PLAN) and p['runner']==sha(Path(__file__)) and
            p['train']==sha(folder/'train.zip') and p['physical_guard']==sha(ROOT/'scripts/ab_acceptance_guard/sitecustomize.py'),
            'Frozen table protocol changed')
    if not (native/'run_outcome.json').exists():
        cli(a,['run',folder/'train.zip','--planner','local','--scheduler','ab-v1','--portfolio','ab-v1',
               '--local-method-search','evidence','--task-profile','train','--fit-limit','11','--seed','51803',
               '--time-limit','1200','--trial-timeout','300','--run-dir',native],meter,arm/'train.log',1260)
    outcome=read(native/'run_outcome.json')
    require(outcome['status']=='completed' and outcome['requested_planner_completed'] and outcome['source_unchanged']
            and outcome['full_refit_completed'] and outcome['new_api_requests']==0 and not outcome['formal_test_accessed'],
            'Table native route incomplete')
    fits=read(native/'classifier_fits.json')
    observed=[json.loads(line) for line in meter.read_text().splitlines()]
    require(0<len(observed)<=11 and len(observed)==fits['completed_classifier_fits']==fits['attempted_classifier_fits'] and
            not fits['unknown_failed_attempts'] and all(v=={'candidate':w['candidate'],'classifier':w['calls'][0]['classifier']}
                for v,w in zip(observed,fits['workers'],strict=True)),'Table fit accounting differs')
    final=read(native/'final_selection.json')
    require(final['refit'] and final['training_rows']==p['training_rows'] and sha(native/'selected_model.joblib')==final['model_sha256'],
            'Table final model not fully refitted')
    history=read(native/'history.json')
    frozen=dict(selection=read(native/'confirmation.json')['b_scheduler_review'],selected=final['family'],physical=observed,
                model=final['model_sha256'],actions=read(native/'actions.json'),
                trials=[dict(candidate=r['candidate'],family=r['estimator']['class'],split=r['split_id'],
                             score=r['search_score'],rows=r['search_metrics']['rows']) for r in history])
    save(arm/'frozen.json',frozen)
    report,choices=audit_arm(a,ARM,arm_spec=('table','standard',11),input_slot='k3')
    verify_portable({ARM:choices})
    save(a.out/'decision_inputs.json',{ARM:choices})
    save(a.out/'decision_audit.json',report)
    require(sha(folder/'features.zip')==p['features'],'Unlabelled features changed')
    exports=[]
    for i in range(2):
        dest=arm/f'export_{i}'/'rows.csv'
        if not dest.exists():
            cli(a,['predict',folder/'features.zip','--run-dir',native,'--output',dest],meter,arm/f'export_{i}.log',300)
        meta=read(dest.parent/'prediction_manifest.json')
        require(meta['prediction_count']==p['holdout_rows'] and not meta['test_labels_read'],'Table delivery check failed')
        exports.append(sha(dest))
    require(exports[0]==exports[1] and sha(native/'selected_model.joblib')==frozen['model'],'Table predictions or frozen model differ')
    save(a.out/'exports.json',dict(frozen=sha(arm/'frozen.json'),predictions=exports))
    labels=task.y_train[mask]
    yhat=read_predictions(arm/'export_0/rows.csv')
    result=dict(correct=int(np.sum(yhat==labels)),rows=len(yhat),groups=len(set(groups[mask])),
                train_majority_correct=int(np.sum(labels==p['majority'])),selected=frozen['selected'],
                distinct_fitted_classifiers=sorted({v['classifier'] for v in observed}),actual_fits=len(observed),
                model_api_calls=0,cost_rmb=0,course_validation_accessed=False,formal_test_accessed=False,
                proof={k:sha(a.out/k) for k in ('protocol.json','decision_inputs.json','decision_audit.json','exports.json')})
    save(a.out/'results.json',result)
    print('table',result['correct'], '/',len(yhat),frozen['selected'],len(observed),'fits',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('training','out','weights'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
