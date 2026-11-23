"""One current-source Task1 autonomous path, then validation-only frozen delivery; no API/test."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import child,environment,physical

ROOT=Path(__file__).resolve().parents[1]
PREFIX='in_vehicle_coupon_recommendation/'
TRAIN=[PREFIX+n for n in ('C_train.npy','Readme.md','y_train.npy')]
VAL=[PREFIX+n for n in ('C_val.npy','y_val.npy')]


def registered_read(source,allowed,access):
    original=zipfile.ZipFile.open
    source=source.resolve()
    def guarded(archive,member,mode='r',*a,**kw):
        name=member.filename if isinstance(member,zipfile.ZipInfo) else member
        if Path(archive.filename).resolve()==source and mode=='r':
            if name not in allowed: raise PermissionError('Unregistered original member read: '+name)
            access.append(name)
        return original(archive,member,mode,*a,**kw)
    with patch.object(zipfile.ZipFile,'open',new=guarded),zipfile.ZipFile(source) as z:
        return {name:z.read(name) for name in allowed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();frozen=hashes(ROOT);env=environment(out);os.environ.update(env)
    save(out/'protocol.json',dict(batch='A135',source_files=frozen,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A133_A135_next_integration_and_task1_plan_20261123_zh.md'),
         original_archive=str(a.source),original_archive_sha256=sha(a.source),train_allowlist=TRAIN,
         validation_allowlist_after_freeze=VAL,formal_test_access_allowed=False,fit_cap=11,
         model_api_cap=0,starting_A_cumulative_fits=1588,seed=707,confidence_cascade=False,
         scope='Current-source autonomous delivery candidate; seen designated validation, not unseen H or final test'))
    accesses=[]
    try:
        from superagent.ab_run import registered_capabilities,freeze_partitions,sources
        from superagent.contracts import TaskSpec
        from superagent.data import load_training_task
        from superagent.metrics import MetricSpec
        original=registered_read(a.source,TRAIN,accesses)
        members={n:hashlib.sha256(b).hexdigest() for n,b in original.items()}
        previous=read(ROOT/'research/a94_portable_joint_inputs/results.json')['coupon']
        if members!=previous['training_members_sha256']: raise ValueError('Original registered train members differ')
        training=out/'anonymous_train.zip'
        with zipfile.ZipFile(training,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for n,b in original.items():z.writestr(n,b)
        task=load_training_task(training);spec=TaskSpec.from_task(task).record()
        if len(task.y_train)!=8117 or spec!=previous['task_spec']: raise ValueError('Task/metric/rows differ from registration')
        if getattr(task,'X_val',None) is not None or getattr(task,'y_val',None) is not None:
            raise ValueError('Unexpected validation in training copy')
        caps=registered_capabilities(spec,len(task.y_train),'full')
        prepared=out/'prepared';prepared.mkdir();freeze_partitions(task,prepared,707,MetricSpec.for_task(task))
        splits={n:sha(prepared/(n+'_split.json')) for n in ('primary','confirmation_1','confirmation_2')}
        save(out/'prepared.json',dict(training_sha256=sha(training),members_sha256=members,
             rows=len(task.y_train),task_spec=spec,capabilities=caps,split_sha256=splits))
        save(out/'source_access.json',dict(opened=accesses,model_frozen=False,formal_test_opened=False))
        run=out/'run';local={**env,'AB_A29_RUN_NAME':'released_task1_current'}
        command=[sys.executable,'-m','superagent','run',str(training),'--run-dir',str(run),
             '--planner','local','--scheduler','ab-fixed','--portfolio','ab-v1','--catalog-scope','full',
             '--preference-catalog','standard','--parameter-semantics','off','--fit-limit','11',
             '--seed','707','--trial-timeout','300','--time-limit','1200','--partial-delivery','checkpoint']
        save(out/'command.json',command)
        print(json.dumps(dict(stage='train_only_registered',rows=len(task.y_train),candidates=len(caps),fit_cap=11)),flush=True)
        timing=child(command,out/'train.log',local,1260)
        outcome=read(run/'run_outcome.json');manifest=read(run/'run_manifest.json');selection=read(run/'final_selection.json')
        counts=outcome['classifier_fit_accounting'];fits=physical(out)
        if (not outcome['requested_planner_completed'] or outcome['new_api_requests'] or
            len(fits)!=counts['attempted_classifier_fits'] or not 2<=len(fits)<=11 or
            counts['completed_classifier_fits']!=len(fits) or counts['unknown_failed_attempts']):
            raise ValueError('Autonomous completion or physical fit accounting failed')
        if manifest['source_files']!=sources() or manifest['capabilities']!=caps: raise ValueError('Source/menu drift')
        for n,d in splits.items():
            if sha(run/(n+'_split.json'))!=d: raise ValueError('Registered row partition drift')
        model=run/selection['model_file']
        if sha(model)!=selection['model_sha256']: raise ValueError('Model digest differs')
        save(out/'model_frozen.json',dict(model_sha256=sha(model),selection_sha256=sha(run/'final_selection.json'),
             source_files=frozen,physical_fits=len(fits),validation_accessed=False,formal_test_accessed=False))
        print(json.dumps(dict(stage='model_frozen',fits=len(fits),selected=selection['family'])),flush=True)
        validation=registered_read(a.source,VAL,accesses)
        x=np.load(io.BytesIO(validation[VAL[0]]),allow_pickle=True)
        y=np.load(io.BytesIO(validation[VAL[1]]),allow_pickle=True).reshape(-1)
        if len(x)!=2030 or len(y)!=2030: raise ValueError('Unexpected designated validation row count')
        query=out/'validation_features.zip'
        with zipfile.ZipFile(query,'x') as z:z.writestr('C_test.npy',validation[VAL[0]])
        joblib.dump(dict(X=x,y=y,row_ids=list(range(len(y)))),out/'validation.joblib')
        save(out/'source_access.json',dict(opened=accesses,model_frozen=True,formal_test_opened=False,
             validation_members_sha256={n:hashlib.sha256(b).hexdigest() for n,b in validation.items()},
             query_note='C_test.npy in derivative query contains only C_val.npy bytes'))
        child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(run),
               '--output',str(out/'native.csv')],out/'native.log',local)
        moved=out/'relocated';moved.mkdir()
        for n in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(run/n,moved/n)
        child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(moved),
               '--output',str(out/'relocated.csv')],out/'relocated.log',local)
        child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(out/'portable')],out/'bundle.log',local)
        child([sys.executable,'-I',str(out/'portable/bundle/predict.py'),'--features',str(query),
               '--output',str(out/'isolated.csv')],out/'isolated.log',local)
        if len({(out/n).read_bytes() for n in ('native.csv','relocated.csv','isolated.csv')})!=1:
            raise ValueError('Independent prediction bytes differ')
        # Reload and score with training/network disabled; validation cannot affect selection above.
        guard=ROOT/'scripts/a38_predict_guard.py'
        scorecode="import runpy,sys;runpy.run_path(sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('scripts.a07_verify_export',run_name='__main__')"
        child([sys.executable,'-c',scorecode,str(guard),'a07','--source-root',str(ROOT),'--run-dir',str(run),
               '--holdout',str(out/'validation.joblib'),'--predictions',str(out/'native.csv'),
               '--output',str(out/'verification.json')],out/'verify.log',local)
        verification=read(out/'verification.json')
        verification['scope']='Seen course-designated validation after current-source model freeze, not unseen transfer or official test'
        save(out/'verification.json',verification)
        with (out/'native.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        if [int(r['row_id']) for r in rows]!=list(range(2030)): raise ValueError('Row order differs')
        if hashes(ROOT)!=frozen or len(physical(out))!=len(fits) or sha(model)!=selection['model_sha256']:
            raise ValueError('Post-export source/fit/model drift')
        save(out/'results.json',dict(training=timing,actual_classifier_fits=len(fits),selected_candidate=selection['family'],
             model_sha256=sha(model),outcome=outcome,verification=verification,rows=len(rows),
             native_relocated_portable_identical=True,prediction_sha256=sha(out/'native.csv'),
             external_resources=read(out/'portable/bundle/bundle_manifest.json')['external_resources'],
             final_submission_frozen=False,formal_test_opened=False))
        save(out/'completion.json',dict(status='passed',actual_classifier_fits=len(fits),real_API_requests=0,
             A_cumulative_fits_after=1588+len(fits),elapsed_seconds=time.monotonic()-started,
             source_unchanged=True,original_formal_test_opened=False,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps(dict(status='passed',fits=len(fits),validation_accuracy=verification['accuracy'])),flush=True)
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=len(physical(out)),
             A_cumulative_fits_after=1588+len(physical(out)),real_API_requests=0,
             elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
