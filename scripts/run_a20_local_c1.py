"""Five-task local C1 development matrix, no original test access or API calls."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts import run_b1_ab_acceptance as fixture
from scripts.run_a10_real_task_pilot import read,sha,launch
from superagent.ab_run import sources,registered_capabilities
from superagent.contracts import TaskSpec
from superagent.data import load_training_task,load_development_task


ORDER=[('coupon','L0'),('coupon','L1'),('preference','L1'),('preference','L0'),('bean','L0'),('bean','L1'),
       ('sms','L1'),('sms','L0'),('photo','L0'),('photo','L1')]


def train_only_copy(source,destination):
    with zipfile.ZipFile(source) as original,zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as target:
        copied=[]
        for name in original.namelist():
            if '__MACOSX' in name:continue
            base=Path(name).name.casefold()
            if base in ('c_train.npy','y_train.npy','train.csv','readme.md','readme.txt'):
                data=original.read(name);target.writestr(name,data);copied.append(dict(member=name,bytes=len(data)))
        if not copied:raise ValueError('No training members copied')
    return copied


def external_inputs(entry,root,args):
    name=entry['task'];folder=root/'evaluation'/name;folder.mkdir(parents=True)
    if name in ('coupon','preference'):
        opened=[];original=zipfile.ZipFile.read
        def no_test(archive,member,*a,**kw):
            text=member.filename if isinstance(member,zipfile.ZipInfo) else member
            if 'test' in text.casefold():raise ValueError('Original formal test access is forbidden')
            opened.append(text);return original(archive,member,*a,**kw)
        with patch.object(zipfile.ZipFile,'read',new=no_test):task=load_development_task(entry['source_archive'])
        x,y=task.X_val,task.y_val
        if x is None or y is None:raise ValueError('No designated validation data')
        query=folder/'features.zip'
        with zipfile.ZipFile(query,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            if name=='coupon':
                buffer=io.BytesIO();np.save(buffer,x);archive.writestr('C_test.npy',buffer.getvalue())
            else:
                buffer=io.StringIO(newline='');writer=csv.writer(buffer);writer.writerow(['headline_A','headline_B'])
                writer.writerows(x);archive.writestr('test.csv',buffer.getvalue())
        save(folder/'source_access.json',dict(original_members_opened=opened,formal_test_opened=False,
            scope='Designated val loaded only after all ten models frozen; output test-named file holds val features'))
        held=dict(X=x,y=y,row_ids=list(range(len(y))))
    else:
        source=args.photo_prepared/'photo_dev' if name=='photo' else args.generic_prepared/name
        shutil.copy2(source/'features.zip',folder/'features.zip');held=joblib.load(source/'holdout.joblib')
        if name=='photo':held['y']=np.asarray(held['y']).astype(str)
    joblib.dump(held,folder/'holdout.joblib');return folder


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);fixture.offline(out)
    began=time.monotonic();frozen=sources();results=[]
    try:
        entries={e['task']:e for e in read(args.readiness/'readiness.json')}
        if read(args.readiness/'protocol.json')['source_files']!=frozen:raise ValueError('Readiness source differs')
        paths={};copied={}
        for name,e in entries.items():
            if e['task_gaps']:raise ValueError('Unresolved input gap: '+name)
            original=Path(e['source_archive'])
            if sha(original)!=e['task_archive_sha256']:raise ValueError('Task archive changed')
            if name in ('coupon','preference'):
                folder=out/'training_inputs'/name;folder.mkdir(parents=True)
                paths[name]=folder/'train.zip';copied[name]=train_only_copy(original,paths[name])
            else:paths[name]=original
            task=load_training_task(paths[name]);spec=TaskSpec.from_task(task).record()
            if spec!=e['task_spec'] or registered_capabilities(spec,len(task.y_train))!=e['capabilities']:
                raise ValueError('Train-only clone changed the task contract or catalogue: '+name)
        save(out/'protocol.json',dict(batch='A20-local-C1',source_files=frozen,
            runner_sha256=sha(Path(__file__)),plan_sha256=sha(fixture.ROOT/'docs/batches/A20_local_C1_plan.md'),
            readiness_protocol_sha256=sha(args.readiness/'protocol.json'),order=ORDER,
            train_only_members=copied,training_archives={n:dict(path=str(p),sha256=sha(p)) for n,p in paths.items()},
            entries=list(entries.values()),fit_limit_per_path=11,batch_fit_limit=110,api_limit=0,
            validate_only=args.validate_only,scope='Development local C1 submatrix, not final six-arm C1'))
        if args.validate_only:
            save(out/'completion.json',dict(status='validated',actual_classifier_fits=0,real_api_requests=0));return
        for name,arm in ORDER:
            if sources()!=frozen:raise ValueError('Production source changed during C1')
            directory=out/(name+'_'+arm);e=entries[name]
            command=[sys.executable,'-m','superagent','run',str(paths[name]),'--run-dir',str(directory),
                '--planner','local','--scheduler','ab-fixed' if arm=='L0' else 'ab-v1','--portfolio','ab-v1',
                '--catalog-scope','full','--fit-limit','11','--seed','707','--time-limit','300','--trial-timeout','60']
            save(out/(name+'_'+arm+'_command.json'),command)
            seconds=launch(command,out/(name+'_'+arm+'.log'),330)
            value=read(directory/'run_outcome.json');counts=value['classifier_fit_accounting']
            if not (value['requested_planner_completed'] and value['new_api_requests']==0 and not counts['unknown_failed_attempts']
                    and 2<=counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11):
                raise ValueError('C1 path did not complete within actual fit cap')
            manifest=read(directory/'run_manifest.json')
            if manifest['source_files']!=frozen or manifest['capabilities']!=e['capabilities']:raise ValueError('C1 source/menu drift')
            for split in e['splits']:
                if read(directory/(split+'_split.json'))!=read(args.readiness/name/(split+'_split.json')):
                    raise ValueError('C1 actual registered split differs')
            selection=read(directory/'final_selection.json')
            if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Model freeze changed')
            results.append(dict(task=name,arm=arm,outcome=value,selection=selection,train_process_seconds=seconds,
                actions=read(directory/'actions.json')))
            save(out/'checkpoint.json',results)
            print(json.dumps(dict(task=name,arm=arm,status='model_frozen',fits=counts['attempted_classifier_fits'],
                seconds=seconds,selected=value['selected_candidate'])),flush=True)
        save(out/'all_models_frozen.json',[dict(task=r['task'],arm=r['arm'],selection=r['selection']) for r in results])
        evaluation={name:external_inputs(e,out,args) for name,e in entries.items()}
        for row in results:
            name,arm=row['task'],row['arm'];directory=out/(name+'_'+arm);evaluation_dir=evaluation[name]
            target=directory/'predictions/rows.csv'
            export=launch([sys.executable,'-m','superagent','predict',str(evaluation_dir/'features.zip'),
                '--run-dir',str(directory),'--output',str(target)],out/(name+'_'+arm+'_predict.log'),60)
            export+=launch([sys.executable,str(fixture.ROOT/'scripts/a07_verify_export.py'),'--source-root',str(fixture.ROOT),
                '--run-dir',str(directory),'--holdout',str(evaluation_dir/'holdout.joblib'),'--predictions',str(target),
                '--output',str(directory/'verification.json')],out/(name+'_'+arm+'_verify.log'),60)
            row.update(verification=read(directory/'verification.json'),export_seconds=export)
            save(out/'results.json',results)
        physical=len(fixture.trace(out));expected=sum(r['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for r in results)
        if physical!=expected or physical>110 or sources()!=frozen:raise ValueError('C1 physical accounting/source mismatch')
        save(out/'completion.json',dict(status='passed',paths_completed=len(results),actual_classifier_fits=physical,
            completed_classifier_fits=expected,real_api_requests=0,source_unchanged=True,
            original_formal_test_opened=False,elapsed_seconds=time.monotonic()-began))
    except Exception as error:
        save(out/'results.json',results);save(out/'completion.json',dict(status='failed',error=str(error),
            actual_classifier_fits=len(fixture.trace(out)),real_api_requests=0,elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--readiness',type=Path,required=True)
    p.add_argument('--generic-prepared',type=Path,required=True);p.add_argument('--photo-prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');run(p.parse_args())
