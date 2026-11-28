"""Freeze two training-only packages, run the same solver and seal both outputs before validation."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, env, make_zip, read, require, save, sha, title_rows

CASES = {'a':('table',['C_train.npy','y_train.npy','Readme.md']), 'b':('titles',['train.csv','README.md'])}
PLAN = ROOT/'research/unified_environment_freeze_1128/protocol.json'


def prepare(args):
    weights = args.out/'weights'
    weights.mkdir(exist_ok=False)
    for path in sorted(args.library.iterdir()):
        if path.is_dir() and not path.name.startswith('.'):
            (weights/path.name).symlink_to(path.resolve(),target_is_directory=True)
    (weights/args.table_weight.name).symlink_to(args.table_weight)
    records={}
    for key,(tag,allowed) in CASES.items():
        folder=args.out/key
        folder.mkdir()
        with zipfile.ZipFile(archive(args.packages,tag)) as source:
            permitted=[n for n in source.namelist() if '__MACOSX' not in n and Path(n).name in allowed]
            require(sorted(map(lambda s:Path(s).name,permitted))==sorted(allowed),'Unexpected training layout')
            make_zip(folder/'training.zip',[('input/'+Path(n).name,source.read(n)) for n in permitted])
        shutil.copyfile(args.table_features if key=='a' else args.text_features,folder/'queries.zip')
        records[key]=dict(training=sha(folder/'training.zip'),queries=sha(folder/'queries.zip'),
                          opened_source_members=permitted,permitted=['input/'+n for n in allowed])
    save(args.out/'inputs.json',dict(protocol=sha(PLAN),solver=source_identity(ROOT),packets=records))


def run(args):
    key=args.case
    root=args.out/key
    run_dir=root/'run'
    inputs=read(args.out/'inputs.json')
    record=inputs['packets'][key]
    require(source_identity(ROOT)==inputs['solver'] and sha(root/'training.zip')==record['training']
            and sha(root/'queries.zip')==record['queries'],'Frozen inputs or solver changed')
    settings=env(args.out/'weights',root/'physical.jsonl')
    for name in list(settings):
        if name.startswith('SUPERAGENT_') or name in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO','PYTORCH_MPS_LOW_WATERMARK_RATIO'):
            del settings[name]
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT,args.runtime))),
                    SUPERAGENT_OFFLINE='1',VECLIB_MAXIMUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                    UNIFIED_RAW_INPUT=str(root/'training.zip'),UNIFIED_RAW_MEMBERS=json.dumps(record['permitted']),
                    UNIFIED_RAW_LOG=str(root/'opened.jsonl'))
    for phase in ('solve','reload'):
        opts=(['solve',root/'training.zip','--prediction-input',root/'queries.zip','--fit-limit','64','--time-limit','1800']
              if phase=='solve' else ['predict',root/'queries.zip'])
        with (root/(phase+'.log')).open('x') as log:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,opts),'--weights-dir',str(args.out/'weights'),
                            '--run-dir',str(run_dir),'--output',str(root/(phase+'.csv'))],
                           env=settings,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2030)
    native=run_dir/'agent'
    result=read(native/'run_outcome.json')
    selected=read(native/'final_selection.json')
    count=len((root/'physical.jsonl').read_text().splitlines())
    require(result['status']=='completed' and result['new_api_requests']==0 and not result['formal_test_accessed']
            and selected['refit'] and 0<count==result['attempted_classifier_fits']<=64 and not result['unknown_failed_attempts']
            and sha(root/'solve.csv')==sha(root/'reload.csv') and selected['model_sha256']==sha(native/selected['model_file']),
            'Delivery, fit ledger or independent replay failed')
    with (root/'solve.csv').open() as stream:
        predictions=list(csv.DictReader(stream))
    require([r['row_id'] for r in predictions]==list(map(str,range(len(predictions)))),'Wrong row ordering')
    opened=sorted({json.loads(l) for l in (root/'opened.jsonl').read_text().splitlines()})
    require(set(opened)==set(record['permitted']),'Unexpected raw input')
    feedback={}
    for file in ('cell_encoder_review','seed_replication','analogy_review','portfolio_review','finalist_race'):
        if (native/(file+'.json')).is_file():
            doc=read(native/(file+'.json'))
            feedback[file]=dict(sha256=sha(native/(file+'.json')),action=doc.get('action'),reason=doc.get('reason'))
    save(root/'sealed.json',dict(solver=inputs['solver'],model=selected['model_sha256'],spec=selected['task_spec_sha256'],
         predictions=sha(root/'solve.csv'),rows=len(predictions),training_rows=selected['training_rows'],family=selected['family'],
         members=result['members'],fits=count,completed_fits=result['completed_classifier_fits'],seconds=result['wall_seconds'],
         new_model_api=0,opened=opened,feedback=feedback,independent_reload=True,course_test_opened=False))
    print(key,result['members'],count,len(predictions),flush=True)


def grade(args):
    seals={k:read(args.out/k/'sealed.json') for k in CASES}
    frozen=read(args.out/'inputs.json')['solver']
    require(all(s['solver']==frozen and s['predictions']==sha(args.out/k/'reload.csv')==sha(args.out/k/'solve.csv')
                for k,s in seals.items()),'Unsealed export')
    save(args.out/'freeze.json',dict(solver=frozen,protocol=sha(PLAN),exports=seals))
    result={}
    references={'a':('080545da199beff0eb22ebf1ac2d3e9705f77bc4c236877bd9ec8989f5cbb5f8',1561),
                'b':('efdd8eeb5ebd9c7016c698085376bc4ae2b971859feb7105f13e21d8c2c27a77',710)}
    for k,(tag,_) in CASES.items():
        sealed=seals[k]
        digest,previous=references[k]
        if sealed['predictions']==digest:
            correct=previous
            source='byte-identical to prior scored validation prediction'
        else:
            path=archive(args.packages,tag)
            if k=='a':
                with zipfile.ZipFile(path) as z:
                    member,=[n for n in z.namelist() if '__MACOSX' not in n and Path(n).name=='y_val.npy']
                    truth=np.load(io.BytesIO(z.read(member)),allow_pickle=False).reshape(-1)
            else:
                truth=np.asarray([y for _,_,y in title_rows(path,'val',labels=True)])
            with (args.out/k/'solve.csv').open() as stream:
                pred=np.asarray([r['prediction'] for r in csv.DictReader(stream)])
            require(len(truth)==len(pred)==sealed['rows'],'Validation ordering mismatch')
            correct=int(np.sum(pred==truth.astype(str)))
            source='one designated validation score after both models and predictions were frozen'
        result[k]=dict(correct=correct,rows=sealed['rows'],accuracy=correct/sealed['rows'],basis=source,
                       previous_correct=previous,test_labels_read=False)
    save(args.out/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','grade'])
    p.add_argument('--case',choices=tuple(CASES))
    for name in ('out','library','table-weight','packages','table-features','text-features','runtime'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=name=='out')
    args=p.parse_args()
    {'prepare':prepare,'run':run,'grade':grade}[args.action](args)
