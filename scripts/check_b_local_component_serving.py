"""Preflight the CLI and an over-budget rejection, using no training."""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from superagent.component_preferences import ComponentModel,sources,worker_env
from superagent import ab_run
from scripts.run_b_local_component_pair import ROOT, SHAPE, packets, save, command, check


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-fixture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp)
        make_fixture="""
import json,os,sys,torch,joblib
from pathlib import Path
sys.path.insert(0,os.environ['SUPERAGENT_B_COMPONENT_RUNTIME'])
folder,out=map(Path,sys.argv[1:])
meta=json.loads((folder/'manifest.json').read_text())
from superagent.pretrained_choice import digest
for k,v in meta['components'].items(): assert digest(folder/k)==v
components={k:joblib.load(folder/(k+'.joblib')) for k in ('base','deep')}
components['residual']={'weights':torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)}
joblib.dump(dict(components=components,selected='stream_hybrid',threshold=meta['threshold']),out,compress=3)
"""
        child_env=worker_env()
        subprocess.run([sys.executable,'-c',make_fixture,str(a.old_fixture.resolve()),str(d/'package.joblib')],
                       cwd=d,env=child_env,check=True)
        run=d/'run'
        run.mkdir()
        family='existing_fixture_for_inference_only'
        pipeline=Pipeline([('preference',ComponentModel((d/'package.joblib').read_bytes(),sources()))])
        joblib.dump(dict(model=pipeline,kind='pair_text',candidate=family),run/'selected_model.joblib')
        save(run/'task_spec.json',dict(task_kind='pair_text_binary',model_kind='pair_text',metric='accuracy',
              metric_spec={'name':'accuracy','labels':[0,1],'direction':'maximize'},input_schema=SHAPE))
        save(run/'final_selection.json',dict(selected_experiment='zero_fit_inference_fixture',family=family,
             model_file='selected_model.joblib',model_format='candidate_bundle_v1',test_accessed=False,
             model_sha256=ab_run.sha(run/'selected_model.joblib'),task_spec_sha256=ab_run.sha(run/'task_spec.json')))
        rows=[('Bright morning news','Later on the same day',0),('Later on the same day','Bright morning news',0),
              ('Bright morning news','Later on the same day',0)]
        packets(rows,[0,1,2],False,d/'unlabeled.zip')
        rc,_=command(['predict',str(d/'unlabeled.zip'),'--run-dir',str(run),'--output',str(d/'rows.csv')],
                     d/'log',worker_env(),150)
        if rc:
            raise RuntimeError((d/'log').read_text()[-2500:])
        with (d/'rows.csv').open() as f:
            results=list(csv.DictReader(f))
        check(len(results)==3 and results[0]['prediction']==results[2]['prediction'],'Serving preflight failed')
        meter_check="""
import sys,tempfile
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from b_component_worker import FitMeter
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp);m=FitMeter(p,2,20,10)
 try: m.run('blocked','logistic',lambda:None,reserve=2)
 except RuntimeError: pass
 else: raise ValueError('Over-budget operation was admitted')
 assert m.attempts==0 and (p/'component_fits.jsonl').stat().st_size==0
"""
        subprocess.run([sys.executable,'-c',meter_check,str(ROOT/'scripts')],env=child_env,cwd=d,check=True)
    save(a.out,dict(status='passed',anonymous_unlabeled_rows=3,normal_predict_entry=True,
                    overbudget_denied_before_start=True,supervised_fits=0,api_calls=0,
                    historical_fixture_in_temporary_directory_only=True))
