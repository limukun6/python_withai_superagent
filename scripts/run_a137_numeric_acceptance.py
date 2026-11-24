"""Bounded anonymous numeric integration acceptance: maximum three actual classifier fits."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import child,environment,physical
from scripts.run_a133_routing_verification import GUARD

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
MENU_CODE='''import json
from superagent.ab_run import registered_capabilities
result={}
for kind in ('numeric_tabular','categorical_tabular','single_text','mixed_tabular','image','preference','symmetric'):
 pair=kind in ('preference','symmetric')
 schema=({'pair_semantics':kind,'pair_label_one':'left' if kind=='preference' else 'equivalent'} if pair else {})
 spec=dict(contract_version=1,objective='classification',kind='pair_text' if pair else kind,input_schema=schema,
           metric=dict(name='accuracy',labels=[0,1],direction='maximize'))
 result[kind]=registered_capabilities(spec,240)
print(json.dumps(result,sort_keys=True))
'''


def package(path,x,y=None):
    split='train' if y is not None else 'test';buffer=io.BytesIO()
    np.savez(buffer,**({'X':x,'y':y} if y is not None else {'X':x}))
    manifest=dict(schema_version=1,objective='classification',modality='tabular',metric='accuracy',
                  splits={split:dict(format='npz',path='rows.npz')})
    with zipfile.ZipFile(path,'x') as z:
        z.writestr('manifest.json',json.dumps(manifest));z.writestr('rows.npz',buffer.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    env=environment(out);env['AB_A29_RUN_NAME']='anonymous_numeric';os.environ.update(env)
    ledger=Path(read(ROOT/'research/a133_a135_summary/current_ledger_owner.json')['ledger_path']);ledger_sha=sha(ledger)
    save(out/'protocol.json',dict(batch='A137',source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A137_numeric_adapter_plan_20261124_zh.md'),
         provider='5bd6b145e63ca59af005aab6cf1e9cd24ff21931',starting_A_fits=1599,fit_cap=3,
         model_API_cap=0,training_rows=240,query_rows=40,synthetic_seed=13724,seed=707,
         scope='Anonymous synthetic autonomous integration; not real-task quality or unseen transfer',ledger_sha256=ledger_sha))
    try:
        # Compare complete default menus under identical assets, without loading any task data.
        before=BASE/'runs/a133_confidence_checkout'
        baseline=json.loads(subprocess.check_output([sys.executable,'-c',MENU_CODE],cwd=before,
                            env={**env,'PYTHONPATH':str(before)},text=True))
        current=json.loads(subprocess.check_output([sys.executable,'-c',MENU_CODE],cwd=ROOT,env=env,text=True))
        if baseline!=current:raise ValueError('One of seven default menus changed')
        save(out/'default_menus.json',dict(default_menus_unchanged=True,menus=current))
        tests=['test_a137_numeric_kernel','test_ab_encoder_routing','test_ab_complement',
               'test_ab_registered_rows','test_ab_confirmation_budget','test_a126_registered_options',
               'test_a120_parameter_semantics','test_task_budget','test_a130_accepted_history']
        testcode=GUARD.split('target=sys.argv[1]')[0]+"\nimport unittest\ns=unittest.TestLoader().loadTestsFromNames("+repr(['tests.'+t for t in tests])+")\nr=unittest.TextTestRunner(verbosity=2).run(s)\nraise SystemExit(not r.wasSuccessful())\n"
        with (out/'tests.log').open('x',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-c',testcode],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if proc.returncode:raise RuntimeError('Pretraining tests failed; no supervised training launched')
        rng=np.random.default_rng(13724);x=rng.normal(size=(240,3));y=(x[:,0]*x[:,1]+.3*x[:,2]>0).astype(int)
        query=rng.normal(size=(40,3));package(out/'train.zip',x,y);package(out/'query.zip',query)
        command=[sys.executable,'-m','superagent','run',str(out/'train.zip'),'--run-dir',str(out/'run'),
            '--planner','local','--scheduler','ab-v1','--portfolio','ab-v1','--catalog-scope','full',
            '--numeric-kernel','--fit-limit','3','--seed','707','--trial-timeout','60','--time-limit','180']
        save(out/'command.json',command)
        training=child(command,out/'train.log',env,240)
        run=out/'run';outcome=read(run/'run_outcome.json');manifest=read(run/'run_manifest.json')
        selection=read(run/'final_selection.json');fits=physical(out)
        from superagent.numeric_kernel import NAME
        history=read(run/'history.json');attempted=[h['candidate'] for h in history]
        if NAME not in manifest['capabilities'] or manifest['numeric_kernel_width']!=3:
            raise ValueError('Explicit numeric capability not visible to executor')
        if (not outcome['requested_planner_completed'] or outcome['new_api_requests'] or not 2<=len(fits)<=3 or
            outcome['classifier_fit_accounting']['attempted_classifier_fits']!=len(fits) or
            outcome['classifier_fit_accounting']['completed_classifier_fits']!=len(fits)):
            raise ValueError('Autonomy or physical fit accounting failed')
        child([sys.executable,'-m','scripts.a38_predict_guard',str(out/'query.zip'),'--run-dir',str(run),
               '--output',str(out/'native.csv')],out/'native.log',env)
        moved=out/'relocated';moved.mkdir()
        for name in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(run/name,moved/name)
        child([sys.executable,'-m','scripts.a38_predict_guard',str(out/'query.zip'),'--run-dir',str(moved),
               '--output',str(out/'relocated.csv')],out/'relocated.log',env)
        child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(out/'portable')],out/'bundle.log',env)
        child([sys.executable,'-I',str(out/'portable/bundle/predict.py'),'--features',str(out/'query.zip'),
               '--output',str(out/'isolated.csv')],out/'isolated.log',env)
        if len({(out/n).read_bytes() for n in ('native.csv','relocated.csv','isolated.csv')})!=1:
            raise ValueError('Three process prediction bytes differ')
        if source!=hashes(ROOT) or sha(ledger)!=ledger_sha or len(physical(out))!=len(fits):
            raise ValueError('Source, ledger or fit count changed after prediction')
        save(out/'results.json',dict(training=training,actual_classifier_fits=len(fits),attempted_candidates=attempted,
             numeric_kernel_attempted=NAME in attempted,selected=selection['family'],
             selected_numeric_kernel=selection['family']==NAME,default_menu_count=len(current),
             three_prediction_processes_identical=True,query_labels_scored=False,
             model_sha256=selection['model_sha256'],prediction_sha256=sha(out/'native.csv')))
        save(out/'completion.json',dict(status='passed',actual_classifier_fits=len(fits),real_API_requests=0,
             A_cumulative_fits_after=1599+len(fits),source_unchanged=True,ledger_unchanged=True,
             elapsed_seconds=time.monotonic()-started,protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps(read(out/'results.json')))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=len(physical(out)),real_API_requests=0,
             A_cumulative_fits_after=1599+len(physical(out)),elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
