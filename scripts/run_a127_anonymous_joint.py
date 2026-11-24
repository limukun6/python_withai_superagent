"""Full-menu autonomous joint scheduler run with physically observed fits and portable prediction."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_b1_05_pair_pilot import pair_package
from scripts.run_a122_current_agent_acceptance import environment,physical,child

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--starting-A-fits',type=int,default=1538);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A127',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A127_anonymous_joint_run_plan_20261123_zh.md'),
        maximum_classifier_fits=10,real_API_requests=0,synthetic_engineering_only=True,starting_A_fits=a.starting_A_fits))
    try:
        if read(a.assets/'completion.json')['status']!='ready':raise ValueError('E5 assets not ready')
        env=environment(a.output)
        # The shared environment helper belongs to this checkout; worker imports must too.
        env['SUPERAGENT_E5_SMALL_WEIGHTS']=str(a.assets/'e5-small-v2')
        env['AB_A29_RUN_NAME']='anonymous_joint'
        rows=[];labels=[]
        for i in range(64):
            good=f'case {i} useful precise account about a bright river bridge'
            weak=f'case {i} vague confusing account about a dull river bridge'
            rows.append((good,weak) if i%2 else (weak,good));labels.append(i%2)
        query=[(f'new {i} clear detailed explanation of a river bridge',f'new {i} vague explanation of a river bridge') for i in range(12)]
        pair_package(a.output/'anonymous.zip',rows,labels,'preference','train')
        pair_package(a.output/'queries.zip',query,None,'preference','test')
        command=[sys.executable,'-m','superagent','run',str(a.output/'anonymous.zip'),'--run-dir',str(a.output/'run'),
            '--planner','local','--scheduler','ab-adaptive-complement','--portfolio','ab-v1',
            '--preference-catalog','deep-e5-familiarity','--cover-untried-encoder','--fit-limit','10',
            '--seed','18643','--time-limit','1200','--trial-timeout','300','--partial-delivery','checkpoint']
        save(a.output/'command.json',command)
        if a.validate_only:
            import os
            os.environ.update(env)
            from scripts import a38_predict_guard
            from superagent.ab_sentence import ALTERNATIVES,verified_files
            if verified_files(ALTERNATIVES['e5']['env'],ALTERNATIVES['e5']['files']) is None:raise ValueError('E5 verification failed')
            save(a.output/'completion.json',dict(status='validated',actual_classifier_fits=0,real_API_requests=0));return
        training=child(command,a.output/'train.log',env,1260)
        run=a.output/'run';outcome=read(run/'run_outcome.json');manifest=read(run/'run_manifest.json')
        trace=physical(a.output);count=len(trace);selection=read(run/'final_selection.json')
        if (not outcome['requested_planner_completed'] or outcome['new_api_requests'] or
            count!=outcome['classifier_fit_accounting']['attempted_classifier_fits'] or not 2<=count<=10 or
            not manifest['complement_active'] or not manifest['adaptive_opening_active']):
            raise ValueError('Joint path failed/count mismatch or controller not active')
        if sha(run/selection['model_file'])!=selection['model_sha256']:raise ValueError('Selected model changed')
        features=str(a.output/'queries.zip')
        child([sys.executable,'-m','scripts.a38_predict_guard',features,'--run-dir',str(run),'--output',str(a.output/'native.csv')],a.output/'native.log',env)
        moved=a.output/'relocated';moved.mkdir()
        for filename in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(run/filename,moved/filename)
        child([sys.executable,'-m','scripts.a38_predict_guard',features,'--run-dir',str(moved),'--output',str(a.output/'relocated.csv')],a.output/'relocated.log',env)
        child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(a.output/'portable')],a.output/'bundle.log',env)
        bundle=a.output/'portable/bundle'
        child([sys.executable,'-I',str(bundle/'predict.py'),'--features',features,'--output',str(a.output/'isolated.csv')],a.output/'isolated.log',env)
        expected=(a.output/'native.csv').read_bytes()
        if expected!=(a.output/'relocated.csv').read_bytes() or expected!=(a.output/'isolated.csv').read_bytes():raise ValueError('Portable predictions differ')
        with (a.output/'native.csv').open(encoding='utf-8',newline='') as stream:predictions=list(csv.DictReader(stream))
        if len(predictions)!=12 or [int(r['row_id']) for r in predictions]!=list(range(12)):raise ValueError('Query row identity differs')
        history=read(run/'history.json');resources=read(bundle/'bundle_manifest.json')['external_resources']
        result=dict(actual_classifier_fits=count,selection=selection,controller_active=True,
            menu_count=len(manifest['capabilities']),tried_candidates=[r['candidate'] for r in history],
            e5_tried=any('e5' in r['candidate'] for r in history),outcome=outcome,
            native_relocated_portable_identical=True,rows=12,model_sha256=selection['model_sha256'],
            prediction_sha256=sha(a.output/'native.csv'),external_resources=resources,training_process=training,
            actual_ensemble_selected=outcome['finalization']['ensemble_selected'],source_unchanged=hashes(ROOT)==source)
        save(a.output/'results.json',result)
        if hashes(ROOT)!=source:raise ValueError('Source changed')
        save(a.output/'completion.json',dict(status='completed_engineering_run',actual_classifier_fits=count,real_API_requests=0,
            A_cumulative_fits_after=a.starting_A_fits+count,source_unchanged=True,elapsed_seconds=time.monotonic()-started))
        print(json.dumps({k:result[k] for k in ('actual_classifier_fits','menu_count','e5_tried','native_relocated_portable_identical','actual_ensemble_selected')}),flush=True)
    except Exception as error:
        count=len(physical(a.output));save(a.output/'completion.json',dict(status='stopped',error=str(error),
            actual_classifier_fits=count,real_API_requests=0,A_cumulative_fits_after=a.starting_A_fits+count,automatic_retry=False));raise


if __name__=='__main__':main()
