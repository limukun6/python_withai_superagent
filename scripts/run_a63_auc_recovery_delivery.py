"""Reuse fitted AUC models for fault rejection and isolated source-bundle predictions; zero fits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
from scripts import a38_predict_guard
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch
from superagent.ab_run import sources
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.partial_delivery import freeze_checkpoint
from tools.export_model_bundle import export


class AlterProbabilities:
    """Controlled fault retains every hard prediction, reverses probability semantics."""
    def __init__(self,base):self.base=base;self.classes_=base.classes_
    def predict(self,x):return self.base.predict(x)
    def predict_proba(self,x):return self.base.predict_proba(x)[:,::-1]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources()
    run=a.reference/'sms_auc';manifest=read(run/'run_manifest.json');original_hash=sha(run/'run_outcome.json')
    save(root/'protocol.json',dict(batch='A63',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A63_auc_recovery_delivery_plan.md'),
        classifier_fitting_forbidden=True,network_forbidden=True,formal_test_or_H_opened=False))
    rows=[]
    try:
        if manifest['source_files']!=frozen:raise ValueError('Need exact current AUC training source')
        task=load_training_task(a.reference/'auc_training.zip');state=read(run/'state_3.json');caps=manifest['capabilities']
        trusted={}
        for split in ('primary','confirmation_1','confirmation_2'):
            path=run/(split+'.joblib');payload=joblib.load(path)
            trusted[split]=dict(row_ids=read(run/(split+'_split.json'))['search_indices'],
                labels=payload['y_search'].tolist(),evaluation_payload_sha256=sha(path))
        modes=('valid','missing_score','changed_scores','changed_columns','changed_metric','changed_label','changed_probability_model')
        for mode in modes:
            directory=root/mode;directory.mkdir();active=deepcopy(state)
            for pattern in ('trial_*.joblib','trial_*.fits.json','primary.joblib','run_manifest.json'):
                for path in run.glob(pattern):shutil.copy2(path,directory/path.name)
            chosen=sorted(active['history'],key=lambda r:(-r['search_score'],r['candidate']))[0]
            evidence=next(p for p in active['prediction_evidence'] if p['trial']==chosen['trial'])
            score=evidence['score_record']
            if mode=='missing_score':del evidence['score_record']
            elif mode=='changed_scores':score['scores'][0]=score['scores'][0][::-1]
            elif mode=='changed_columns':score['class_order']=score['class_order'][::-1]
            elif mode=='changed_metric':score['metric_spec']['name']='accuracy'
            elif mode=='changed_label':score['label_sha256']='0'*64
            elif mode=='changed_probability_model':
                path=directory/f"trial_{chosen['trial']}.joblib";bundle=joblib.load(path)
                bundle['model']=AlterProbabilities(bundle['model']);joblib.dump(bundle,path)
            try:
                result=freeze_checkpoint(task,active,trusted,caps,directory,RunLogger(directory),'controlled AUC failure')
            except ValueError as error:
                if mode=='valid':raise
                if (directory/'final_selection.json').exists():raise ValueError('Invalid score evidence produced delivery')
                rows.append(dict(case=mode,status='rejected',error=str(error)))
            else:
                if mode!='valid':raise ValueError('Corrupted AUC input accepted: '+mode)
                rows.append(dict(case=mode,status='accepted_partial',recovery=result))
            save(root/'fault_results.json',rows)
        # Partial and complete packages use the exact recorded project source; no training in export.
        delivery=[];fresh=root/'independent_cwd';fresh.mkdir()
        for name,source_run in (('full',run),('partial',root/'valid')):
            bundle=export(source_run,root/(name+'_package'))
            destination=root/(name+'_isolated.csv')
            with (root/(name+'_isolated.log')).open('w',encoding='utf-8') as log:
                code=subprocess.run([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(a.reference/'evaluation/features.zip'),
                           '--output',str(destination)],cwd=fresh,stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
            if code:raise RuntimeError('AUC isolated prediction failed: '+name)
            # Run a second process on the original run and compare CSV probabilities byte for byte.
            local=root/(name+'_original.csv')
            code,_=launch([sys.executable,'-m','scripts.a38_predict_guard',str(a.reference/'evaluation/features.zip'),
                '--run-dir',str(source_run),'--output',str(local)],root/(name+'_original.log'),None,60)
            if code or local.read_bytes()!=destination.read_bytes():raise ValueError('Isolated AUC probability CSV differs')
            delivery.append(dict(case=name,probability_csv_identical=True,source_from_bundle=True,
                prediction_rows=sum(1 for _ in destination.open(encoding='utf-8'))-1,
                delivery_status=read(source_run/'final_selection.json').get('delivery_status','complete'),
                classifier_fits=0,real_api_requests=0))
            save(root/'delivery_results.json',delivery)
        if sha(run/'run_outcome.json')!=original_hash or sources()!=frozen:raise ValueError('Original input/source changed')
        save(root/'completion.json',dict(status='passed',valid_partial_cases=1,corruption_cases_rejected=6,
            isolated_deliveries=len(delivery),actual_classifier_fits=0,real_api_requests=0,
            original_run_modified=False,source_unchanged=True,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_api_requests=0,
            fault_cases_completed=len(rows),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
