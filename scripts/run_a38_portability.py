"""Move copied frozen model packages and compare predictions, never read scoring labels."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True);parser.add_argument('--transfer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    save(root/'protocol.json',dict(batch='A38-portability',source_files=frozen,actual_classifier_fit_limit=0,api_limit=0,
        runner_sha256=sha(Path(__file__)),guard_sha256=sha(Path(__file__).with_name('a38_predict_guard.py')),
        holdout_labels_opened=False,plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A38_portable_delivery_plan.md')))
    try:
        for parent in (args.matrix,args.transfer):
            if read(parent/'completion.json')['status']!='completed' or read(parent/'protocol.json')['source_files']!=frozen:
                raise ValueError('Completed same-source studies required')
        for name in ('coupon','preference','sms','photo'):
            parent=args.matrix if name in ('coupon','preference') else args.transfer
            arm='R0_routes' if name in ('coupon','preference') else 'R1_routes'
            original=parent/(name+'_'+arm);moved=root/name;moved.mkdir()
            selected=read(original/'final_selection.json')
            if sha(original/selected['model_file'])!=selected['model_sha256']:raise ValueError('Original model hash differs')
            for file in ('task_spec.json','final_selection.json',selected['model_file']):shutil.copy2(original/file,moved/file)
            prediction=moved/'predictions/rows.csv'
            cmd=[sys.executable,'-m','scripts.a38_predict_guard',str(parent/'evaluation'/name/'features.zip'),
                '--run-dir',str(moved),'--output',str(prediction)]
            with (root/(name+'.log')).open('w',encoding='utf-8') as log:
                code=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
            if code:raise RuntimeError('Guarded portable prediction failed: '+name)
            original_csv=original/'predictions/rows.csv'
            if prediction.read_bytes()!=original_csv.read_bytes():raise ValueError('Portable predictions differ: '+name)
            with prediction.open(encoding='utf-8',newline='') as file:count=sum(1 for _ in csv.DictReader(file))
            results.append(dict(task=name,arm=arm,model_sha256=selected['model_sha256'],prediction_sha256=sha(prediction),
                rows=count,relocated_predictions_identical=True,training_forbidden=True,network_forbidden=True,labels_scored=False))
            save(root/'results.json',results)
        save(root/'completion.json',dict(status='passed',models=len(results),actual_classifier_fits=0,real_api_requests=0,
            holdout_labels_opened=False,source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start))
        print(json.dumps(dict(status='passed',models=len(results),classifier_fits=0,API=0)))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),models_verified=len(results),automatic_retry=False,
            real_api_requests=0,actual_classifier_fits=0));raise
