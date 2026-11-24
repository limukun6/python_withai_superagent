"""Summarize completed paired development runs; no training or model selection."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from scripts.a_batch_common import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    completion=read(args.run/'completion.json')
    if completion['status']!='passed':raise ValueError('Cannot summarize an incomplete comparison')
    results=read(args.run/'results.json')
    summary=[]
    args.output.mkdir(parents=True,exist_ok=False)
    for task in ('bean','sms','digits'):
        pair={r['scheduler']:r for r in results if r['task']==task}
        fixed,dynamic=(pair[k] for k in ('ab-fixed','ab-v1'))
        rows={}
        for label,result in [('fixed',fixed),('dynamic',dynamic)]:
            outcome=result['outcome']
            rows[label]=dict(score=result['verification']['score'],primary_metric=result['verification']['primary_metric'],
                fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                cli_seconds=outcome['elapsed_seconds'],train_process_seconds=result['train_process_seconds'],
                export_seconds=result['export_seconds'],selected=outcome['selected_candidate'],
                action_counts=outcome['action_counts'],model_sha256=result['selection']['model_sha256'],
                migrated_prediction_identical=result['relocated_prediction_identical'])
            source=args.run/result['name'];dest=args.output/result['name'];dest.mkdir()
            for name in ('run_manifest.json','run_outcome.json','actions.json','classifier_fits.json',
                         'final_selection.json','export_verification.json','confirmation.json'):
                shutil.copy2(source/name,dest/name)
        predictions=[]
        for result in (fixed,dynamic):
            with (args.run/result['name']/'predictions/rows.csv').open(encoding='utf-8',newline='') as file:
                values=list(csv.DictReader(file))
            if [r['row_id'] for r in values]!=list(map(str,range(len(values)))):
                raise ValueError('Prediction row order differs')
            predictions.append(np.asarray([r['prediction'] for r in values]))
        # Models were all frozen and scored already; labels here serve descriptive analysis only.
        labels=np.asarray(joblib.load(args.prepared/task/'holdout.joblib')['y']).astype(str)
        f,d=predictions
        if len(f)!=len(labels) or len(d)!=len(labels):raise ValueError('Unaligned paired predictions')
        summary.append(dict(task=task,**rows,score_delta=rows['dynamic']['score']-rows['fixed']['score'],
            fit_delta=rows['dynamic']['fits']-rows['fixed']['fits'],
            cli_seconds_delta=rows['dynamic']['cli_seconds']-rows['fixed']['cli_seconds'],
            prediction_disagreements=int(np.sum(f!=d)),dynamic_only_correct=int(np.sum((d==labels)&(f!=labels))),
            fixed_only_correct=int(np.sum((f==labels)&(d!=labels))),
            interpretation='Single development partition; descriptive paired results, no significance test'))
    for fname in ('completion.json','protocol.json','fairness.json','all_models_frozen.json','environment.txt','physical_fits.jsonl'):
        shutil.copy2(args.run/fname,args.output/fname)
    result=dict(completion=completion,comparisons=summary)
    save(args.output/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
