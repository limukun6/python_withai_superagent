"""Describe the five-task local C1 submatrix without any new training or requests."""
import argparse
import csv
from pathlib import Path
import shutil

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    completion=read(a.run/'completion.json')
    if completion['status']!='passed':raise ValueError('Cannot summarize incomplete local submatrix')
    results=read(a.run/'results.json');rows=[]
    for task in ('coupon','preference','bean','sms','photo'):
        pair={r['arm']:r for r in results if r['task']==task};arms={};vectors=[]
        for arm in ('L0','L1'):
            result=pair[arm];outcome=result['outcome'];directory=a.run/(task+'_'+arm)
            arms[arm]=dict(score=result['verification']['score'],primary_metric=result['verification']['primary_metric'],
                rows=result['verification']['rows'],fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                cli_seconds=outcome['elapsed_seconds'],export_seconds=result['export_seconds'],
                selected=outcome['selected_candidate'],action_counts=outcome['action_counts'])
            with (directory/'predictions/rows.csv').open(encoding='utf-8',newline='') as file:
                data=list(csv.DictReader(file))
            vectors.append(np.asarray([r['prediction'] for r in data]))
            target=a.output/(task+'_'+arm);target.mkdir()
            for name in ('run_manifest.json','run_outcome.json','actions.json','history.json',
                         'classifier_fits.json','final_selection.json','verification.json'):
                shutil.copy2(directory/name,target/name)
        labels=np.asarray(joblib.load(a.run/'evaluation'/task/'holdout.joblib')['y']).astype(str)
        old,new=vectors
        rows.append(dict(task=task,**arms,score_delta=arms['L1']['score']-arms['L0']['score'],
            fit_delta=arms['L1']['fits']-arms['L0']['fits'],prediction_disagreements=int(np.sum(old!=new)),
            dynamic_only_correct=int(np.sum((new==labels)&(old!=labels))),
            fixed_only_correct=int(np.sum((old==labels)&(new!=labels))),
            evaluation='designated val, not test' if task in ('coupon','preference') else 'outer development holdout'))
    for name in ('protocol.json','completion.json','all_models_frozen.json','physical_fits.jsonl'):
        shutil.copy2(a.run/name,a.output/name)
    summary=dict(completion=completion,comparisons=rows,formal_six_arm_C1_completed=False)
    save(a.output/'summary.json',summary);print(__import__('json').dumps(summary,indent=2))
