"""Archive completed A88 evidence and paired correctness; never train or load models."""
import argparse
import csv
from pathlib import Path
import shutil
import joblib
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    completion=read(a.run/'completion.json')
    if completion['status']!='completed':raise ValueError('Only summarize four frozen and scored models')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);rows=read(a.run/'results.json');protocol=read(a.run/'protocol.json');tasks={}
    for name in ('coupon','photo'):
        arms={};predictions={};truth=[str(v) for v in joblib.load(a.run/'evaluation'/name/'holdout.joblib')['y']]
        for row in rows:
            if row['task']!=name:continue
            scope=row['scope'];directory=a.run/(name+'_'+scope);target=out/directory.name;target.mkdir()
            for filename in ('run_manifest.json','final_selection.json','classifier_fits.json','verification.json','actions.json'):
                shutil.copyfile(directory/filename,target/filename)
            with (directory/'predictions/rows.csv').open(encoding='utf-8',newline='') as stream:
                output=list(csv.DictReader(stream))
            if [int(r['row_id']) for r in output]!=list(range(len(truth))):raise ValueError('Prediction identities differ')
            predictions[scope]=[r['prediction'] for r in output]
            arms[scope]=dict(score=row['evaluation']['score'],metric=row['evaluation']['primary_metric'],rows=row['evaluation']['rows'],
                selected=row['outcome']['selected_candidate'],classifier_fits=row['actual_classifier_fits'],process_seconds=row['process_seconds'],
                reload_identical=row['evaluation']['reload_identical'],prediction_sha256=sha(directory/'predictions/rows.csv'),
                executed_candidates=[r['candidate'] for r in read(directory/'history.json')])
        base,full=predictions['baseline'],predictions['full']
        tasks[name]=dict(candidate_counts={k:len(v) for k,v in protocol['entries'][name]['current_capabilities'].items()},arms=arms,
            full_minus_baseline=arms['full']['score']-arms['baseline']['score'],prediction_disagreements=sum(a!=b for a,b in zip(base,full,strict=True)),
            full_correct_only=sum(f==y and b!=y for b,f,y in zip(base,full,truth,strict=True)),
            baseline_correct_only=sum(b==y and f!=y for b,f,y in zip(base,full,truth,strict=True)),
            jointly_wrong=sum(b!=y and f!=y for b,f,y in zip(base,full,truth,strict=True)))
    save(out/'summary.json',dict(tasks=tasks,completion=completion,scope='Two known tasks, one registered development holdout each; not full Q/C1/H',
        actual_classifier_fits=0,real_api_requests=0))
    for name in ('protocol.json','completion.json','physical_fits.jsonl','all_models_frozen.json'):
        shutil.copyfile(a.run/name,out/name)
    print({name:dict(delta=row['full_minus_baseline'],disagreements=row['prediction_disagreements'],
        full_correct_only=row['full_correct_only'],baseline_correct_only=row['baseline_correct_only']) for name,row in tasks.items()})
