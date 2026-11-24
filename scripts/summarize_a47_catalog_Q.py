"""Compact catalogue study evidence, no model loading or training."""
import argparse
import csv
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    completion=read(args.run/'completion.json')
    if completion['status']!='completed':raise ValueError('All models must have frozen and been scored before summary')
    args.output.mkdir(parents=True,exist_ok=False);rows=read(args.run/'results.json');protocol=read(args.run/'protocol.json');tasks={}
    for name in ('coupon','preference','sms','photo','bean'):
        current=[r for r in rows if r['task']==name];entry=protocol['entries'][name];arms={};predictions={}
        for row in current:
            directory=args.run/(name+'_'+row['scope']);target=args.output/directory.name;target.mkdir()
            for file in ('run_manifest.json','final_selection.json','classifier_fits.json','verification.json'):
                shutil.copy2(directory/file,target/file)
            with (directory/'predictions/rows.csv').open(encoding='utf-8',newline='') as file:
                predictions[row['scope']]=[r['prediction'] for r in csv.DictReader(file)]
            arms[row['scope']]=dict(score=row['evaluation']['score'],metric=row['evaluation']['primary_metric'],rows=row['evaluation']['rows'],
                selected=row['outcome']['selected_candidate'],classifier_fits=row['actual_classifier_fits'],seconds=row['process_seconds'],
                reload_identical=row['evaluation']['reload_identical'],prediction_sha256=sha(directory/'predictions/rows.csv'))
        shared=entry['deduplicated_same_policy']
        tasks[name]=dict(candidate_counts={k:len(v) for k,v in entry['current_capabilities'].items()},arms=arms,
            deduplicated_same_policy=shared,
            full_minus_baseline=None if shared else arms['full']['score']-arms['baseline']['score'],
            prediction_disagreements=None if shared else sum(a!=b for a,b in zip(predictions['full'],predictions['baseline'],strict=True)),
            effect_estimable=not shared)
    save(args.output/'summary.json',dict(tasks=tasks,completion=completion,scope='Known development tasks, one split; no Flash or held-out transfer claim'))
    for name in ('protocol.json','completion.json','physical_fits.jsonl'):shutil.copy2(args.run/name,args.output/name)
    print(json.dumps(dict(tasks=tasks,actual_classifier_fits=completion['actual_classifier_fits'],seconds=completion['elapsed_seconds'])))
