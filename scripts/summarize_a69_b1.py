"""Describe paired outcomes, actual resource use and prediction agreement without fitting."""
import argparse
import csv
from pathlib import Path
import statistics
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);done=read(a.study/'completion.json')
    if done['status']!='completed':raise ValueError('No interim outer-score analysis before all paths completed')
    rows=read(a.study/'results.json');pairs=[]
    for value in read(a.study/'comparison.json'):
        name,seed=value['task'],value['seed'];f=next(r for r in rows if (r['task'],r['seed'],r['scheduler'])==(name,seed,'ab-fixed'))
        d=next(r for r in rows if (r['task'],r['seed'],r['scheduler'])==(name,seed,'ab-v1'))
        def labels(row):
            with (a.study/row['label']/'predictions/rows.csv').open(encoding='utf-8',newline='') as file:return list(csv.DictReader(file))
        fp,dp=labels(f),labels(d)
        if [r['row_id'] for r in fp]!=[r['row_id'] for r in dp]:raise ValueError('Paired predictions have different ordered row identities')
        pairs.append(dict(**value,rows=len(fp),prediction_disagreements=sum(x['prediction']!=y['prediction'] for x,y in zip(fp,dp,strict=True)),
            same_selected_candidate=f['outcome']['selected_candidate']==d['outcome']['selected_candidate'],
            fixed_primary_candidates=f['distinct_primary_candidates'],dynamic_primary_candidates=d['distinct_primary_candidates'],
            fixed_confirmation_fits=f['confirmation_fits'],dynamic_confirmation_fits=d['confirmation_fits'],
            fixed_promotions=f['promotions'],dynamic_promotions=d['promotions']))
    save(a.output/'pairs.json',pairs)
    save(a.output/'completion.json',dict(status='completed',pairs=len(pairs),tasks=len({r['task'] for r in pairs}),
        improved_pairs=sum(r['dynamic_minus_fixed']>1e-12 for r in pairs),tied_pairs=sum(abs(r['dynamic_minus_fixed'])<=1e-12 for r in pairs),
        worse_pairs=sum(r['dynamic_minus_fixed']<-1e-12 for r in pairs),prediction_identical_pairs=sum(r['prediction_disagreements']==0 for r in pairs),
        fixed_fits=sum(r['fixed_fits'] for r in pairs),dynamic_fits=sum(r['dynamic_fits'] for r in pairs),
        fixed_process_seconds=sum(r['fixed_seconds'] for r in pairs),dynamic_process_seconds=sum(r['dynamic_seconds'] for r in pairs),
        median_pair_relative_time_change=statistics.median(r['dynamic_seconds']/r['fixed_seconds']-1 for r in pairs),
        actual_classifier_fits=0,real_api_requests=0,source_results_sha256=sha(a.study/'results.json'),
        independent_outer_splits=False,scope='Descriptive paired development results; no significance claim'))
