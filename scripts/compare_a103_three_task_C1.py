"""Matched-source, per-task C1 contrasts; do not pool distinct primary metrics or select F*."""
import argparse
import csv
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]
PAIRS=[('E0','R0'),('E1','R1'),('R1','R0'),('E1','E0'),('R0','L0'),('E1','L1')]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sms-summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    studies=[('coupon','accuracy',ROOT/'research/a89_coupon_C1'),('flowers','accuracy',ROOT/'research/a97_flower_C1'),
             ('sms','macro_f1',a.sms_summary)]
    frozen=None;rows=[];inputs={}
    for task,metric,folder in studies:
        summary=read(folder/'summary.json');protocol=read(folder/'protocol.json')
        if summary['completion']['status']!='completed':raise ValueError('Incomplete source study')
        if frozen is not None and protocol['source_files']!=frozen:raise ValueError('Different production source cannot be pooled')
        frozen=protocol['source_files'];arms=summary['arms']
        inputs[task]=dict(summary_sha256=sha(folder/'summary.json'),protocol_sha256=sha(folder/'protocol.json'))
        for left,right in PAIRS:
            contrast=next(c for c in summary['contrasts'] if c['left']==left and c['right']==right)
            rows.append(dict(task=task,primary_metric=metric,left=left,right=right,
                left_score=arms[left]['score'],right_score=arms[right]['score'],
                delta_pp=100*(arms[left]['score']-arms[right]['score']),
                prediction_disagreements=contrast['prediction_disagreements']))
    save(a.output/'protocol.json',dict(batch='A103_three_task',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A103_sms_results_analysis_plan.md'),input_hashes=inputs,source_files=frozen,
        actual_classifier_fits=0,real_API_requests=0))
    save(a.output/'results.json',dict(task_contrasts=rows,F_star_selected=False,
        missing_current_matrix_tasks=['preference','bean'],
        limitation='Three known tasks, one schedule each; task metrics distinct; no pooled accuracy or final five-task promotion'))
    with (a.output/'per_task_contrasts.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(a.output/'completion.json',dict(status='passed',tasks=3,contrasts=len(rows),actual_classifier_fits=0,
        real_API_requests=0,elapsed_seconds=time.monotonic()-start))
    print(rows)
