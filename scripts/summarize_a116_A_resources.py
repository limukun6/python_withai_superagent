"""Read-only matched C1 resource accounting for A's checked feedback; no model execution."""
import argparse
import csv
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]
STUDIES={'coupon':'a89_coupon_C1','flowers':'a97_flower_C1','sms':'a102_sms_C1'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    inputs={}
    save(a.output/'protocol.json',dict(batch='A116',runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A116_A_track_resources_plan.md'),actual_classifier_fits=0,real_API_requests=0))
    try:
        source=None
        totals={g:dict(requests=0,fits=0,input_tokens=0,output_tokens=0,estimated_rmb=0.) for g in ('raw','checked')}
        rows=[]
        for task,study in STUDIES.items():
            folder=ROOT/'research'/study
            summary=read(folder/'summary.json')
            protocol=read(folder/'protocol.json')
            if summary['completion']['status']!='completed': raise ValueError('Incomplete task '+task)
            if source is not None and source!=protocol['source_files']: raise ValueError('C1 sources differ')
            source=protocol['source_files']
            inputs[task]=dict(summary_sha256=sha(folder/'summary.json'),protocol_sha256=sha(folder/'protocol.json'))
            for group,arms in [('raw',('R0','R1')),('checked',('E0','E1'))]:
                for arm in arms:
                    row=summary['arms'][arm]
                    fits=row.get('new_fits',row.get('new_classifier_fits'))
                    cost=row.get('known_estimated_rmb',row.get('new_known_estimated_rmb'))
                    if row['new_API_requests']!=3 or fits not in (10,11): raise ValueError('Unexpected C1 accounting')
                    result=dict(task=task,arm=arm,feedback=group,score=row['score'],fits=fits,
                                requests=row['new_API_requests'],input_tokens=row['input_tokens'],
                                output_tokens=row['output_tokens'],estimated_rmb=cost)
                    rows.append(result)
                    for key in totals[group]: totals[group][key]+=result[key]
        reuse=read(ROOT/'research/a69_a71_d3_completion/A71/completion.json')
        if reuse['actual_classifier_fits']+reuse['reused_evaluations']!=reuse['logical_evaluations']:
            raise ValueError('Reuse receipt inconsistent')
        save(a.output/'results.json',dict(per_arm=rows,totals=totals,
             checked_cost_change_percent=100*(totals['checked']['estimated_rmb']/totals['raw']['estimated_rmb']-1),
             checked_request_saving=totals['raw']['requests']-totals['checked']['requests'],
             checked_fit_saving=totals['raw']['fits']-totals['checked']['fits'],
             evaluator_reuse=dict(logical_evaluations=reuse['logical_evaluations'],actual_fits=reuse['actual_classifier_fits'],
                  reused_evaluations=reuse['reused_evaluations'],avoided_fit_fraction=reuse['reused_evaluations']/reuse['logical_evaluations'],
                  naive_full_refit_baseline_actually_run=False,wall_time_saving_measured=False),
             input_hashes=inputs,source_files=source,actual_platform_debit_verified=False,
             conclusion='Checked feedback did not save requests/fits and increased estimated token fees; evaluator dedup avoided redundant fits'))
        with (a.output/'per_arm_resources.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(a.output/'completion.json',dict(status='completed_readonly',actual_classifier_fits=0,real_API_requests=0,
             tasks=3,arms=12,elapsed_seconds=time.monotonic()-start))
        print(totals,flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__': main()
