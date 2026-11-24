"""Apply existing deployment rules without hiding partial failures or reused runtime caveats."""
import argparse
import csv
from pathlib import Path
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from tools.ab_deployment_criteria import assess, ranking_key

ROOT=Path(__file__).resolve().parents[1]
ARMS=('L0','L1','R0','R1','E0','E1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--B-receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A118',runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A118_A119_five_task_and_independent_review_plan.md'),
         criterion_code_sha256=sha(ROOT/'tools/ab_deployment_criteria.py'),actual_classifier_fits=0,real_API_requests=0))
    try:
        rows=[];tasks={};inputs={};source=None;A_new_fits=A_reused_fits=A_calls=0;A_known=0.
        for task,metric,study in [('coupon','accuracy','a89_coupon_C1'),('flowers','accuracy','a97_flower_C1'),('sms','macro_f1','a102_sms_C1')]:
            folder=ROOT/'research'/study;summary=read(folder/'summary.json');protocol=read(folder/'protocol.json')
            if summary['completion']['status']!='completed':raise ValueError('Incomplete A study')
            if source is not None and source!=protocol['source_files']:raise ValueError('Different A C1 sources')
            source=protocol['source_files'];inputs[task]=dict(summary_sha256=sha(folder/'summary.json'),protocol_sha256=sha(folder/'protocol.json'))
            tasks[task]={}
            for arm,row in summary['arms'].items():
                current=row.get('process_seconds',row.get('new_process_seconds'))
                old=row.get('reference_process_seconds',row.get('reused_reference_process_seconds'))
                reused=bool(row.get('reused',row.get('reused_fits',0)))
                record=dict(task=task,metric=metric,arm=arm,score=row['score'],complete=True,delivery='complete',
                    fits=row.get('new_fits',row.get('new_classifier_fits'))+row.get('reused_fits',row.get('reused_classifier_fits',0)),
                    requests=row['new_API_requests'],seconds=current if current is not None else old,
                    runtime_reused=reused,prediction_sha256=row['prediction_sha256'],rows=row['rows'],failure_reason=None)
                tasks[task][arm]=record;rows.append(record)
            c=summary['completion'];A_new_fits+=c['actual_classifier_fits'];A_reused_fits+=c['reused_classifier_fits'];A_calls+=c['real_api_requests']
            A_known+=summary['known_incremental_estimated_rmb']
        for task in ('preference','bean'):
            folder=a.B_receipts/'research/b_c1_validation_1123'/task
            result=read(folder/'results.json');protocol=read(folder/'protocol.json')
            declared=[v for v in protocol.values() if isinstance(v,dict) and 'ab_run.py' in v]
            if not declared or any(v!=source for v in declared):raise ValueError('B C1 source differs')
            tasks[task]={};inputs[task]=dict(results_sha256=sha(folder/'results.json'),protocol_sha256=sha(folder/'protocol.json'))
            for arm,row in {**result['arms'],**result.get('partial_delivery',{})}.items():
                record=dict(task=task,metric='accuracy',arm=arm,score=row['accuracy'],complete=arm in result['arms'],
                    delivery=row['delivery'],fits=row['fits'],requests=row['requests'],seconds=row['elapsed_seconds'],
                    runtime_reused=False,prediction_sha256=row['prediction_sha256'],rows=row['rows'],failure_reason=row.get('failure_reason'))
                tasks[task][arm]=record;rows.append(record)
        if len(rows)!=30 or any(set(v)!=set(ARMS) for v in tasks.values()):raise ValueError('Expected all 30 terminals')
        assessments={}
        for arm in ARMS[1:]:
            pairs=[]
            for task,arms in tasks.items():
                baseline,row=arms['L0'],arms[arm]
                pairs.append(dict(task=task,split='registered_development_outer',metric=row['metric'],
                    baseline_score=baseline['score'],score=row['score'],baseline_seconds=baseline['seconds'],
                    seconds=row['seconds'],requests=row['requests']))
            report=assess(pairs)
            incomplete=[t for t in tasks if not tasks[t][arm]['complete']]
            reused_times=[t for t in tasks if tasks[t][arm]['runtime_reused'] or tasks[t]['L0']['runtime_reused']]
            # A89 explicitly declared these reused timings unfit for a fresh efficiency claim.
            efficiency_supported=not reused_times
            promotable=not incomplete and (report['quality_condition'] or (report['efficiency_condition'] and efficiency_supported))
            assessments[arm]=dict(numeric_criteria_including_references=report,completed_tasks=5-len(incomplete),
                incomplete_tasks=incomplete,reused_runtime_tasks=reused_times,efficiency_evidence_adequate=efficiency_supported,
                eligible_for_recommendation=promotable,ranking_key=ranking_key(report),
                interpretation='Numeric timing pass is distinct from a supported efficiency promotion; partials retained')
        eligible=[arm for arm in assessments if assessments[arm]['eligible_for_recommendation']]
        recommended=min(eligible,key=lambda arm:tuple(assessments[arm]['ranking_key'])) if eligible else 'L0'
        B_resources=read(a.B_receipts/'research/b_c1_validation_1123/resource_summary.json')['scopes']
        fit_totals={arm:sum(tasks[t][arm]['fits'] for t in tasks) for arm in ARMS}
        call_totals={arm:sum(tasks[t][arm]['requests'] for t in tasks) for arm in ARMS}
        known=A_known+sum(v['known_course_estimate_rmb'] for v in B_resources.values())
        result=dict(tasks=tasks,assessments=assessments,recommended_F_star_for_C1_evidence=recommended,
            recommendation_reason='Conservative original baseline; L1 numerical efficiency pass depends on explicitly unverified reused runtime' if recommended=='L0' else 'Passed original rule with adequate completion and evidence',
            shared_deployment_version_frozen=False,new_native_L12_source_performance_matrix_completed=False,
            observed_terminal_models=30,completed_paths=sum(r['complete'] for r in rows),partial_paths=[(r['task'],r['arm']) for r in rows if not r['complete']],
            historical_resource_totals=dict(new_fits=A_new_fits+sum(v['fits'] for v in B_resources.values()),
                reused_fits=A_reused_fits,logical_fit_total=sum(fit_totals.values()),requests=A_calls+sum(v['requests'] for v in B_resources.values()),
                known_estimated_rmb=known,complete_estimated_rmb=None,new_unknown_requests=1,
                logical_fits_by_arm=fit_totals,requests_by_arm=call_totals,
                L1_fit_reduction_vs_L0_fraction=(fit_totals['L0']-fit_totals['L1'])/fit_totals['L0']),
            input_hashes=inputs,production_source_files=source,
            limitations=['One outer schedule per task; reused paths are not fresh repetitions','Runtime order/cache/hardware caveats; no universal speed claim',
                         'Task2 course designated validation and Bean development holdout explicitly distinguished',
                         'No formal H freeze or paid execution in this analysis'])
        save(a.output/'results.json',result)
        with (a.output/'five_task_arms.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(a.output/'completion.json',dict(status='completed_analysis',actual_classifier_fits=0,real_API_requests=0,
            tasks=5,terminal_models=30,completed_paths=28,recommended_F_star=recommended,
            joint_deployment_freeze=False,A_cumulative_fits_after=1505,elapsed_seconds=time.monotonic()-started))
        print(dict(recommended=recommended,fit_totals=fit_totals,call_totals=call_totals,
                   numerical_passes=[a for a,v in assessments.items() if v['numeric_criteria_including_references']['meets_measured_criteria']],
                   eligible=eligible),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
