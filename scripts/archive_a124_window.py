"""Archive compact evidence and audit current autonomous paths; never copy data/models/SQLite."""
from collections import Counter
import argparse
import json
from pathlib import Path
import shutil
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import ROOT,BASE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    output=a.output;output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(output/'protocol.json',dict(batch='A124',source_files=hashes(ROOT),runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A120_A124_next_5h_plan_20261123_zh.md'),maximum_classifier_fits=0,real_API_requests=0))
    folders=['a120_cases','a121_preflight','a121_evaluation','a121_transport_regression',
             'a122_prepared','a122_prepared_v2','a122_acceptance','a122_acceptance_continued',
             'a123_H1_prepared','a123_H1_gate','a123_score_gate','a123_delivery_faults','a123_delivery_faults_v2']
    for name in folders:
        source=BASE/'runs'/name;target=ROOT/'research'/name
        target.mkdir(exist_ok=True)
        for filename in ('protocol.json','completion.json','results.json','cases.json','case_results.json','reply_results.json','regression.log'):
            path=source/filename
            if path.exists():
                if (target/filename).exists():
                    if sha(path)!=sha(target/filename):raise ValueError('Existing archive differs: '+name+'/'+filename)
                else:shutil.copy2(path,target/filename)
    completion=read(BASE/'runs/a122_acceptance_continued/completion.json')
    paths=read(BASE/'runs/a122_acceptance_continued/results.json');rows=[]
    for row in paths:
        folder=Path(row.get('reused_from',str(BASE/'runs/a122_acceptance_continued'/row['task'])))
        run=folder/'run';history=read(run/'history.json');actions=read(run/'actions.json');outcome=read(run/'run_outcome.json')
        normalized={key.removeprefix('superagent/'):value for key,value in hashes(ROOT).items()}
        if normalized!=read(run/'run_manifest.json')['source_files']:
            raise ValueError('Current source no longer matches acceptance models')
        split_counts=dict(Counter(r['split_id'] for r in history if r['status']=='completed'))
        rows.append(dict(task=row['task'],selected=row['selected_candidate'],rows=row['rows'],
            fits=outcome['classifier_fit_accounting'],action_counts=dict(Counter(r['action'] for r in actions)),
            measured_split_counts=split_counts,independent_confirmation_measured=any(k!='primary' for k in split_counts),
            original_independent_confirmation_observed_field=outcome['confirmation'].get('independent_confirmation_observed'),
            field_interpretation='Existing field tracks promotion basis, not whether confirmation workers ran',
            training_cli_seconds=outcome['elapsed_seconds'],original_run=str(run),
            prediction_sha256=row['prediction_sha256'],model_sha256=row['model_sha256'],
            native_relocated_portable_identical=row['native_relocated_portable_identical']))
    physical=[]
    for name in ('a122_acceptance','a122_acceptance_continued'):
        p=BASE/'runs'/name/'physical_fits.jsonl'
        physical.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines())
    if len(physical)!=33 or completion['A_cumulative_fits_after']!=1538:raise ValueError('Resource totals differ')
    summary=dict(paths=rows,new_classifier_fits=33,completed_classifier_fits=33,A_cumulative_fits_after=1538,
        real_API_requests=0,new_known_API_cost_rmb=0,actual_platform_debit=None,
        team_complete_historical_cost=None,team_history='B summary357/unknown12, actual returned SQLite not received',
        formal_test_accessed=False,H_outer_scores_computed=0,production_source_equals_all_three_model_sources=True,
        old_C1_source='c64bee4, retained as historical development evidence; not current-version reruns',
        experimental_semantics_default='off',experimental_semantics_promoted=False,
        elapsed_window_claim='Actual sub-batch times retained; not claimed to have run five wall-clock hours')
    save(output/'results.json',summary)
    save(output/'completion.json',dict(status='completed_archive',actual_classifier_fits=0,real_API_requests=0,
        A_cumulative_fits_after=1538,elapsed_seconds=time.monotonic()-start))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
