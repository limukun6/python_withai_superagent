"""Audit unchanged accepted history and publish H1 scores/resources without any new calls or fitting."""
from pathlib import Path
import json
import shutil
import sqlite3
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from tools.accepted_usage_snapshot import row_digest,accepted_history
from superagent.task_budget import TaskBudgetLedger
from superagent.research_budget import research_preflight

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]


def rows(path,table):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM '+table)]


def main():
    out=ROOT/'research/a130_a132_H1_summary';out.mkdir(parents=True,exist_ok=False)
    source=hashes(ROOT);ledger_path=BASE/'runs/a129_B_ledger_return/private/ledger.sqlite'
    baseline=BASE/'runs/a130_accepted_history_v2/private/before.sqlite'
    authorization=BASE/'runs/a130_accepted_history_v2/accepted_history.json'
    save(out/'protocol.json',dict(batch='A130-A132-summary',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A130_A132_H1_paid_continuation_plan_20261123_zh.md'),
        actual_classifier_fits=0,real_API_requests=0))
    previous=rows(baseline,'requests');current=rows(ledger_path,'requests');index={r['id']:r for r in current}
    if len(previous)!=357 or any(index.get(r['id'])!=r for r in previous):raise ValueError('Accepted old requests changed')
    old_events=rows(baseline,'transitions');new_events={json.dumps(r,sort_keys=True) for r in rows(ledger_path,'transitions')}
    if any(json.dumps(r,sort_keys=True) not in new_events for r in old_events):raise ValueError('Original ledger transitions changed')
    ledger=TaskBudgetLedger(ledger_path);project=ledger.project_summary()
    with accepted_history(authorization,sha(authorization)):research_preflight(ledger,'h1_transfer',.30)
    new=[r for r in current if r['id'] not in {x['id'] for x in previous}]
    if len(new)!=6 or any(r['status']!='known' or r['scope']!='h1_transfer' for r in new):raise ValueError('Expected six known new H1 requests')
    fee=sum(r['course_units'] for r in new)/1e9;details=[];tokens=[0,0]
    for seed in ('61411','61421'):
        byarm={}
        for arm,folder in (('L0','a128_H1_local'),('R0','a131_H1_paid')):
            run=BASE/'runs'/folder/seed/'run';outcome=read(run/'run_outcome.json');history=read(run/'history.json')
            usage=[json.loads(s) for s in (run/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
            calls=[r for r in usage if r.get('budget_request_id')]
            local_tokens=[sum(r.get(k,0) or 0 for r in calls) for k in ('input_tokens','output_tokens')]
            tokens=[a+b for a,b in zip(tokens,local_tokens)]
            byarm[arm]=dict(selected=outcome['selected_candidate'],fits=outcome['classifier_fit_accounting'],
                seconds=outcome['elapsed_seconds'],calls=len(calls),input_tokens=local_tokens[0],output_tokens=local_tokens[1],
                known_course_cost_rmb=outcome['known_course_cost_rmb'],
                primary_trials=[dict(candidate=r['candidate'],score=r['search_score']) for r in history if r['split_id']=='primary'])
            manifest=read(run/'run_manifest.json')
            if manifest['source_files']!={k.removeprefix('superagent/'):v for k,v in source.items()}:raise ValueError('Model source differs')
        details.append(dict(outer=seed,arms=byarm))
    evaluated=read(BASE/'runs/a132_H1_evaluation/results.json')
    summary=dict(new_classifier_fits=22,A_cumulative_fits_after=1588,real_API_requests=6,input_tokens=tokens[0],output_tokens=tokens[1],
        known_incremental_course_cost_rmb=fee,actual_account_debit_rmb=None,project_summary=project,
        accepted_prior_unknowns_preserved=12,all_previous_requests_preserved=357,prior_transitions_preserved=len(old_events),
        new_unknowns=0,pending_requests=project['pending_requests'],source_identical_to_all_four_models=True,
        comparisons=evaluated['comparisons'],paths=details,independent_deliveries=evaluated['deliveries'],
        H_not_used_for_selection=True,independent_new_tasks=1,outer_partitions=2,
        interpretation='Raw Flash explored a symmetric MiniLM route which local fixed planning did not try within its four exploration slots; no checked-feedback or preference-E5 contribution tested here',
        historical_total_cost_complete=False)
    save(out/'results.json',summary)
    owner=dict(owner='A',window_status='exclusive_idle',ledger_path=str(ledger_path),ledger_sha256=sha(ledger_path),
        project_summary=project,accepted_history_sha256=sha(authorization),
        existing_unknowns_accepted_by_user=True,new_unknowns_require_stop=True,active_paid_queue=False)
    save(BASE/'runs/a131_H1_paid/current_ledger_owner.json',owner)
    save(out/'current_ledger_owner.json',owner)
    for name in ('a130_accepted_history','a130_accepted_history_v2','a130_policy_checks','a131_H1_paid_preflight','a131_H1_paid','a132_H1_evaluation'):
        origin=BASE/'runs'/name;target=ROOT/'research'/name;target.mkdir(parents=True,exist_ok=False)
        for filename in ('protocol.json','results.json','completion.json','accepted_history.json','ledger_owner.json','tests.log'):
            path=origin/filename
            if path.exists():shutil.copy2(path,target/filename)
    save(out/'completion.json',dict(status='completed_audit',actual_classifier_fits=0,real_API_requests=0,
        A_cumulative_fits_after=1588,all_old_records_preserved=True,active_queue=False))
    print(json.dumps({k:summary[k] for k in ('new_classifier_fits','A_cumulative_fits_after','real_API_requests','input_tokens','output_tokens','known_incremental_course_cost_rmb','project_summary','comparisons')}),flush=True)


if __name__=='__main__':main()
