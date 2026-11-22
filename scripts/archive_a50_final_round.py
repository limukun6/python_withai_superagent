"""Compact final-round evidence and actual fit/API reconciliation, never binary models or datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.task_budget import TaskBudgetLedger


def canonical(path):
    value=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.suffix=='.jsonl' else read(path)
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','ledger','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);index=[]
    sets={
        'a52_relocation':['protocol.json','completion.json','results.json'],
        'a53_action_audit':['protocol.json','completion.json','comparison.json','physical_fits.jsonl','actions.json'],
        'a54_selfcontained_delivery':['protocol.json','completion.json','results.json'],
        'a55_fresh_states':['protocol.json','completion.json','states.json'],
        'a55_fresh_answers':['protocol.json','completion.json','answers.json'],
        'a55_fresh_evaluation':['protocol.json','completion.json','comparison.json','physical_fits.jsonl'],
        'a56_handoff_acceptance_final':['protocol.json','completion.json','results.json'],
        'a57_fresh_environment':['protocol.json','completion.json','results.json'],
        'a58_feedback_closed_loop':['protocol.json','completion.json','results.json','physical_fits.jsonl'],
        'a59_selection_audit':['protocol.json','completion.json'],
        'a50_final_regression':['protocol.json','completion.json','fit_observations.json'],
        'a50_contract_checks':['protocol.json','completion.json','fit_observations.json'],
    }
    for name,files in sets.items():
        target=root/name;target.mkdir()
        for file in files:
            source=a.runs/name/file
            if not source.exists() and file=='protocol.json' and name=='a50_final_regression':continue
            shutil.copy2(source,target/file)
            index.append(dict(run=name,file=file,original_runtime_bytes_sha256=sha(source),canonical_value_sha256=canonical(source)))
    for folder in (a.runs/'a55_fresh_answers').iterdir():
        if folder.is_dir() and (folder/'llm_usage.jsonl').exists():
            target=root/'a55_fresh_answers'/folder.name;target.mkdir();shutil.copy2(folder/'llm_usage.jsonl',target/'llm_usage.jsonl')
    target=root/'a58_feedback_closed_loop/preference_E';target.mkdir()
    for file in ('run_manifest.json','final_selection.json','verification.json','classifier_fits.json','llm_usage.jsonl'):
        shutil.copy2(a.runs/'a58_feedback_closed_loop/preference_E'/file,target/file)
    # Actual fit entries, including the two failed integration checks, never only successful training.
    counts={}
    for name in ('a32_coverage_matrix','a42_recovery_supplement','a44_a46_offline','a44_a46_offline_fixed','a44_a46_offline_final',
                 'a47_catalog_Q','a48_calibrated_mock_loop','a50_facts_calibration','a51_joint_transfer','a53_action_audit',
                 'a55_fresh_evaluation','a58_feedback_closed_loop'):
        path=a.runs/name/'physical_fits.jsonl';counts[name]=sum(bool(line.strip()) for line in path.read_text(encoding='utf-8').splitlines())
    for name in ('a31_coverage_checks','a44_regression','a50_final_regression'):
        counts[name]=len(read(a.runs/name/'fit_observations.json'))
    window=read(a.runs/'three_hour_20261121_2324/window.json');calls=[];attempts=[]
    for name in ('a32_coverage_matrix','a42_recovery_supplement','a50_facts_calibration','a51_joint_transfer','a55_fresh_answers','a58_feedback_closed_loop'):
        for file in (a.runs/name).glob('*/llm_usage.jsonl'):
            for row in (json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()):
                if row['event']!='ledger_initialized':attempts.append(row)
                if row['event']=='llm_call':calls.append(row)
    if len({r['budget_request_id'] for r in attempts})!=len(attempts):raise ValueError('Double-counted request ticket')
    project=TaskBudgetLedger(a.ledger).project_summary();cost=sum(r['course_cost_rmb'] for r in calls)
    if project['requests']-window['requests_at_start']!=len(attempts) or abs(project['known_course_cost_rmb']-window['starting_known_course_cost_rmb']-cost)>1e-8:
        raise ValueError('Real cumulative ledger and run receipts differ')
    audit=dict(starting_tracked_A_fits=window['starting_tracked_A_window_fits'],actual_new_fit_entries_by_batch=counts,
        actual_new_fit_entries=sum(counts.values()),tracked_A_fits_after=window['starting_tracked_A_window_fits']+sum(counts.values()),
        new_chat_request_attempt_records=len(attempts),successful_new_chat_responses=len(calls),
        reviewed_not_dispatched=len(attempts)-len(calls),input_tokens=sum(r['input_tokens'] for r in calls),
        output_tokens=sum(r['output_tokens'] for r in calls),known_new_course_cost_rmb=cost,project_usage_after=project,
        historical_unknown_retained=11,no_new_unknown=project['unresolved_requests']==11,no_pending=project['pending_requests']==0,
        model_list_GETs_not_counted_as_chat=True,platform_actual_debit_rmb=None,course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False,formal_test_or_H_opened=False,
        runtime_note='Original A32 elapsed includes standby and cannot be added as normal training time')
    save(root/'resource_audit.json',audit)
    save(root/'index.json',dict(files=index,line_endings='Git normalizes text to LF; original_runtime_bytes hashes describe local original logs. Use canonical_value_sha256 for JSON value verification after checkout.',
        models_or_raw_data_or_credentials_or_ledgers_copied=False))
    print(json.dumps(audit))
