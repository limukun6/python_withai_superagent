"""Audit a preserved interrupted matrix; no deserialization of models or score labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import physical
from scripts.status_a29_choice_matrix import completion_failure
from superagent.ab_run import sources
from superagent.task_budget import TaskBudgetLedger


def run(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    protocol = read(args.matrix/'protocol.json')
    if read(args.matrix/'completion.json')['status'] == 'completed':
        raise ValueError('Use the ordinary complete-matrix analysis for a complete study')
    window = read(args.window/'window.json')
    save(root/'protocol.json', dict(batch='A40-interruption-audit', original_protocol_sha256=sha(args.matrix/'protocol.json'),
        matrix_completion_sha256=sha(args.matrix/'completion.json'), source_files=sources(),
        runner_sha256=sha(Path(__file__)), original_matrix_source_unchanged=protocol['source_files']==sources(),
        original_matrix_modified=False, no_model_or_label_deserialization=True, real_api_request_limit=0, classifier_fit_limit=0))
    def offline(event, values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Network forbidden in interruption audit')
    sys.addaudithook(offline)
    traced = physical(args.matrix); names=Counter(row['run_name'] for row in traced)
    results=[]; usage=[]
    ledger=TaskBudgetLedger(args.ledger)
    with sqlite3.connect(args.ledger.resolve().as_uri()+'?mode=ro', uri=True) as db:
        charges={key:dict(status=status,units=units) for key,status,units in db.execute('SELECT id,status,course_units FROM requests')}
    for task,arm in protocol['order']:
        directory=args.matrix/(task+'_'+arm)
        outcome_path=directory/'run_outcome.json'
        if not outcome_path.exists():
            results.append(dict(task=task,arm=arm,status='not_started'))
            continue
        outcome=read(outcome_path); counts=outcome['classifier_fit_accounting']
        if names[directory.name] != counts['attempted_classifier_fits']:
            raise ValueError('Physical count differs: '+directory.name)
        manifest=read(directory/'run_manifest.json')
        if manifest['source_files']!=protocol['source_files']:
            raise ValueError('Path differs from frozen production version')
        records=[json.loads(line) for line in (directory/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
        calls=[row for row in records if row['event']=='llm_call']
        if len(calls)!=outcome['new_api_requests'] or any(row.get('usage_known') is False for row in records):
            raise ValueError('Unknown/unbound request; stop without retry')
        for row in calls:
            charge=charges[row['budget_request_id']]
            if charge['status']!='known' or abs(charge['units']/1e9-row['course_cost_rmb'])>1e-9:
                raise ValueError('Persistent usage settlement differs')
        usage.extend(calls)
        history=read(directory/'history.json')
        primary=[row for row in history if row['status']=='completed' and row['split_id']=='primary']
        routes=list(dict.fromkeys(manifest['capabilities'][row['candidate']]['route'] for row in primary))
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        selection=read(directory/'final_selection.json') if (directory/'final_selection.json').exists() else None
        if selection and sha(directory/selection['model_file'])!=selection['model_sha256']:
            raise ValueError('Checkpoint model changed')
        results.append(dict(task=task,arm=arm,status='completed' if outcome['requested_planner_completed'] else 'partial',
            model_available=outcome['model_available'],full_refit_completed=outcome.get('full_refit_completed'),
            attempted_classifier_fits=counts['attempted_classifier_fits'],completed_classifier_fits=counts['completed_classifier_fits'],
            unknown_failed_fit_attempts=counts['unknown_failed_attempts'],real_api_requests=outcome['new_api_requests'],
            known_course_cost_rmb=outcome['known_course_cost_rmb'],elapsed_seconds_including_host_pause=outcome['elapsed_seconds'],
            selected_candidate=outcome.get('selected_candidate'),explored_primary_routes=routes,
            coverage_projection_events=sum(row['kind']=='candidate_coverage_projection' for row in events),
            failure=completion_failure(outcome),outcome_sha256=sha(outcome_path),
            model_sha256=selection['model_sha256'] if selection else None))
    if len({row['budget_request_id'] for row in usage})!=len(usage):
        raise ValueError('A request was counted twice')
    project=ledger.project_summary(); current=read(args.matrix/'completion.json')
    if project['requests']-window['requests_at_start']!=len(usage):
        raise ValueError('Concurrent or missing API use; manual attribution required')
    costs=sum(row['course_cost_rmb'] for row in usage)
    if abs(project['known_course_cost_rmb']-window['starting_known_course_cost_rmb']-costs)>1e-8:
        raise ValueError('Project delta and batch receipts differ')
    save(root/'results.json',results)
    counts=Counter(row['status'] for row in results)
    save(root/'completion.json',dict(status='audited_incomplete_matrix',planned_paths=len(results),path_status_counts=dict(counts),
        observed_A32_classifier_fits=len(traced),A32_real_api_requests=len(usage),known_A32_course_cost_rmb=costs,
        input_tokens=sum(row['input_tokens'] for row in usage),output_tokens=sum(row['output_tokens'] for row in usage),
        new_unknown=project['unresolved_requests']-window['historical_unknown'],project_usage_after=project,
        original_batch_elapsed_including_host_pause=current['elapsed_seconds'],
        completed_paths_cli_seconds=sum(row.get('elapsed_seconds_including_host_pause',0) for row in results if row['status']=='completed'),
        validation_opened=(args.matrix/'evaluation').exists(),all_models_frozen=(args.matrix/'all_models_frozen.json').exists(),
        A34_A35_A36_started=False,tracked_A_window_fits=window['starting_tracked_A_window_fits']+7+len(traced),
        actual_new_audit_classifier_fits=0,real_new_audit_api_requests=0,course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False,platform_actual_account_debit_rmb=None))
    print(json.dumps(read(root/'completion.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','window','ledger','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
