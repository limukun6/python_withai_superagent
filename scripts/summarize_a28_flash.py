"""Read-only A28 summary/archive: actual requests, usage, choices and frozen-model delivery."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from superagent.task_budget import TaskBudgetLedger


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def summarize(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = [('connectivity', args.connectivity), ('sms_pilot', args.pilot)]
    for name, root in roots:
        target = output / name
        target.mkdir()
        for filename in ('protocol.json', 'completion.json', 'results.json', 'budget_before.json',
                         'answer.json', 'llm_usage.jsonl', 'events.jsonl', 'all_models_frozen.json'):
            if (root / filename).exists():
                shutil.copy2(root / filename, target / filename)
    pilot_completion = read(args.pilot / 'completion.json')
    arms, common, prediction_vectors = {}, {}, {}
    total_usage = []
    for name, root in [('connectivity', args.connectivity), *[(mode, args.pilot / mode)
                        for mode in ('raw-history', 'checked-history')]]:
        if not root.exists():
            continue
        events = records(root / 'events.jsonl')
        usage = [row for row in records(root / 'llm_usage.jsonl') if row.get('event') != 'ledger_initialized']
        total_usage.extend(usage)
        details = dict(request_attempts=sum(row.get('kind') == 'llm_request' for row in events),
            responses=sum(row.get('kind') == 'llm_response' for row in events),
            input_tokens=sum(row.get('input_tokens') or 0 for row in usage),
            output_tokens=sum(row.get('output_tokens') or 0 for row in usage),
            known_course_cost_rmb=sum(row.get('course_cost_rmb') or 0 for row in usage),
            unknown_usage_calls=sum(row.get('usage_known') is False for row in usage),
            call_latencies_seconds=[row['latency_seconds'] for row in usage if 'latency_seconds' in row])
        if name == 'connectivity':
            connectivity = details
            continue
        target = output / 'sms_pilot' / name
        target.mkdir()
        for filename in ('run_manifest.json', 'run_outcome.json', 'actions.json', 'history.json',
                         'classifier_fits.json', 'final_selection.json', 'verification.json',
                         'events.jsonl', 'llm_usage.jsonl'):
            if (root / filename).exists():
                shutil.copy2(root / filename, target / filename)
        outcome = read(root / 'run_outcome.json')
        details.update(status=outcome['status'], classifier_fit_accounting=outcome['classifier_fit_accounting'],
            selected_candidate=outcome.get('selected_candidate'), action_counts=outcome['action_counts'],
            elapsed_seconds=outcome['elapsed_seconds'], source_unchanged=outcome['source_unchanged'],
            error=outcome.get('error'), evaluation=read(root / 'verification.json') if (root / 'verification.json').exists() else None)
        history = read(root / 'history.json') if (root / 'history.json').exists() else []
        details['trial_trace'] = [{key: row.get(key) for key in ('trial', 'candidate', 'split_id', 'search_score',
            'metric_identity', 'configuration_fact_check', 'hypothesis_check', 'explanation_audit')}
            for row in history]
        details['flash_proposals'] = [row['proposal'] for row in events if row.get('kind') == 'ab_flash_proposal']
        first = next((event for event in events if event.get('kind') == 'llm_request'), None)
        if first:
            context = json.loads(first['messages'][1]['content'])
            common[name] = {key: value for key, value in context.items() if key not in ('checks', 'memory')}
            details['first_request_common_context_sha256'] = fingerprint(common[name])
            details['first_request_checks_present'] = 'checks' in context
            details['first_request_memory_present'] = 'memory' in context
        prediction = root / 'predictions' / 'rows.csv'
        if prediction.exists():
            with prediction.open(encoding='utf-8', newline='') as file:
                prediction_vectors[name] = [row['prediction'] for row in csv.DictReader(file)]
            details['prediction_csv_sha256'] = hashlib.sha256(prediction.read_bytes()).hexdigest()
        arms[name] = details
    complete = pilot_completion['status'] == 'completed'
    comparison = None
    if complete:
        a, b = (arms[name] for name in ('raw-history', 'checked-history'))
        if not all(arm['source_unchanged'] for arm in (a, b)):
            raise ValueError('Cannot compare changed-source arms')
        scores = [arm['evaluation']['score'] for arm in (a, b)]
        comparison = dict(primary_metric=a['evaluation']['primary_metric'], rows=a['evaluation']['rows'],
            raw_score=scores[0], checked_score=scores[1], checked_minus_raw=scores[1] - scores[0],
            prediction_disagreements=sum(x != y for x, y in zip(prediction_vectors['raw-history'],
                prediction_vectors['checked-history'], strict=True)),
            first_request_common_context_identical=common['raw-history'] == common['checked-history'],
            scope='One real SMS development split, raw then checked; no statistical, H-transfer or six-arm claim')
    request_ids = [row.get('response_id') for row in total_usage if row.get('response_id')]
    summary = dict(pilot_completion=pilot_completion, connectivity=connectivity, arms=arms,
        comparison=comparison, total_request_attempts=sum(arm['request_attempts'] for arm in arms.values()) + connectivity['request_attempts'],
        total_input_tokens=sum(row.get('input_tokens') or 0 for row in total_usage),
        total_output_tokens=sum(row.get('output_tokens') or 0 for row in total_usage),
        total_known_course_cost_rmb=sum(row.get('course_cost_rmb') or 0 for row in total_usage),
        duplicate_response_ids=len(request_ids) - len(set(request_ids)),
        project_usage=TaskBudgetLedger(args.ledger).project_summary(),
        course_budget_compliance='unverified_history', valid_for_budgeted_final_report=False,
        analysis_new_classifier_fits=0, analysis_new_api_requests=0)
    save(output / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('comparison', 'total_request_attempts', 'total_input_tokens',
        'total_output_tokens', 'total_known_course_cost_rmb', 'duplicate_response_ids', 'project_usage')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connectivity', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args())
