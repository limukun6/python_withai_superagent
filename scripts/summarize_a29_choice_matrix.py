"""Summarize a finished six-arm matrix; no fitting, API, raw data or model loading."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read
from superagent.task_budget import TaskBudgetLedger


ARMS = ('L0', 'L1', 'R0', 'E0', 'R1', 'E1')
PAIRS = [('L1','L0'), ('E0','R0'), ('E1','R1'), ('R1','R0'), ('E1','E0'), ('E1','L1')]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    completion = read(args.matrix / 'completion.json')
    coverage=args.study=='route-coverage'
    arms=('L0','L1',*[base+'_'+mode for base in ('R0','E0','R1','E1') for mode in ('off','routes')]) if coverage else ARMS
    pairs=([('L1','L0')]+[(base+'_routes',base+'_off') for base in ('R0','E0','R1','E1')]+
        [(a+'_'+mode,b+'_'+mode) for mode in ('off','routes') for a,b in [('E0','R0'),('E1','R1'),('R1','R0'),('E1','E0')]]) if coverage else PAIRS
    if completion['status'] != 'completed' or completion['paths_completed'] != 2*len(arms):
        raise ValueError('Only complete matrices can be scored/summarized as six-arm results')
    for name in ('protocol.json', 'completion.json', 'all_models_frozen.json', 'physical_fits.jsonl'):
        shutil.copy2(args.matrix / name, output / name)
    tasks = {}
    all_usage = []
    all_responses = []
    for taskname in ('coupon', 'preference'):
        details = {}
        predictions = {}
        contexts = {}
        for arm in arms:
            directory = args.matrix / (taskname + '_' + arm)
            target = output / directory.name
            target.mkdir()
            for name in ('run_manifest.json', 'run_outcome.json', 'actions.json', 'history.json',
                         'classifier_fits.json', 'final_selection.json', 'verification.json', 'llm_usage.jsonl', 'events.jsonl'):
                shutil.copy2(directory / name, target / name)
            outcome, manifest = read(directory / 'run_outcome.json'), read(directory / 'run_manifest.json')
            score = read(directory / 'verification.json')
            history = read(directory / 'history.json')
            events = records(directory / 'events.jsonl')
            usage = [row for row in records(directory / 'llm_usage.jsonl') if row['event'] != 'ledger_initialized']
            all_usage.extend(usage)
            all_responses.extend(row['response_id'] for row in usage if row.get('response_id'))
            with (directory / 'predictions' / 'rows.csv').open(encoding='utf-8', newline='') as file:
                predictions[arm] = [row['prediction'] for row in csv.DictReader(file)]
            primary = [row for row in history if row['split_id'] == 'primary']
            selected_at = next(row['trial'] for row in primary if row['candidate'] == outcome['selected_candidate'])
            summary = dict(score=score['score'], primary_metric=score['primary_metric'], rows=score['rows'],
                selected_candidate=outcome['selected_candidate'], selected_first_observed_trial=selected_at,
                actual_classifier_fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                action_counts=outcome['action_counts'], path_seconds=outcome['elapsed_seconds'],
                requests=outcome['new_api_requests'], known_course_cost_rmb=outcome['known_course_cost_rmb'],
                input_tokens=sum(row.get('input_tokens') or 0 for row in usage),
                output_tokens=sum(row.get('output_tokens') or 0 for row in usage),
                call_latencies_seconds=[row['latency_seconds'] for row in usage if 'latency_seconds' in row],
                coverage_projection_events=[{key:row.get(key) for key in ('trial','target_routes','observed_primary_routes',
                    'projected_routes','reason','original_eligible_ids','projected_eligible_ids')}
                    for row in events if row['kind']=='candidate_coverage_projection'],
                explored_candidates=[row['candidate'] for row in primary], candidate_count=len(manifest['capabilities']),
                explored_routes=sorted({manifest['capabilities'][row['candidate']]['route'] for row in primary}),
                available_routes=sorted({cap['route'] for cap in manifest['capabilities'].values()}),
                untried_candidates=len(manifest['capabilities']) - len(primary),
                source_unchanged=outcome['source_unchanged'], reload_identical=score['reload_identical'],
                independent_winner_confirmation=outcome.get('confirmation', {}).get('independent_confirmation_observed'),
                selected_candidate_confirmation_splits=sorted({row['split_id'] for row in history
                    if row['candidate']==outcome['selected_candidate'] and row['split_id']!='primary'}),
                configuration_checks=dict(Counter((row.get('configuration_fact_check') or {}).get('status') for row in history)),
                hypothesis_checks=dict(Counter((row.get('hypothesis_check') or {}).get('status') for row in history)),
                scope='Designated development val, not formal test')
            if arm.startswith(('R','E')):
                request = next(row for row in events if row['kind'] == 'llm_request')
                context = json.loads(request['messages'][1]['content'])
                common = {key:value for key,value in context.items() if key not in ('checks','memory')}
                summary.update(first_request_context_sha256=fingerprint(context),
                    first_request_messages_sha256=fingerprint(request['messages']),
                    first_request_common_context_sha256=fingerprint(common),
                    first_request_checks_present='checks' in context, first_request_memory_present='memory' in context)
                contexts[arm] = common
            details[arm] = summary
        differences = {}
        for left, right in pairs:
            differences[left + '-' + right] = dict(score_delta=details[left]['score'] - details[right]['score'],
                classifier_fit_delta=details[left]['actual_classifier_fits'] - details[right]['actual_classifier_fits'],
                prediction_disagreements=sum(a != b for a,b in zip(predictions[left], predictions[right], strict=True)),
                known_course_cost_delta_rmb=details[left]['known_course_cost_rmb'] - details[right]['known_course_cost_rmb'])
        if coverage:
            prompt_checks={mode:dict(common_context_identical=all(contexts[base+'_'+mode]==contexts['R0_'+mode] for base in ('R0','E0','R1','E1')),
                raw_messages_identical=details['R0_'+mode]['first_request_messages_sha256']==details['R1_'+mode]['first_request_messages_sha256'],
                checked_messages_identical=details['E0_'+mode]['first_request_messages_sha256']==details['E1_'+mode]['first_request_messages_sha256'])
                for mode in ('off','routes')}
        else:
            prompt_checks=dict(first_flash_common_context_identical=all(context == contexts['R0'] for context in contexts.values()),
                raw_first_prompt_repeated=details['R0']['first_request_messages_sha256']==details['R1']['first_request_messages_sha256'],
                checked_first_prompt_repeated=details['E0']['first_request_messages_sha256']==details['E1']['first_request_messages_sha256'])
        tasks[taskname] = dict(arms=details,differences=differences,prompt_checks=prompt_checks)
        target = output / 'evaluation' / taskname
        target.mkdir(parents=True)
        shutil.copy2(args.matrix / 'evaluation' / taskname / 'source_access.json', target / 'source_access.json')
    summary = dict(completion=completion, tasks=tasks,
        input_tokens=sum(row.get('input_tokens') or 0 for row in all_usage),
        output_tokens=sum(row.get('output_tokens') or 0 for row in all_usage),
        known_new_course_cost_rmb=sum(row.get('course_cost_rmb') or 0 for row in all_usage),
        unknown_usage_calls=sum(row.get('usage_known') is False for row in all_usage),
        duplicate_response_ids=len(all_responses)-len(set(all_responses)),
        project_usage=TaskBudgetLedger(args.ledger).project_summary(),
        analysis_new_classifier_fits=0, analysis_new_api_requests=0,
        course_budget_compliance='unverified_history', valid_for_budgeted_final_report=False,
        study=args.study,scope='Two development tasks, one development split each, one run per arm; not final five-task C1 or H')
    save(output / 'summary.json', summary)
    print(json.dumps(dict(completion=completion, scores={name:{arm:row['score'] for arm,row in value['arms'].items()}
        for name,value in tasks.items()}, input_tokens=summary['input_tokens'], output_tokens=summary['output_tokens'],
        known_new_course_cost_rmb=summary['known_new_course_cost_rmb'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study',choices=['six-arm','route-coverage'],default='six-arm')
    run(parser.parse_args())
