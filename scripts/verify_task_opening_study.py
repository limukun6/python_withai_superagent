"""Replay opening diagnoses from stored training predictions, without fitting or opening the external labels."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import joblib
import numpy as np

from scripts.run_task_opening_study import CASES, PLAN
from scripts.unified_released_study import read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.conditional_review import MEAN_GAIN, group_observation
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.opening_evidence import instructions, split_audit, subset


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_errors(task, metric, held, check, prediction, claimed):
    mask = subset(task, metric, held, check)
    wrong = np.asarray(prediction) != task.y_train[held]
    counts = (int(mask.sum()), int((~mask).sum()), int((wrong & mask).sum()), int((wrong & ~mask).sum()))
    n, other, errors, other_errors = counts
    require((n, other) == (claimed['rows'], claimed['others']), 'Incorrect diagnostic population')
    if min(n, other) < 5:
        require(claimed['supported'] is None, 'Too few examples for a verdict')
        return
    gap = errors / n - other_errors / other
    require((errors, other_errors) == (claimed['errors'], claimed['other_errors']) and
            abs(gap - claimed['gap']) < 1e-12 and claimed['supported'] == (gap + 1e-12 >= .05),
            'The reported diagnosis disagrees with the underlying row predictions')


def net_rescues(base, candidate, mask):
    labels = base['labels'][mask]
    return int(np.sum(candidate['predictions'][mask] == labels) - np.sum(base['predictions'][mask] == labels))


def measured_alternatives(task, metric, brief, folds, record, bank):
    """Post-hoc training-only inventory of other questions the reader could have asked."""
    names = record.get('confirmation', [])
    diagnostics = {}
    if len(names) != 2:
        return diagnostics
    members = record['incumbent_members']
    incumbent = {name: group_observation(bank[name], members, metric) for name in names}
    first, second = names
    for key, check in brief['checks'].items():
        if check['kind'] not in ('class', 'slice'):
            continue
        masks, gaps = {}, {}
        for name in names:
            row = incumbent[name]
            mask = subset(task, metric, folds[name][1], check)
            masks[name] = mask
            gaps[name] = None
            if min(int(mask.sum()), int((~mask).sum())) >= 5:
                errors = row['labels'] != row['predictions']
                gaps[name] = float(errors[mask].mean() - errors[~mask].mean())
        supported = all(g is not None and g + 1e-12 >= .05 for g in gaps.values())
        rescuers = 0
        matched = 0
        passed = 0
        if supported:
            for candidate, row in bank[first].items():
                if candidate in members:
                    continue
                if net_rescues(incumbent[first], row, masks[first]) <= 0:
                    continue
                rescuers += 1
                if candidate not in bank[second]:
                    continue
                matched += 1
                other = bank[second][candidate]
                deltas = (row['score'] - incumbent[first]['score'], other['score'] - incumbent[second]['score'])
                if (min(deltas) > 0 and np.mean(deltas) >= MEAN_GAIN and
                        net_rescues(incumbent[second], other, masks[second]) > 0):
                    passed += 1
        diagnostics[key] = dict(error_gaps=gaps, corroborated=supported, alternatives_rescuing_primary=rescuers,
                                of_those_measured_on_second=matched, passing_both_rules=passed)
    return diagnostics


def audit_case(folder):
    run = folder / 'flash/run/agent'
    record = read(run / 'task_opening.json')
    lesson = read(run / 'opening_lesson.json')
    brief = read(run / 'opening_brief.json')
    sequence = events(run / 'events.jsonl')
    requests = [e for e in sequence if e['kind'] == 'llm_request']
    require(len(requests) == 1 and requests[0]['role'] == 'task_opening', 'Unexpected opening request')
    request = requests[0]
    message = json.loads(request['messages'][-1]['content'])
    message.pop('instruction')
    require(message == brief and brief == read(folder / 'local/run/agent/opening_brief.json'),
            'The evidence sent to the reader differs from the matched local copy')
    # The admission file is created immediately before the worker is launched, before fit() begins.
    lead = (run / 'trial_1.admission.json').stat().st_mtime - datetime.fromisoformat(request['time_utc']).timestamp()
    require(lead >= 0, 'The opening call was dispatched after the first worker was admitted')
    diagnosis = record.get('diagnosis', {})
    require(lesson['check_status'] == record['check_status'] and lesson['observed'] == diagnosis and
            lesson['result'] == record['result'], 'Saved lesson differs from the measured outcome')
    task, _, _ = instructions(folder / 'train.zip', load_training_task(folder / 'train.zip', read_declarations=True))
    metric = MetricSpec.for_task(task)
    folds = {}
    for path in sorted(run.glob('*_split.json')):
        split = read(path)
        folds[path.name[:-len('_split.json')]] = (np.array(split['fit']), np.array(split['held']), split['resource'])
    key = record.get('check')
    check = brief['checks'].get(key)
    updates_path = run / 'opening_updates.json'
    updates = read(updates_path) if updates_path.is_file() else []
    require(len(updates) == record['updates_during_search'], 'Missing in-search measurements')
    timeline = {update['trial']: update for update in updates}
    require(len(timeline) == len(updates), 'Duplicate observation for a single trial')
    bank = {fold: {} for fold in folds}
    def check_measurements(measurements, members):
        if check['kind'] in ('entity', 'temporal'):
            require(measurements == split_audit(task, folds), 'Actual split separation differs')
            return
        for fold, claimed in measurements.items():
            row = group_observation(bank[fold], members, metric)
            require(row is not None, 'The hypothesis cited an unmeasured model')
            check_errors(task, metric, folds[fold][1], check, row['predictions'], claimed)
    for trial in read(run / 'history.json'):
        if trial['status'] != 'completed':
            continue
        number, fold = trial['trial'], trial['split_id']
        held = folds[fold][1]
        bundle = joblib.load(run / f'trial_{number}.joblib')
        payload = {'labels': task.y_train[held], 'sha256': sha(run / (fold + '.joblib'))}
        bank[fold][trial['candidate']] = planner_observation(bundle, trial, payload, metric)
        if number in timeline:
            update = timeline[number]
            require(update['check'] == key, 'A later diagnosis was substituted into the timeline')
            check_measurements(update['observations'], [update.get('reference')])
    if check:
        check_measurements(diagnosis, record['incumbent_members'])
    for mode in ('local', 'flash'):
        receipt = read(folder / mode / 'sealed.json')
        for key, directory in [('base', 'original-predictions'), ('final', 'replay')]:
            require(sha(folder / mode / directory / 'predictions.csv') == receipt[key],
                    'A reloaded model produced different predictions')
    verified = dict(sent_before_first_worker=True, original_brief_matched=True, saved_lesson_verified=True,
                    diagnosis=record['check_status'], verified_diagnostic_folds=len(diagnosis),
                    progressive_updates_replayed=len(updates), fresh_prediction_replays=4)
    return verified, measured_alternatives(task, metric, brief, folds, record, bank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--training-inventory', type=Path)
    args = parser.parse_args()
    report = read(args.panel / 'completion.json')
    checks, opportunities = {}, {}
    for case in CASES:
        checks[case], opportunities[case] = audit_case(args.panel / case)
    outbox = read(args.panel / 'pending_merge.json')
    transitions = outbox['transitions']
    for request in outbox['requests']:
        sequence = [row['event'] for row in transitions if row['request_id'] == request['id']]
        require(sequence == ['reserved', 'known'], 'A request was not settled exactly once')
    require(len(outbox['requests']) == report['calls'], 'The number of accounted requests differs')
    save(args.output, dict(protocol_sha256=sha(PLAN), amendment_sha256=sha(PLAN.with_name('intake_amendment.json')),
            scored_result_sha256=sha(args.panel / 'completion.json'),
            frozen_predictions_sha256=sha(args.panel / 'all_predictions_frozen.json'),
            cases=checks, settled_calls=report['calls'], team_merge=outbox['state'], actual_provider_debit=None))
    if args.training_inventory:
        save(args.training_inventory, dict(
            scope='Post-hoc description of the original training trials only. No external data, new fits, calls or policy changes.',
            cases=opportunities))


if __name__ == '__main__':
    main()
