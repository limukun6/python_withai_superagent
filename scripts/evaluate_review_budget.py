"""Score the frozen budget experiment and compare actual trajectories on the same rows."""
import argparse
from collections import Counter
import csv
from pathlib import Path

import joblib
import numpy as np

from scripts.continue_review_budget import PLAN as CHALLENGE_PLAN
from scripts.run_review_budget_study import PREVIOUS, STUDY, audit, source_identity
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.metrics import MetricSpec


PAIRS = {
    'more_budget_passive': ('passive9', 'passive16'),
    'active_at_nine': ('passive9', 'active9'),
    'active_at_sixteen': ('passive16', 'active16'),
    'more_budget_active': ('active9', 'active16'),
    'combined': ('passive9', 'active16'),
}


def compare(before, after, answers, labels):
    a, b = (answers[key] for key in (before, after))
    hit_a, hit_b = a == labels, b == labels
    return dict(different_predictions=int(np.sum(a != b)), gained=int(np.sum(~hit_a & hit_b)),
                lost=int(np.sum(hit_a & ~hit_b)))


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right), 1):
        if a != b:
            return dict(step=i, before=a, after=b)
    return None



def measured_against_stopping(run, record):
    # Secondary diagnostic on identical TRAINING holdout rows, using the incumbent known before the choice.
    # A later full-run score can only be attributed to the complete policy.
    if not record['information_changed']:
        return {}
    choices, history = read(run / 'decisions.json'), read(run / 'history.json')
    payload = joblib.load(run / 'primary.joblib')
    metric = MetricSpec.from_payload(payload)
    labels = np.asarray(payload['y_search'])
    manifest = {'labels': labels, 'sha256': sha(run / 'primary.joblib')}
    observations = {}
    for row in history:
        if row['status'] == 'completed' and row['split_id'] == 'primary':
            bundle = joblib.load(run / f"trial_{row['trial']}.joblib")
            observations[row['trial']] = planner_observation(bundle, row, manifest, metric)
    measured = {}
    for event in record['information_changed']:
        step = event['step']
        row = history[step - 1]
        best_stop = next((opt for opt in choices[step - 1]['alternatives'] if opt['action'] == 'stop'), None)
        if best_stop is None or row['trial'] not in observations:
            continue
        earlier = {r['candidate']: observations[r['trial']] for r in history[:step-1] if r['trial'] in observations}
        incumbents = [earlier[m] for m in best_stop['members']]
        if len(incumbents) == 1:
            baseline = incumbents[0]['predictions']
        else:
            probabilities = np.mean([obs['probabilities'] for obs in incumbents], axis=0)
            baseline = np.asarray(metric.labels)[probabilities.argmax(axis=1)]
        incumbent_score = metric.evaluate(labels, baseline)['score']
        require(abs(incumbent_score - best_stop['evidence']['scores']['primary']) < 1e-10, 'Incumbent rows disagree')
        new = observations[row['trial']]
        measured[step] = dict(scope='same primary training holdout', rows=len(labels), members=best_stop['members'],
                              incumbent_score=incumbent_score, explored_score=new['score'],
                              delta=new['score']-incumbent_score,
                              **compare('stop', 'explore', {'stop':baseline, 'explore':new['predictions']}, labels))
    return measured


def per_cell(record, run):
    statuses = Counter(v['status'] for v in record['proposals'])
    physical = record['counts']
    information = []
    against_stop = measured_against_stopping(run, record)
    for movement in record['information_changed']:
        later = record['actions'][movement['step']:]
        matches = [index for index, action in enumerate(later, movement['step'] + 1)
                   if action == movement['before']]
        information.append(dict(movement, same_rows_vs_incumbent=against_stop.get(movement['step']),
                                previous_option_taken_later=bool(matches),
                                previous_option_later_step=matches[0] if matches else None))
    decision = read(run / 'decisions.json')[-1]
    best_alternatives = {}
    for kind in ('explore', 'confirm'):
        available = [a for a in decision['alternatives'] if a['action'] == kind]
        if available:
            alternative = available[0]
            best_alternatives[kind] = dict(count=len(available), members=alternative['members'],
                stop_advantage=decision['choice']['utility'] - alternative['utility'])
    stopping = dict(unspent_after_refit=decision['remaining_fits'] - physical['refit'],
                    available_at_stop=best_alternatives)
    return dict(stopping=stopping, fits=record['fits'], allocation=physical, measured_both=physical['explore'] > 0 and physical['confirm'] > 0,
                proposed=len(record['proposals']), ruled=statuses['accepted'] + statuses['rejected'],
                accepted=statuses['accepted'], pending=statuses['shadow'], active_moves=len(information),
                changes_after_adopting_rule=record['adopted_rule_changes'], information=information,
                winner=record['winner'], prediction=record['prediction'], model=record['model'],
                seconds=round(record['seconds'], 3))


def run(root, prior):
    plan, config = read(STUDY), read(root / 'protocol.json')
    seals = read(root / 'sealed-predictions.json')
    previous = read(PREVIOUS)
    legacy, independent = plan['legacy_cases'], plan['independent_check']['id']
    require(sha(STUDY) == config['preregistration'] and source_identity(ROOT) == config['runtime'] and
            sha(Path(__file__).with_name('run_review_budget_study.py')) == config['harness'], 'Experiment changed')
    records, paths = {}, {}
    expected = set()
    for case, spec in config['cases'].items():
        records[case], paths[case] = {}, {}
        for cell in (plan['cells'] if case in legacy else plan['independent_check']['arms']):
            existing = cell == 'passive9'
            path = prior / 'paired' / case / 'review' if existing else root / case / cell
            record = audit(path, cell)
            if existing:
                require(record['prediction'] == previous['paired_cases'][case]['prediction_sha256']['review'],
                        'Existing nine-fit control changed')
            else:
                key = case + '/' + cell
                expected.add(key)
                require(record == read(path / 'sealed.json') and record['prediction'] == seals[key],
                        'Previously frozen predictions or run records differ')
            require(record['metric'] == spec['metric'], 'Declared objective differs from score')
            records[case][cell] = record
            paths[case][cell] = path / 'delivery' / 'predictions.csv'
        first = next(iter(records[case].values()))
        require(all(r['primary_split'] == first['primary_split'] and r['actions'][0] == first['actions'][0]
                    for r in records[case].values()), 'Different initial state within a task')
    check_case = legacy[0]
    sentinel = audit(root / check_case / 'passive9', 'passive9')
    key = check_case + '/passive9'
    expected.add(key)
    require(seals[key] == sentinel['prediction'] == records[check_case]['passive9']['prediction'] and
            sentinel['actions'] == records[check_case]['passive9']['actions'], 'Nine-fit replication mismatch')
    require(expected == set(seals), 'Some experiment outputs are missing')
    challenge_dir = root / 'stop-challenge'
    extra = {}
    if challenge_dir.exists():
        signed = read(challenge_dir / 'protocol.json')
        frozen_extra = read(challenge_dir / 'sealed-predictions.json')
        require(signed['protocol'] == sha(CHALLENGE_PLAN) and signed['parent'] == sha(root / 'sealed-predictions.json')
                and set(frozen_extra) == set(records), 'Continuation runs have not all been sealed')
        for case in frozen_extra:
            path = challenge_dir / case
            item = read(path / 'sealed.json')
            gate = [v['delta'] for v in item['comparisons']]
            accepted = len(gate) == 2 and min(gate) > 0 and float(np.mean(gate)) >= .003
            require(accepted == (item['status'] == 'accepted') and item['prediction'] == frozen_extra[case] ==
                    sha(path / 'delivery/predictions.csv') == sha(path / 'replay/predictions.csv') and
                    item['parent_fits'] + item['new_fits'] <= 16, 'Continuation gate, prediction or budget mismatch')
            extra[case] = item
    results = {}
    for case, controls in records.items():
        spec = config['cases'][case]
        truth = root / 'evaluation' / (case + '.npy')
        require(sha(truth) == spec['truth_sha256'], 'Held-out rows changed')
        labels = np.load(truth, allow_pickle=False)
        scorer = MetricSpec(spec['metric'], tuple(np.unique(labels).tolist()))
        answers, cells, comparisons = {}, {}, {}
        for cell, record in controls.items():
            with paths[case][cell].open(encoding='utf-8') as handle:
                data = list(csv.DictReader(handle))
            require(len(data) == spec['held_rows'] == len(labels) and
                    [r['row_id'] for r in data] == list(map(str, range(len(labels)))), 'Evaluation row mismatch')
            answers[cell] = np.asarray([r['prediction'] for r in data], dtype=labels.dtype)
            primary = scorer.evaluate(labels, answers[cell])['score']
            if cell == 'passive9':
                require(abs(primary - previous['paired_cases'][case]['scores']['review']) < 1e-12,
                        'Historical comparison did not reproduce its recorded score')
            run_path = paths[case][cell].parents[1] / 'run' / 'agent'
            cells[cell] = dict(per_cell(record, run_path), score=float(primary), correct=int(np.sum(answers[cell] == labels)))
        for name, (before, after) in PAIRS.items():
            if before in cells and after in cells:
                comparisons[name] = dict(compare(before, after, answers, labels),
                    delta=cells[after]['score'] - cells[before]['score'],
                    first_divergence=first_difference(controls[before]['actions'], controls[after]['actions']))
        results[case] = dict(metric=spec['metric'], rows=len(labels), cells=cells, comparisons=comparisons)
        if case in extra:
            with (challenge_dir / case / 'delivery/predictions.csv').open(encoding='utf-8') as handle:
                followup = list(csv.DictReader(handle))
            require([r['row_id'] for r in followup] == list(map(str, range(len(labels)))), 'Continuation row mismatch')
            delivered = np.asarray([r['prediction'] for r in followup], dtype=labels.dtype)
            score = float(scorer.evaluate(labels, delivered)['score'])
            results[case]['stop_challenge'] = dict(extra[case], score=score,
                score_gain=score-cells['active16']['score'],
                **compare('stop','continue', {'stop':answers['active16'], 'continue':delivered}, labels))


    summaries = {}
    for cell in plan['cells']:
        rows = [results[c]['cells'][cell] for c in legacy]
        distribution = {k:sum(r['allocation'][k] for r in rows) for k in ('explore','confirm','refit')}
        events = [v for row in rows for v in row['information']]
        changes_from_confirm = [v for v in events if v['before'][0] == 'confirm']
        summaries[cell] = dict(fits=sum(distribution.values()), allocation=distribution,
            proposals=sum(r['proposed'] for r in rows), completed_verdicts=sum(r['ruled'] for r in rows),
            accepted=sum(r['accepted'] for r in rows), pending=sum(r['pending'] for r in rows),
            information_changed_choices=len(events), changed_from_confirm=len(changes_from_confirm),
            exact_confirmation_later=sum(v['previous_option_taken_later'] for v in changes_from_confirm),
            distinct_tasks_with_exploration_and_confirmation=sum(r['measured_both'] for r in rows))
    effects = {}
    for name in PAIRS:
        values = [results[c]['comparisons'][name] for c in legacy]
        differences = [v['delta'] for v in values]
        effects[name] = dict(mean_delta=float(np.mean(differences)), improved=sum(v > 1e-12 for v in differences),
            declined=sum(v < -1e-12 for v in differences), tied=sum(abs(v) <= 1e-12 for v in differences),
            gained_rows=sum(v['gained'] for v in values), lost_rows=sum(v['lost'] for v in values))
    used = sum(r['fits'] for c in records.values() for cell,r in c.items() if cell != 'passive9') + sentinel['fits']
    require(used <= plan['spending']['new_fits_max'] and all(r['api_calls'] == 0 for rows in records.values()
            for r in rows.values()), 'Study exceeded its approved budget')
    final = dict(runtime=config['runtime'], study=config['preregistration'], training_harness=config['harness'],
                 score_program=sha(Path(__file__)), independent_case=independent, cases=results,
                 four_task_summaries=summaries, four_task_effects=effects, new_classifier_fits=used,
                 reused_classifier_fits=sum(records[c]['passive9']['fits'] for c in legacy),
                 sentinel_reproduced=True, sealed_new_deliveries=len(seals), model_api_calls=0,
                 model_api_cost_rmb=0, formal_test_accessed=False)
    if extra:
        supplemental = sum(r['new_fits'] for r in extra.values())
        require(used + supplemental <= plan['spending']['new_fits_max'], 'Combined spending differs from plan')
        final.update(continuation_fits=supplemental, total_including_continuation=used+supplemental,
                     continuation_program=read(challenge_dir/'protocol.json')['implementation'],
                     continuation_plan=sha(CHALLENGE_PLAN))
    save(root / 'completion.json', final)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    cli.add_argument('prior', type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.prior.resolve())
