"""Measure one affordable challenge to an autonomous stopping decision, preserving its frozen model."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np

from scripts.run_review_budget_study import ROOT, STUDY, WALL, source_identity
from scripts.unified_released_study import env, read, require, save, sha
from superagent.ab_run import Executor, registered_capabilities
from superagent.autonomous_run import freeze_model, installed_methods, planner_observation
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.open_methods import capability


PLAN = ROOT / 'research/rsi_budget_1125/continuation_protocol.json'


def incumbent(run, history, members, fold, evidence, metric):
    records = {r['candidate']:r for r in history if r['split_id'] == fold and r['status'] == 'completed'}
    observations = []
    for name in members:
        row = records[name]
        bundle = joblib.load(run / f"trial_{row['trial']}.joblib")
        observations.append(planner_observation(bundle, row, evidence, metric))
    if len(observations) == 1:
        return observations[0]['predictions']
    mean_probabilities = np.mean([r['probabilities'] for r in observations], axis=0)
    return np.asarray(metric.labels)[mean_probabilities.argmax(axis=1)]


def run_case(root, case):
    started = time.monotonic()
    parent = root / case / 'active16'
    run = parent / 'run' / 'agent'
    work = root / 'stop-challenge' / case
    recipient = work / 'run' / 'agent'
    recipient.mkdir(parents=True)
    log = RunLogger(recipient)
    log.initialize_usage_ledger()
    baseline = read(parent / 'sealed.json')
    time_left = WALL - baseline['seconds']
    limit = 16 - baseline['fits']
    decision = read(run / 'decisions.json')[-1]
    stop = decision['choice']
    history = read(run / 'history.json')
    require(stop['action'] == 'stop' and stop['members'] == baseline['winner'], 'Incumbent mismatch')
    require(sha(run / 'selected_model.joblib') == baseline['model'], 'Frozen incumbent changed')
    require(sha(run / decision['registry']) == decision['registry_sha256'], 'Frozen terminal registry changed')
    registry = read(run / decision['registry'])
    shutil.copyfile(run / decision['registry'], recipient / 'candidate_registry.json')
    task = load_training_task(root / case / 'p.zip', read_declarations=True)
    spec = read(run / 'task_contract.json')
    metric = MetricSpec.for_task(task)
    width = task.X_train.shape[1] if task.kind == 'numeric_tabular' else None
    caps = registered_capabilities(spec, len(task.y_train), numeric_kernel_width=width)
    caps.update({name:capability(entry) for name,entry in registry['entries'].items()})
    options = [o['members'][0] for o in decision['alternatives'] if o['action'] == 'explore']
    candidate = next((n for n in options if 2*caps[n]['classifier_fits'] + caps[n]['refit_classifier_fits'] <= limit), None)
    confirmations = sorted(f for f in stop['evidence']['scores'] if f.startswith('confirmation_'))
    challenge = bool(candidate and confirmations and time_left > 0)
    folds = ['primary', confirmations[0]] if challenge else []
    save(work / 'choice.json', dict(parent_model=baseline['model'], incumbent=stop['members'],
         candidate=candidate, terminal_explorations=len(options), remaining=limit, matched_folds=folds,
         training_data_sha256=sha(root / case / 'p.zip')))
    executor = Executor(recipient, caps, metric, started, time_left, 300, limit, log)
    scores, comparisons = {}, []
    accepted = False
    if challenge:
        for i, fold in enumerate(folds):
            path = run / (fold + '.joblib')
            payload = joblib.load(path)
            evidence = dict(sha256=sha(path), labels=payload['y_search'])
            reserve = caps[candidate]['refit_classifier_fits'] + (1-i)*caps[candidate]['classifier_fits']
            destination = recipient / ('challenge_' + fold + '.joblib')
            _, facts = executor.worker(path, candidate, destination, 300, reserve=reserve)
            challenger = planner_observation(joblib.load(destination), facts, evidence, metric)
            pred = incumbent(run, history, stop['members'], fold, evidence, metric)
            original = metric.evaluate(evidence['labels'], pred)['score']
            require(abs(original - stop['evidence']['scores'][fold]) < 1e-10, 'Incumbent row alignment changed')
            a, b = pred == evidence['labels'], challenger['predictions'] == evidence['labels']
            scores[fold] = challenger['score']
            comparisons.append(dict(fold=fold, rows=len(a), incumbent=original, challenger=challenger['score'],
                delta=challenger['score']-original, gained=int(np.sum(~a & b)), lost=int(np.sum(a & ~b))))
        deltas = [c['delta'] for c in comparisons]
        accepted = min(deltas) > 0 and float(np.mean(deltas)) >= .003
    final_run = parent / 'run'
    if accepted:
        choice = dict(action='stop', members=[candidate], utility=float(np.mean(list(scores.values()))),
                      evidence={'scores':scores}, reason='Improved on both matched training folds')
        freeze_model(task, spec, metric, choice, executor, recipient, log, 300)
        final_run = work / 'run'
    predictions = []
    for key in ('delivery', 'replay'):
        path = work / key / 'predictions.csv'
        subprocess.run([sys.executable, '-m', 'unified_agent', 'predict', str(root / case / 'q.zip'),
                        '--run-dir', str(final_run), '--output', str(path)], cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, check=True, timeout=WALL)
        predictions.append(sha(path))
    require(predictions[0] == predictions[1] and (accepted or predictions[0] == baseline['prediction']),
            'Continuation delivery differs after reload')
    physical = work / 'fits.jsonl'
    count = len(physical.read_text().splitlines()) if physical.exists() else 0
    require(count == executor.accounting()['attempted_classifier_fits'] == executor.accounting()['completed_classifier_fits']
            and baseline['fits'] + count <= 16, 'Joint path exceeded its budget')
    save(work / 'sealed.json', dict(incumbent=stop['members'], candidate=candidate,
         status='accepted' if accepted else ('retained' if challenge else 'insufficient_room_or_evidence'),
         comparisons=comparisons, parent_fits=baseline['fits'], new_fits=count, combined_fits=baseline['fits']+count,
         parent_model=baseline['model'], delivered_model=sha(final_run/'agent'/'selected_model.joblib'),
         prediction=predictions[0], model_api_calls=0, external_truth_opened=False))
    print(case, 'accepted' if accepted else 'kept incumbent', count, 'additional physical fits', flush=True)


def run(root, weights):
    plan = read(STUDY)
    config = read(root / 'protocol.json')
    frozen = read(root / 'sealed-predictions.json')
    require(config['runtime'] == source_identity(ROOT), 'Parent runtime changed')
    base_fits = sum(read(root / key / 'sealed.json')['fits'] for key in frozen)
    dest = root / 'stop-challenge'
    dest.mkdir()
    save(dest / 'protocol.json', dict(protocol=sha(PLAN), implementation=sha(Path(__file__)), parent=sha(root/'sealed-predictions.json'),
                                     paid_parent_fits=base_fits))
    prior_cost = base_fits
    for case in config['cases']:
        case_dir = dest / case
        parent = read(root / case / 'active16' / 'sealed.json')
        room = 16 - parent['fits']
        require(prior_cost + room <= plan['spending']['new_fits_max'], 'Insufficient remaining study allowance')
        settings = env(weights, case_dir / 'fits.jsonl')
        settings.pop('AB_B1_PER_WORKER_FIT_LIMIT', None)
        for key in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID'):
            settings.pop(key, None)
        subprocess.run([sys.executable, '-m', 'scripts.continue_review_budget', 'case', str(root), '--case', case],
                       cwd=ROOT, env=settings, check=True, stdin=subprocess.DEVNULL, timeout=WALL+90)
        prior_cost += read(case_dir / 'sealed.json')['new_fits']
    sealed = {case:read(dest / case / 'sealed.json')['prediction'] for case in config['cases']}
    save(dest / 'sealed-predictions.json', sealed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run','case'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--weights', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.root.resolve(), args.weights.resolve())
    else:
        with installed_methods():
            run_case(args.root.resolve(), args.case)
