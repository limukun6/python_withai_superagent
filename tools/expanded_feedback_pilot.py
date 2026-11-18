"""Prepare and run a paired study on training-derived holdouts only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import joblib
import numpy as np

from superagent.adaptive_local import save, sha
from superagent.benchmark import run_one
from superagent.cli import CAMPUS_CHAT_ENDPOINT, CAMPUS_FLASH_MODEL
from superagent.data import DevelopmentTask, TrainingTask, load_training_task
from superagent.evidence_schedule import duel
from superagent.experiment_llm import ExperimentLLM
from superagent.models import predict_with_model, search_candidates
from superagent.pricing import resolve_pricing
from superagent.splits import effective_grouping, grouped_partition, overlap_audit, take
from superagent.task_budget import TaskBudgetAccount, TaskBudgetLedger


TASKS = ('task1', 'task2')
SEARCH_SEED = 707
RESERVE_SEED = 41101
OUTER_SEED = 42011
ALLOWANCE_RMB = 0.10
RUN_OPTIONS = dict(max_trials=4, time_limit=600, trial_timeout=90, anchor_first=True,
                   refit_full_train=True, confirmation_repeats=3, confirmation_top_k=3,
                   confirmation_grouping='shared_text', search_grouping='shared_text',
                   portfolio='expanded', feature_cache=True)


def prepare_task(path):
    whole = load_training_task(path)
    if whole.spec.get('metric', 'accuracy') != 'accuracy':
        raise ValueError('This pilot requires accuracy for both released tasks')
    grouping = effective_grouping(whole.kind, 'shared_text')
    keep, _, _ = grouped_partition(whole, RESERVE_SEED, grouping)
    dev = TrainingTask(whole.kind, take(whole.X_train, keep), whole.y_train[keep], whole.spec)
    train, held, _ = grouped_partition(dev, OUTER_SEED, grouping)
    audit = overlap_audit(dev, train, held)
    if audit.get('search_duplicate_rows', 0) or audit.get('search_rows_sharing_text', 0):
        raise ValueError('Outer holdout overlaps model-development inputs')
    fit = DevelopmentTask(dev.kind, 'training_partition', take(dev.X_train, train), dev.y_train[train],
                          None, None, dev.spec)
    record = dict(archive_sha256=sha(path), kind=dev.kind, train_rows=keep[train].tolist(),
                  held_rows=keep[held].tolist(), audit=audit,
                  candidates=[c.family for c in search_candidates(dev.kind, 'expanded')])
    return fit, take(dev.X_train, held), dev.y_train[held], record


def protocol(tasks):
    package = Path(__file__).resolve().parents[1] / 'superagent'
    return dict(tasks={name: task[3] for name, task in tasks.items()},
                source={p.name: sha(p) for p in sorted(package.glob('*.py'))}, pilot_sha256=sha(Path(__file__)),
                python=platform.python_version(), dependencies={name: importlib.metadata.version(name)
                    for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'catboost', 'requests', 'Pillow')},
                search_seed=SEARCH_SEED, reserve_seed=RESERVE_SEED, outer_seed=OUTER_SEED,
                run_options=RUN_OPTIONS, model=CAMPUS_FLASH_MODEL, allowance_rmb_per_task=ALLOWANCE_RMB)


def freeze_run(task, strategy, directory, client):
    result = run_one(task, strategy, SEARCH_SEED, directory, client, **RUN_OPTIONS)
    expected_calls = RUN_OPTIONS['max_trials'] - 1 if client is not None else 0
    fits = result['classifier_fit_accounting']['completed_classifier_fits']
    if not (result['valid_policy_run'] and result['usage_complete'] and result['refit_succeeded']
            and result['confirmation']['valid_confirmation'] and result['test_accessed'] is False
            and result['trials_attempted'] == RUN_OPTIONS['max_trials'] and result['failed_trials'] == 0
            and result['llm_calls'] == expected_calls and fits == 11):
        raise ValueError(f'Incomplete {strategy} run. See {directory / "summary.json"}')
    record = dict(candidate=result['selected_candidate'], fits=fits, elapsed_seconds=result['elapsed_seconds'],
                  calls=result['llm_calls'], tokens=result['tokens'], course_cost_rmb=result['course_cost_rmb'],
                  model_sha256=sha(directory / 'selected_model.joblib'))
    save(directory / 'pilot_frozen.json', record)
    return record


def frozen_control(directory):
    record = json.loads((directory / 'pilot_frozen.json').read_text(encoding='utf-8'))
    if record['calls'] != 0 or record['fits'] != 11:
        raise ValueError('Unexpected fixed baseline budget')
    if sha(directory / 'selected_model.joblib') != record['model_sha256']:
        raise ValueError(f'Frozen model hash changed: {directory}')
    return record


def score_pair(root, features, labels, arms):
    masks = {}
    for arm, record in arms.items():
        artifact = root / arm / 'selected_model.joblib'
        if sha(artifact) != record['model_sha256']:
            raise ValueError(f'Frozen model hash changed: {artifact}')
        bundle = joblib.load(artifact)
        if bundle['candidate'] != record['candidate']:
            raise ValueError(f'Frozen candidate changed: {artifact}')
        predicted = np.asarray(predict_with_model(bundle['kind'], bundle['model'], features)).reshape(-1)
        if predicted.shape != labels.shape:
            raise ValueError(f'Unexpected prediction shape: {artifact}')
        masks[arm] = predicted == labels
        record.update(accuracy=float(masks[arm].mean()), correct=int(masks[arm].sum()))
    return dict(arms=arms, paired=duel(masks['fixed'], masks['feedback']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'compare'))
    parser.add_argument('--task1', type=Path, required=True)
    parser.add_argument('--task2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--budget-ledger', type=Path)
    args = parser.parse_args()
    if args.mode == 'compare' and (not args.api_key_file or not args.budget_ledger):
        parser.error('compare needs an API key file and an initialized course-cost ledger')
    if args.mode == 'compare' and os.environ.get('SUPERAGENT_OFFLINE') == '1':
        parser.error('LLM calls are disabled in offline mode')
    root = args.output.resolve()
    manifest = root / 'pilot_protocol.json'
    tasks = {name: prepare_task(getattr(args, name)) for name in TASKS}
    expected_kinds = ('categorical_tabular', 'pair_text')
    if tuple(tasks[name][0].kind for name in TASKS) != expected_kinds:
        parser.error('Expected the released tabular task followed by the released text-pair task')
    plan = protocol(tasks)
    if args.mode == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
        save(manifest, plan)
        for name, (task, _, _, _) in tasks.items():
            result = freeze_run(task, 'local_interleaved_routes', root / name / 'fixed', None)
            print(f'{name} fixed: {result["candidate"]}, {result["fits"]} fits', flush=True)
        return

    if json.loads(manifest.read_text(encoding='utf-8')) != plan:
        parser.error('The frozen code, environment, inputs or splits changed after prepare')
    if any((root / name / 'feedback').exists() for name in TASKS):
        parser.error('A feedback run already exists in this directory. Inspect it before any new spend')
    controls = {name: frozen_control(root / name / 'fixed') for name in TASKS}
    ledger = TaskBudgetLedger(args.budget_ledger)
    for name in TASKS:
        ledger.preflight(name, ALLOWANCE_RMB)
    price = resolve_pricing(CAMPUS_FLASH_MODEL)
    results = {}
    for name, (task, features, labels, _) in tasks.items():
        directory = root / name / 'feedback'
        client = ExperimentLLM(CAMPUS_CHAT_ENDPOINT, args.api_key_file, CAMPUS_FLASH_MODEL,
                               ALLOWANCE_RMB, max_requests=RUN_OPTIONS['max_trials'] - 1,
                               allowed_served_models={CAMPUS_FLASH_MODEL, price.official_model_version},
                               account=TaskBudgetAccount(ledger, name, str(directory)), streaming=True)
        feedback = freeze_run(task, 'scalar_feedback', directory, client)
        results[name] = score_pair(root / name, features, labels, dict(fixed=controls[name], feedback=feedback))
        save(root / 'paired_results.json', results)
        print(json.dumps(dict(task=name, **results[name])), flush=True)


if __name__ == '__main__':
    main()
