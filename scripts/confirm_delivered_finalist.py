"""Measure the frozen choice on one previously unused internal split, outside the official solve."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN = ROOT / 'research/task2_finalist_confirmation_1127/protocol.json'


def confirm(source, output, weights, course):
    plan = read(PLAN)
    agent = source / 'titles/run/agent'
    selected = read(agent / 'final_selection.json')
    baseline = selected['single_candidate']
    require(source_identity(ROOT) == plan['runtime_sha256'] and selected['refit']
            and selected['model_sha256'] == sha(agent / selected['model_file']) == plan['original_selection_file_sha256_of_model'],
            'Runtime or delivered model differs from registration')
    history = read(agent / 'history.json')
    prior = [r for r in history if r['candidate'] == baseline and r['status'] == 'completed']
    require(len(prior) == 1 and prior[0]['split_id'] == 'primary'
            and set(selected['fold_scores']) == {'primary'}, 'Selected method has already been confirmed')
    previous = prior[0]
    decision = next(d for d in read(agent / 'decisions.json') if d['step'] == previous['trial'])
    registry = agent / decision['registry']
    require(sha(registry) == previous['resolved_configuration']['registry_sha256'] == decision['registry_sha256'],
            'Candidate registry differs from the winning trial')
    new_fold = read(agent / 'confirmation_1_split.json')
    primary = read(agent / 'primary_split.json')
    payload = agent / 'confirmation_1.joblib'
    require(sha(payload) == new_fold['payload_sha256'] == plan['confirmation_payload_sha256'], 'Changed fold payload')
    data = joblib.load(payload)
    fit_ids, held_ids = new_fold['fit'], new_fold['held']
    y = np.asarray(data['y_search'])
    require(data['kind'] == 'pair_text' and fingerprint(y) == new_fold['label_sha256']
            and fingerprint(np.asarray(held_ids)) == new_fold['row_sha256'], 'Changed evaluation rows or labels')
    size = selected['training_rows']
    require(sorted(fit_ids + held_ids) == list(range(size)) and len(fit_ids) == len(data['X_train'])
            and len(held_ids) == len(data['X_search']) == len(y), 'Incomplete internal partition')
    pairs = [None] * size
    for indexes, rows in ((fit_ids, data['X_train']), (held_ids, data['X_search'])):
        for index, row in zip(indexes, rows):
            pairs[index] = row
    groups = sample_groups(SimpleNamespace(kind=data['kind'], X_train=pairs), 'shared_text')
    new_groups = set(groups[held_ids])
    require(set(primary['held']).isdisjoint(held_ids) and new_groups.isdisjoint(groups[primary['held']])
            and new_groups.isdisjoint(groups[fit_ids]), 'Shared rows or linked headline groups')
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / 'worker.fits.json'
    admission = output / 'admission.json'
    save(admission, dict(maximum_classifier_fits=1, ledger_path=str(ledger)))
    physical = output / 'observed_fits.jsonl'
    settings = env(weights, physical)
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / 'scripts/joint_feedback_observer', ROOT))),
                    REFLECTION_STUDY_LIVE='0', JOINT_OUTER_ANSWERS=str(source / 'validation.json'),
                    JOINT_UPSTREAM_SOURCES=str(course), JOINT_PREDICTION_INPUT=str(source / 'titles/features.zip'))
    settings[PREFERENCE_CATALOG_ENV] = 'all-installed'
    artifact = output / 'diagnostic.joblib'
    with (output / 'fit.log').open('x', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'superagent.experiment_worker', str(payload), baseline,
                        str(artifact), str(registry), '--fit-admission', str(admission)],
                       env=settings, cwd=ROOT, check=True, stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, timeout=plan['limits']['fit_seconds'])
    fitted = joblib.load(artifact)
    metrics = fitted['metrics']
    predictions = np.asarray(fitted['search_predictions'])
    correct = int(np.count_nonzero(predictions == y))
    measured = read(ledger)
    require(measured['attempted_classifier_fits'] == measured['completed_classifier_fits'] == 1
            and len(physical.read_text().splitlines()) == 1 and metrics['resolved_configuration'] == previous['resolved_configuration']
            and predictions.shape == y.shape and metrics['primary_metric'] == selected['primary_metric'] == 'accuracy'
            and metrics['search_score'] == correct / len(y), 'Worker identity, fit count or metric mismatch')
    result = dict(runtime_sha256=plan['runtime_sha256'], protocol_sha256=sha(PLAN), delivered_model_sha256=selected['model_sha256'],
                  candidate=baseline, registry_sha256=sha(registry), primary_rows=len(primary['held']),
                  primary_score=previous['search_score'], confirmation_fit_rows=len(fit_ids), confirmation_rows=len(y),
                  confirmation_correct=correct, confirmation_score=metrics['search_score'], confirmation_groups=len(new_groups),
                  primary_to_confirmation_shared_rows=0, primary_to_confirmation_shared_groups=0,
                  confirmation_fit_to_evaluation_shared_groups=0, additional_fits=1, new_model_requests=0,
                  new_course_model_cost_rmb=0, course_holdout_used=False, delivered_model_updated=False)
    save(output / 'results.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'weights', 'course'):
        p.add_argument('--' + name, required=True, type=lambda s: Path(s).resolve())
    args = p.parse_args()
    confirm(args.source, args.output, args.weights, args.course)
