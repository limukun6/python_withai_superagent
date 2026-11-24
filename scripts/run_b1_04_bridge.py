"""Run the local B adapter through A's dispatch, worker, refit and frozen predictor."""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import joblib

from scripts import run_b1_ab_acceptance as fixture
from superagent.ab_catalog import ADAPTER_VERSION, capabilities_for, untried_candidates
from superagent.ab_scheduler import CONFIRMATIONS, ROW_SCOPE, next_action, selection_evidence
from superagent.contracts import TaskSpec
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.scheduler_bridge import dispatch_action, fingerprint as state_fingerprint


A_BASE = '67d9874ebb8fa2411ab142c94570078860c8cc34'
B_BASE = '781e8c5efed8b263a3972643d7fa067fce6d77d1'


def splits_for(partitions):
    return {split: dict(id=split, seed=fixture.SEEDS[split], row_ids_sha256=fingerprint(part['row_ids']),
                        label_sha256=fingerprint(part['labels']),
                        evaluation_payload_sha256=part['evaluation_payload_sha256'])
            for split, part in partitions.items()}


def execute_trial(plan, directory, number, metric, capabilities, began):
    if plan['action']['action'] == 'stop':
        return dict(status='completed', classifier_fits=0)
    candidate = plan['candidate']
    name = candidate['id']
    split = plan['split']
    payload = directory / (split['id'] + '.joblib')
    if fixture.sha(payload.read_bytes()) != split['evaluation_payload_sha256']:
        raise ValueError('Registered partition has changed')
    timeout = min(fixture.TRIAL_SECONDS, fixture.WALL_SECONDS - (time.monotonic() - began))
    if timeout <= 0:
        raise TimeoutError('Shared wall-clock budget exhausted')
    os.environ['AB_B1_PER_WORKER_FIT_LIMIT'] = str(plan['expected_classifier_fits'])
    model, facts = fixture.checked_worker(directory, payload, name, directory / f'trial_{number}.joblib', timeout)
    config = facts['resolved_configuration']
    cap = capabilities[name]
    if config['id'] != name or config['params'] != candidate['params'] or candidate['params'] != cap['params']:
        raise ValueError('Resolved worker configuration differs from the admitted action')
    if (facts['estimator']['family'] != cap['estimator']
            or hasattr(model, 'predict_proba') != cap['supports_probabilities']):
        raise ValueError('Numeric worker estimator or probability support differs from the registered capability')
    if metric.check_identity(facts['metric_spec'])['status'] != 'confirmed':
        raise ValueError('Worker did not record metric identity')
    return {key: facts[key] for key in ('status', 'classifier_fits', 'search_score', 'search_accuracy',
                                        'primary_metric', 'metric_spec', 'estimator', 'resolved_configuration')}


def row_evidence(directory, trial, name, split, partitions):
    raw = joblib.load(directory / f'trial_{trial}.joblib')['prediction_record']
    # A accepts both wire spellings. Exercise each on observations of real fits.
    identity_key = 'candidate' if trial % 2 else 'candidate_id'
    return {identity_key: name, 'trial': trial, 'split_id': split, 'row_id_scope': ROW_SCOPE,
            'row_ids': partitions[split]['row_ids'], 'predictions': raw['predictions'].tolist(),
            **{k: raw[k] for k in ('prediction_sha256', 'label_sha256', 'evaluation_payload_sha256')}}


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    fixture.offline(directory)
    began = time.monotonic()
    actions, history, evidence = [], [], []
    results = {'actions': actions, 'observations': history}
    try:
        task, query = fixture.packages(directory)
        spec = TaskSpec.from_task(task).record()
        metric = MetricSpec.for_task(task)
        caps = capabilities_for(spec, train_rows=len(task.y_train))
        partitions = fixture.freeze_partitions(task, directory)
        splits = splits_for(partitions)
        budget = dict(fit_limit=11, completed_classifier_fits=0, attempted_classifier_fits=0,
                      remaining_fit_allowance=11, reserved_final_refit=1, unknown_failed_attempts=0,
                      llm_request_limit=0, used_llm_requests=0, time_limit_seconds=fixture.WALL_SECONDS)
        plan = fixture.ROOT / 'docs/batches/B1_04_plan.md'
        fixture.write(directory / 'protocol.json', dict(batch='B1-04', adapter_version=ADAPTER_VERSION,
            a_base=A_BASE, b_base=B_BASE, source_sha256=fixture.digest(dict(modules=fixture.sources(),
            driver=fixture.sha(Path(__file__).read_bytes()), plan=fixture.sha(plan.read_bytes()))),
            scope='one anonymous synthetic numeric training task, local decisions',
            data_seed=932, training_rows=len(task.y_train), unlabeled_query_seed=975, splits=splits,
            candidate_ids=list(caps), capabilities_sha256=fixture.digest(caps), initial_budget=budget.copy(),
            trial_time_limit_seconds=fixture.TRIAL_SECONDS))
        logger = RunLogger(directory)
        if not caps:
            raise ValueError('No supported executable candidate at task start')
        while True:
            budget['elapsed_seconds'] = time.monotonic() - began
            menu = untried_candidates(caps, history)
            state = dict(ab_schema_version=1, task=spec, history=history, available_candidates=menu,
                         prediction_evidence=evidence, budget=budget, registered_confirmation_splits=list(CONFIRMATIONS))
            action = next_action(state, capabilities=caps, trusted_partitions=partitions)
            trial = len(history) + 1
            outcome = dispatch_action(state, action, caps, splits,
                lambda plan: execute_trial(plan, directory, trial, metric, caps, began), logger)
            if outcome['state_sha256'] != state_fingerprint(state):
                raise ValueError('Dispatcher fingerprint differs from the state given to B')
            before = budget['attempted_classifier_fits']
            budget = outcome['budget']
            observed = outcome['observation']
            if budget['attempted_classifier_fits'] != len(fixture.trace(directory)):
                raise ValueError('Dispatcher ledger differs from physical fit calls')
            visible = [entry['id'] for entry in menu]
            actions.append(dict(**action, visible_candidate_ids=visible, state_sha256=outcome['state_sha256'],
                                fits_before=before, fits_after=budget['attempted_classifier_fits']))
            if action['action'] == 'stop':
                break
            name, split = action['candidate_id'], action['split_id']
            if (name in visible) != (action['action'] == 'explore'):
                raise ValueError('Confirmation leaked into the untried menu or exploration was absent')
            history.append(dict(trial=trial, candidate=name, params=observed['resolved_configuration']['params'],
                                split_id=split, **observed))
            evidence.append(row_evidence(directory, trial, name, split, partitions))

        counts = Counter(action['action'] for action in actions)
        if not (counts['explore'] and counts['confirm'] and counts['stop'] == 1):
            raise ValueError('The three actions were not all exercised on the anonymous task')
        review = selection_evidence(state, trusted_partitions=partitions)
        refit = caps[review['incumbent']]['refit_classifier_fits']
        if refit > min(budget['remaining_fit_allowance'], budget['reserved_final_refit']):
            raise ValueError('Insufficient budget for final refit')
        os.environ['AB_B1_PER_WORKER_FIT_LIMIT'] = str(refit)
        final = fixture.finish(directory, task, history, began, review)
        trace = fixture.trace(directory)
        if final['full_refits'] != refit or trace[-1]['candidate'] != final['family']:
            raise ValueError('Physical refit does not match the selected capability')
        prediction = fixture.predict_twice(directory, query)
        results.update(action_counts=dict(counts), physical_fit_calls=trace, final=final,
                       search_budget_at_stop=budget, fresh_process_prediction=prediction,
                       elapsed_seconds=round(time.monotonic() - began, 3))
        fixture.write(directory / 'results.json', results)
        fixture.write(directory / 'completion.json', dict(status='passed', observed_fit_calls=len(trace),
            action_counts=dict(counts), refits=final['full_refits'], prediction_processes=2,
            matching_unlabeled_rows=prediction['rows_per_prediction'], course_data_accessed=False,
            execution='A dispatch with real workers in the B acceptance driver', official_run_cli_exercised=False,
            api_requests_this_run=0, api_rmb_this_run=0))
    except Exception as error:
        fixture.write(directory / 'results.json', results)
        fixture.write(directory / 'completion.json', dict(status='failed', error=str(error),
                      observed_fit_calls=len(fixture.trace(directory)), api_requests_this_run=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory.resolve())
