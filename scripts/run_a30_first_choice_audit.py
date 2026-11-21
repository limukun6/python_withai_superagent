"""Audit already frozen first Flash choices on two fresh train-only partitions; zero API."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import joblib

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import physical
from superagent.ab_run import sources
from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, take, overlap_audit
from superagent.task_semantics import declared_semantics


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('R0', 'E0', 'R1', 'E1')
SEEDS = (17171, 17181)


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    frozen = sources()
    observer = ROOT / 'scripts' / 'a29_worker_observer' / 'sitecustomize.py'
    observer_hash = sha(observer)
    save(root / 'protocol.json', dict(batch='A30-first-choice-audit', source_files=frozen,
        runner_sha256=sha(Path(__file__)), observer_sha256=observer_hash,
        plan_sha256=sha(ROOT / 'docs/batches/A30_first_choice_audit_plan.md'), execute_requested=args.execute,
        maximum_classifier_fits=16, maximum_api_requests=0, seeds=SEEDS))
    results = []
    try:
        if not (args.matrix / 'completion.json').is_file() or read(args.matrix / 'completion.json')['status'] != 'completed':
            raise ValueError('A29 must complete before starting A30')
        parent = read(args.matrix / 'protocol.json')
        if parent['source_files'] != frozen or read(args.prepared / 'protocol.json')['source_files'] != frozen:
            raise ValueError('Source differs from frozen proposals/prepared training data')
        entries = {entry['task']: entry for entry in read(args.prepared / 'readiness.json')}
        first_choices = {}
        unique = []
        payloads = {}
        for taskname in ('coupon', 'preference'):
            entry = entries[taskname]
            caps = entry['capabilities']
            common = {}
            choices = {}
            full_context = {}
            for arm in ARMS:
                directory = args.matrix / (taskname + '_' + arm)
                records = events(directory / 'events.jsonl')
                request = next(row for row in records if row.get('kind') == 'llm_request')
                proposal_event = next(row for row in records if row.get('kind') == 'ab_flash_proposal')
                proposal = proposal_event['proposal']
                context = json.loads(request['messages'][1]['content'])
                if len(context['history']) != 1 or context['history'][0]['split_id'] != 'primary':
                    raise ValueError('First request does not share the registered one-trial primary prefix')
                if ('checks' in context) != arm.startswith('E') or ('memory' in context) != arm.startswith('E'):
                    raise ValueError('Feedback visibility differs from the recorded arm')
                common[arm] = {key: value for key, value in context.items() if key not in ('checks', 'memory')}
                full_context[arm] = fingerprint(context)
                name = proposal['candidate']
                if name not in caps or proposal['params'] != caps[name]['params']:
                    raise ValueError('First choice differs from trusted registered configuration')
                actual = next(row for row in read(directory / 'history.json') if row['trial'] == proposal_event['trial'])
                if actual['candidate'] != name or actual['params'] != caps[name]['params'] or actual['split_id'] != 'primary':
                    raise ValueError('Proposed first choice was not executed as recorded')
                choices[arm] = dict(candidate=name, params=caps[name]['params'], actual_trial=actual['trial'],
                    original_primary_score=actual['search_score'], request_context_sha256=full_context[arm],
                    request_budget_id=request['budget_request_id'], role=request['role'])
                if (taskname, name) not in unique:
                    unique.append((taskname, name))
            if any(context != common['R0'] for context in common.values()):
                raise ValueError('Public first-request fields are not identical; no fixed-history claim allowed: ' + taskname)
            first_choices[taskname] = dict(common_context_sha256=fingerprint(common['R0']), choices=choices,
                raw_two_inputs_identical=full_context['R0'] == full_context['R1'],
                checked_two_inputs_identical=full_context['E0'] == full_context['E1'],
                raw_choice_consistent=choices['R0']['candidate'] == choices['R1']['candidate'],
                checked_choice_consistent=choices['E0']['candidate'] == choices['E1']['candidate'])
            train_archive = Path(entry['training_archive'])
            if sha(train_archive) != entry['training_archive_sha256']:
                raise ValueError('Prepared train-only input changed')
            task = load_training_task(train_archive)
            metric = MetricSpec.for_task(task)
            folder = root / taskname
            folder.mkdir()
            old_held = {frozenset(read(args.prepared / taskname / (split + '_split.json'))['search_indices'])
                        for split in entry['splits']}
            for seed in SEEDS:
                train, held, resource = grouped_partition(task, seed, 'shared_text')
                if frozenset(held.tolist()) in old_held:
                    raise ValueError('Fresh audit partition repeats a previously exposed split')
                old_held.add(frozenset(held.tolist()))
                path = folder / f'partition_{seed}.joblib'
                joblib.dump(dict(kind=task.kind, X_train=take(task.X_train, train), y_train=task.y_train[train],
                    X_search=take(task.X_train, held), y_search=task.y_train[held], metric_spec=metric.record(),
                    pair_semantics=declared_semantics(task)), path)
                payloads[(taskname, seed)] = path
                save(folder / f'partition_{seed}.json', dict(seed=seed, train_indices=train.tolist(), search_indices=held.tolist(),
                    row_ids_sha256=fingerprint(held.tolist()), payload_sha256=sha(path), split_resource=resource,
                    overlap=overlap_audit(task, train, held)))
        if len(unique) > 8:
            raise ValueError('Too many unique first actions')
        save(root / 'first_choices_frozen.json', first_choices)
        save(root / 'work_order.json', [dict(task=t, candidate=n, seed=s) for t, n in unique for s in SEEDS])
        if not args.execute:
            save(root / 'completion.json', dict(status='validated_not_executed', unique_states=2,
                unique_actions=len(unique), maximum_classifier_fits=2 * len(unique), actual_classifier_fits=0, real_api_requests=0))
            print(json.dumps(dict(status='validated_not_executed', unique_actions=len(unique))))
            return
        os.environ.update(AB_A29_OBSERVE_FITS='1', AB_A29_FIT_LOG=str(root / 'physical_fits.jsonl'),
            PYTHONPATH=os.pathsep.join([str(observer.parent), str(ROOT)]), SUPERAGENT_OFFLINE='1',
            OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
        def reject(event, values):
            if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
                raise PermissionError('Network disabled for the A30 offline audit')
        sys.addaudithook(reject)
        for index, (taskname, name) in enumerate(unique):
            for seed in SEEDS:
                if sources() != frozen or sha(observer) != observer_hash:
                    raise ValueError('Frozen code changed during audit')
                prefix = root / taskname / f'action_{index}_{seed}'
                ledger = prefix.with_suffix('.fits.json')
                admission = prefix.with_suffix('.admission.json')
                save(admission, dict(maximum_classifier_fits=1, ledger_path=str(ledger)))
                os.environ['AB_A29_RUN_NAME'] = f'{taskname}_{index}_{seed}'
                artifact, facts = train_candidate(payloads[(taskname, seed)], name, prefix.with_suffix('.joblib'),
                    60, fit_admission_path=admission)
                count = read(ledger) if ledger.exists() else None
                row = dict(task=taskname, candidate=name, params=entries[taskname]['capabilities'][name]['params'],
                    seed=seed, facts=facts, classifier_fits=count)
                results.append(row)
                save(root / 'results.json', results)
                if artifact is None or count is None or count['status'] != 'completed' or count['completed_classifier_fits'] != 1:
                    raise RuntimeError('Action audit worker failed without retry: ' + name)
                print(json.dumps(dict(task=taskname, candidate=name, seed=seed, score=facts['search_score'])), flush=True)
        mapped = {}
        for taskname, state in first_choices.items():
            mapped[taskname] = {}
            for arm, choice in state['choices'].items():
                values = [next(row['facts']['search_score'] for row in results
                    if row['task'] == taskname and row['candidate'] == choice['candidate'] and row['seed'] == seed) for seed in SEEDS]
                mapped[taskname][arm] = dict(candidate=choice['candidate'], audit_scores=values, mean_score=sum(values) / 2,
                    values_reused_from_unique_action=True)
            for checked, raw in (('E0','R0'), ('E1','R1')):
                mapped[taskname][checked]['difference_from_' + raw] = mapped[taskname][checked]['mean_score'] - mapped[taskname][raw]['mean_score']
        fits = len(physical(root))
        if fits != len(results) or fits > 16:
            raise ValueError('Independent physical fit count differs from audit actions')
        save(root / 'comparison.json', mapped)
        save(root / 'completion.json', dict(status='completed', source_unchanged=sources() == frozen,
            actual_classifier_fits=fits, logical_action_partition_evaluations=16, unique_actions=len(unique),
            reused_action_partition_evaluations=16-fits, unique_states=2, real_api_requests=0,
            external_validation_opened=False, elapsed_seconds=time.monotonic() - start))
    except Exception as error:
        save(root / 'completion.json', dict(status='stopped', error=str(error), automatic_retry=False,
            observed_fit_entries=len(physical(root)), real_api_requests=0, elapsed_seconds=time.monotonic() - start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args())
