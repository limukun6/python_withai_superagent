"""Compare fixed and adaptive training allocations on previously registered headline holdouts."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import time

from scripts import b1_allocation_entry, run_b1_ab_acceptance as fixture, run_b1_05_pair_pilot as pilot
from scripts.check_b_pair_protocol import compare
from scripts.run_b1_06_mrpc import paired_counts
from superagent.ab_run import sources
from superagent.ab_sentence import verified_directory
from superagent.scheduler_bridge import fingerprint


ARMS = ('fixed', 'adaptive')
ORDER = ((45102, 'fixed'), (45102, 'adaptive'), (45103, 'adaptive'), (45103, 'fixed'))


def setup(root, released, frozen):
    entries = {}
    data = pilot.pair_source('headlines', released)
    for seed in (45102, 45103):
        shared = pilot.prepare_pairs(root, str(seed), *data, outer_seed=seed)
        for arm in ARMS:
            name = f'{seed}_{arm}'
            shutil.copytree(root / str(seed), root / name)
            entries[name] = dict(shared, name=name, outer_seed=seed, allocation=arm)
    fixture.write(root / 'protocol.json', dict(batch='B1-07', source_sha256=frozen,
        driver_sha256=fixture.sha(Path(__file__).read_bytes()), entry_sha256=fixture.sha(Path(b1_allocation_entry.__file__).read_bytes()),
        helper_sha256=fixture.sha(Path(pilot.__file__).read_bytes()), environment=pilot.environment(), execution_order=ORDER,
        plan_sha256=fixture.sha((pilot.ROOT / 'docs/batches/B1_07_plan.md').read_bytes()), tasks=entries,
        fit_limit_per_path=11, fit_limit_batch=44, inner_seed=pilot.INNER_SEED, wall_seconds=pilot.WALL,
        trial_seconds=pilot.TRIAL, api_requests_allowed=0, selected_dataset_members=['train.csv']))
    return entries


def verify_policy(path, arm, actions):
    run = path / 'run'
    sidecar = fixture.read(run / 'research_policy.json')
    if sidecar['policy'] != arm or len(sidecar['decisions']) != len(actions):
        raise ValueError('Explicit experimental policy differs from the requested arm')
    for i, (record, action) in enumerate(zip(sidecar['decisions'], actions, strict=True), 1):
        state = fixture.read(run / f'state_{i}.json')
        if record != dict(trial=i, state_sha256=fingerprint(state), action_sha256=fingerprint(action)):
            raise ValueError('Policy sidecar differs from A decision input or executed action')
    if arm == 'fixed' and [a['action'] for a in actions] != ['explore'] * 4 + ['confirm'] * 6 + ['stop']:
        raise ValueError('Full-resource fixed path did not match its registered four/three/two schedule')


def run(root, released, weights, reference, prepare_only):
    root.mkdir(parents=True, exist_ok=False)
    fixture.offline(root)
    os.environ.update(SUPERAGENT_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      SUPERAGENT_MINILM_WEIGHTS=str(weights))
    started = time.monotonic()
    paths = []
    try:
        if verified_directory() is None:
            raise ValueError('Shared pinned encoder failed local verification')
        frozen = fixture.digest(sources())
        entries = setup(root, released, frozen)
        if reference:
            checks = compare(reference, root)
            fixture.write(root / 'preflight.json', checks)
            if not checks['comparable']:
                raise ValueError('External protocol mismatch before training: ' + ', '.join(checks['differences']))
        if prepare_only:
            fixture.write(root / 'completion.json', dict(status='prepared', fits=0, api_requests=0))
            return
        for seed, arm in ORDER:
            name = f'{seed}_{arm}'
            before = len(fixture.trace(root))
            path = pilot.train_one(root / name, entries[name], frozen,
                                   launcher=('-m', 'scripts.b1_allocation_entry', arm, 'run'))
            verify_policy(root / name, arm, path['actions'])
            if len(fixture.trace(root)) - before != path['fits']:
                raise ValueError('Independent fit observation disagrees for ' + name)
            paths.append(path)
            fixture.write(root / 'results.json', dict(paths=paths))
        fixture.write(root / 'all_models_frozen.json', {p['name']: p['model_sha256'] for p in paths})
        for p in paths:
            p['delivery'] = pilot.score_one(root / p['name'], entries[p['name']])
        comparisons = {str(seed): paired_counts(root, seed, ARMS) for seed in (45102, 45103)}
        holds = [set(json.loads(gzip.decompress((root / str(seed) / 'row_ids.json.gz').read_bytes()))['outer']['held'])
                 for seed in (45102, 45103)]
        if fixture.digest(sources()) != frozen:
            raise ValueError('Production code changed during the batch')
        fixture.write(root / 'results.json', dict(paths=paths, paired_counts=comparisons, outer_holdout_shared_rows=len(holds[0] & holds[1])))
        fixture.write(root / 'completion.json', dict(status='passed', paths=len(paths), fits=len(fixture.trace(root)),
            api_requests=0, api_rmb=0, models_frozen_before_scoring=True, course_validation_and_test_opened=False,
            h1_accessed=False, seconds=time.monotonic() - started))
    except Exception as error:
        fixture.write(root / 'results.json', dict(paths=paths))
        fixture.write(root / 'completion.json', dict(status='failed', error=str(error), fits_observed=len(fixture.trace(root)),
                      api_requests=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path, help='Require an exact match to this prepared protocol before any fit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.output.resolve(), args.released.resolve(), args.weights.resolve(), args.reference, args.prepare_only)
