"""Complete the fixed/adaptive allocation study on the three other real development tasks."""
import argparse
from pathlib import Path
import shutil

from scripts import b1_allocation_entry, run_b1_ab_acceptance as fixture, run_b1_05_pair_pilot as pilot
from scripts import run_b1_07_allocation as allocation, run_bq_01_portfolios as portfolios


TASKS = ('bean', 'sms', 'coupon')


def prepare_tasks(root, args, frozen):
    entries = {}
    for seed in (45102, 45103):
        shared = portfolios.prepare_shared(root, args.task, args, outer_seed=seed)
        group = root / str(seed)
        (root / args.task).rename(group)
        for arm in allocation.ARMS:
            name = f'{seed}_{arm}'
            shutil.copytree(group, root / name)
            entries[name] = dict(shared, name=name, outer_seed=seed, allocation=arm)
    inputs = [Path(portfolios.__file__), pilot.ROOT / 'scripts/build_external_tasks.py']
    fixture.write(root / 'protocol.json', dict(batch='B1-08-' + args.task, source_sha256=frozen,
        environment=pilot.environment(), tasks=entries, execution_order=allocation.ORDER,
        driver_sha256=fixture.sha(Path(__file__).read_bytes()), runner_sha256=fixture.sha(Path(allocation.__file__).read_bytes()),
        helper_sha256=fixture.sha(Path(pilot.__file__).read_bytes()), entry_sha256=fixture.sha(Path(b1_allocation_entry.__file__).read_bytes()),
        source_builder_sha256=fixture.digest({p.name: fixture.sha(p.read_bytes()) for p in inputs}),
        plan_sha256=fixture.sha((pilot.ROOT / 'docs/batches/B1_08_plan.md').read_bytes()),
        fit_limit_per_path=11, fit_limit_batch=44, inner_seed=pilot.INNER_SEED, wall_seconds=pilot.WALL,
        trial_seconds=pilot.TRIAL, api_requests_allowed=0))
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    for name in ('coupon', 'uci', 'weights', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and not args.reference:
        parser.error('A frozen preflight reference is required for training')
    allocation.run(args.output.resolve(), args, args.weights.resolve(), args.reference, args.prepare_only,
                   prepare_tasks=prepare_tasks)
