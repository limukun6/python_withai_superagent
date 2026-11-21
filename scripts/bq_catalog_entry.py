"""Run the experiment's explicit original/expanded executable catalogue through A's command."""
from pathlib import Path
import sys

from scripts import run_b1_ab_acceptance as fixture
from superagent import ab_run
from superagent.ab_catalog import capabilities_for
from superagent.cli import build_parser


def registered_catalog(task, mode, **constraints):
    if mode not in ('original', 'expanded'):
        raise ValueError('Unknown catalogue condition')
    full = capabilities_for(task, **constraints)
    if mode == 'original':
        return {name: cap for name, cap in full.items() if cap['origin'] == 'shared_baseline'}
    return full


def main(argv):
    if not argv or argv[0] not in ('original', 'expanded'):
        raise ValueError('Select original or expanded catalogue')
    mode = argv[0]
    args = build_parser().parse_args(argv[1:])
    if args.scheduler != 'ab-v1' or not args.run_dir:
        raise ValueError('Expected AB scheduler and an explicit run directory')
    original = ab_run.capabilities_for
    receipt = {}

    def register(task, **constraints):
        if receipt:
            raise ValueError('Q requires a single catalogue fixed at task start')
        caps = registered_catalog(task, mode, **constraints)
        receipt.update(mode=mode, ids=list(caps), sha256=fixture.digest(caps))
        return caps

    ab_run.capabilities_for = register
    try:
        return ab_run.command(args)
    finally:
        ab_run.capabilities_for = original
        output = Path(args.run_dir) / 'research_catalog.json'
        if receipt and output.parent.is_dir() and not output.exists():
            fixture.write(output, receipt)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
