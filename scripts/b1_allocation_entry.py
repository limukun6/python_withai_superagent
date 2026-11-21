"""Experiment-only process entry. Bind one B policy, then execute A's unmodified command."""
from pathlib import Path
import sys

from scripts import run_b1_ab_acceptance as fixture
from superagent import ab_run
from superagent.cli import build_parser
from superagent.scheduler_bridge import fingerprint


def main(argv):
    if not argv or argv[0] not in ('fixed', 'adaptive'):
        raise ValueError('Explicitly select fixed or adaptive allocation')
    policy = argv[0]
    args = build_parser().parse_args(argv[1:])
    if args.scheduler != 'ab-v1' or not args.run_dir:
        raise ValueError('Expected AB scheduler and an explicit output directory')
    original = ab_run.next_action
    records = []

    def choose(state, **local_evidence):
        action = original(state, allocation=policy, **local_evidence)
        records.append(dict(trial=len(records) + 1, state_sha256=fingerprint(state), action_sha256=fingerprint(action)))
        return action

    # This A revision imports its scheduler once. Only its policy binding is scoped to
    # this standalone process. A still checks all actions and owns every worker and ledger.
    ab_run.next_action = choose
    try:
        return ab_run.command(args)
    finally:
        ab_run.next_action = original
        output = Path(args.run_dir)
        if output.is_dir() and not (output / 'research_policy.json').exists() and records:
            fixture.write(output / 'research_policy.json', dict(policy=policy, decisions=records))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
