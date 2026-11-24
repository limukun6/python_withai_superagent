"""A151 conditional small trial: one real closed loop per arm, capped at 3 fits and 3 Flash calls.

Default is `--validate-only`: it resolves the official entry, checks the ledger,
the accepted-history snapshot, the task package and both commands without
fitting anything. Only `--execute` spends money.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.a_batch_common import save
from scripts.campus_direct import CAMPUS_IP, campus_resolution
from superagent.ab_run import sources
from superagent.cli import build_parser
from tools.accepted_usage_snapshot import accepted_history

BATCH = 'a151_small_trial'
EVIDENCE = REPO / 'research' / BATCH
MAX_FITS = 3
MAX_REQUESTS = 3
ALLOWANCE = .15
ARMS = {
    'L0': {'task_budget_id': 'a151_loop_off',
           'extra': ['--feedback-view', 'compact', '--proposal-contract', 'calibrated-v2']},
    'L1': {'task_budget_id': 'a151_loop_by_class',
           'extra': ['--feedback-view', 'compact', '--proposal-contract', 'grounded-v1',
                     '--issue-layering', 'by-class']},
}


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command_for(arm, task, run_dir, args):
    return ['run', str(task), '--run-dir', str(run_dir),
            '--planner', 'checked-history', '--scheduler', 'ab-v1', '--portfolio', 'ab-v1',
            '--catalog-scope', 'full', '--preference-catalog', 'standard', '--parameter-semantics', 'off',
            '--fit-limit', str(MAX_FITS), '--seed', '707', '--trial-timeout', '120', '--time-limit', '600',
            '--partial-delivery', 'checkpoint', '--history-policy', 'incremental-research',
            '--api-key-file', str(args.api_key_file), '--budget-ledger', str(args.ledger),
            '--task-budget-id', ARMS[arm]['task_budget_id'], '--llm-budget-rmb', str(ALLOWANCE),
            *ARMS[arm]['extra']]


def check_outcome(outcome, calls, delta):
    fits = outcome['classifier_fit_accounting']
    attempted, completed = fits['attempted_classifier_fits'], fits['completed_classifier_fits']
    problems = []
    if not outcome.get('requested_planner_completed'):
        problems.append('planner did not complete within the requested budget')
    if not 2 <= attempted <= MAX_FITS:
        problems.append(f'attempted fits {attempted} outside 2..{MAX_FITS}')
    if attempted != completed or fits.get('unknown_failed_attempts'):
        problems.append('fit accounting is incomplete')
    if calls > MAX_REQUESTS:
        problems.append(f'{calls} API calls exceed {MAX_REQUESTS}')
    if calls != delta:
        problems.append(f'ledger delta {delta} differs from reported calls {calls}')
    if not outcome.get('full_refit_completed') and outcome.get('refit_requested'):
        problems.append('final refit did not complete')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=EVIDENCE)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True)
    began = time.monotonic()
    frozen = sources()
    from superagent.task_budget import TaskBudgetLedger
    ledger = TaskBudgetLedger(args.ledger)
    before = ledger.project_summary()
    auth_sha = sha(args.authorization)
    protocol = {'batch': 'A151-small-trial', 'source_files': frozen, 'runner_sha256': sha(Path(__file__)),
                'plan_sha256': sha(REPO / 'docs/batches/A151_small_trial_plan_20261124_zh.md'),
                'task': str(args.task.resolve()), 'task_sha256': sha(args.task),
                'task_kind': 'pair_text', 'seed': 707, 'scheduler': 'ab-v1', 'portfolio': 'ab-v1',
                'maximum_fits_per_arm': MAX_FITS, 'maximum_requests_per_arm': MAX_REQUESTS,
                'allowance_rmb_per_arm': ALLOWANCE, 'arms': ARMS,
                'ledger_path': str(args.ledger.resolve()), 'ledger_sha256_before': sha(args.ledger),
                'project_before': before, 'accepted_history_sha256': auth_sha,
                'execute_requested': args.execute, 'automatic_retries': False,
                'not_claiming': ['score gain', 'generality on unseen tasks', 'training saving without a matched pairing']}
    save(out / 'protocol.json', protocol)
    commands = {}
    for arm in ARMS:
        commands[arm] = command_for(arm, args.task.resolve(), out / arm / 'run', args)
        parsed = build_parser().parse_args(commands[arm])
        if parsed.planner != 'checked-history' or parsed.fit_limit != MAX_FITS:
            raise SystemExit(f'{arm}: unexpected parsed budget')
        save(out / f'command_{arm}.json', commands[arm])
    if not args.execute:
        save(out / 'completion.json', {'status': 'validated_not_executed', 'actual_classifier_fits': 0,
                                       'real_API_requests': 0, 'course_cost_rmb': 0,
                                       'elapsed_seconds': round(time.monotonic() - began, 3)})
        print(json.dumps({'status': 'validated_not_executed', 'commands': len(commands)}))
        return 0
    results = []
    try:
        for arm, command in commands.items():
            if sources() != frozen:
                raise RuntimeError('Source changed before the paid arm')
            spent_before = ledger.project_summary()['requests']
            parsed = build_parser().parse_args(command)
            with accepted_history(args.authorization, auth_sha), campus_resolution(CAMPUS_IP):
                parsed.handler(parsed)
            outcome = read(out / arm / 'run' / 'run_outcome.json')
            calls = ledger.project_summary()['requests'] - spent_before
            reported = outcome.get('new_api_requests')
            problems = check_outcome(outcome, calls, reported)
            prediction = out / f'predictions_{arm}.csv'
            predict = subprocess.run([sys.executable, '-m', 'superagent', 'predict', str(args.task.resolve()),
                                      '--run-dir', str(out / arm / 'run'), '--output', str(prediction)],
                                     cwd=REPO, capture_output=True, text=True)
            rows = len(prediction.read_text(encoding='utf-8').splitlines()) - 1 if prediction.is_file() else None
            results.append({'arm': arm, 'task_budget_id': ARMS[arm]['task_budget_id'],
                            'candidate': outcome.get('selected_candidate'),
                            'fits': outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                            'api_calls': calls, 'problems': problems,
                            'prediction_rows': rows,
                            'prediction_ok': predict.returncode == 0 and (rows or 0) > 0,
                            'prediction_error': None if predict.returncode == 0 else predict.stderr[-800:]})
            if problems:
                raise RuntimeError(f'{arm} failed acceptance: ' + '; '.join(problems))
            if sources() != frozen:
                raise RuntimeError('Source changed during the paid arm')
        after = ledger.project_summary()
        completion = {'status': 'complete', 'arms': results,
                      'actual_classifier_fits': sum(r['fits'] for r in results),
                      'real_API_requests': sum(r['api_calls'] for r in results),
                      'course_cost_rmb': (after['known_course_cost_rmb'] or 0) - (before['known_course_cost_rmb'] or 0),
                      'ledger_sha256_after': sha(args.ledger),
                      'new_unknown_requests': after['unresolved_requests'] - before['unresolved_requests'],
                      'pending_requests': after['pending_requests'],
                      'elapsed_seconds': round(time.monotonic() - began, 3), 'automatic_retries': False}
        save(out / 'completion.json', completion)
        save(out / 'current_ledger_owner.json', dict(owner='A', window_status='exclusive_idle',
             ledger_path=str(args.ledger.resolve()), ledger_sha256=sha(args.ledger), project_summary=after,
             active_paid_queue=False, accepted_history_sha256=auth_sha,
             existing_unknowns_accepted_by_user=True, new_unknowns_require_stop=True))
        print(json.dumps(completion, ensure_ascii=False, indent=1))
        return 0
    except Exception as error:
        detail = f'{type(error).__name__}: {error}'
        try:
            secret = Path(args.api_key_file).read_text(encoding='utf-8').strip()
            if secret:
                detail = detail.replace(secret, '[redacted]')
        except OSError:
            pass
        save(out / 'completion.json', {'status': 'stopped', 'error': detail, 'arms': results,
                                       'elapsed_seconds': round(time.monotonic() - began, 3),
                                       'automatic_retries': False})
        print(json.dumps({'status': 'stopped', 'error': detail}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
