"""A151 matrix: run the pre-registered W1/W2 paths one at a time.

Default is a parse-and-plan check with no requests. `--execute` runs each path
in order and stops the whole matrix on the first failed acceptance item, a new
unknown ledger row, or a missing required confirmation observation.
"""
import argparse
from contextlib import nullcontext
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.a_batch_common import save
from scripts.campus_direct import CAMPUS_IP, campus_resolution
from superagent.ab_run import sources
from superagent.cli import build_parser
from superagent.task_budget import TaskBudgetLedger
from tools.accepted_usage_snapshot import accepted_history

BATCH = 'a151_matrix'
EVIDENCE = REPO / 'research' / BATCH


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command_for(path, fixed, out, args):
    menu_view = path.get('menu_view', fixed.get('menu_view', 'full'))
    extra = ['--feedback-view', fixed['feedback_view'], '--menu-view', menu_view,
             '--proposal-contract', path['contract']]
    if path['layering'] != 'off':
        extra += ['--issue-layering', path['layering']]
    if path.get('request_limit'):
        extra += ['--llm-request-limit', str(path['request_limit'])]
    if path.get('plan_fragment'):
        extra += ['--plan-fragment', path['plan_fragment']]
    return ['run', path['task'], '--run-dir', str(out / path['name'] / 'run'),
            '--planner', 'checked-history', '--scheduler', path['scheduler'],
            '--portfolio', 'ab-v1', '--catalog-scope', 'full', '--preference-catalog', 'standard',
            '--parameter-semantics', 'off', '--fit-limit', str(path['fit_limit']),
            '--seed', str(path.get('seed', fixed.get('seed', 707))), '--trial-timeout', str(fixed['trial_timeout']),
            '--time-limit', str(fixed['time_limit']), '--partial-delivery', fixed['partial_delivery'],
            '--history-policy', 'incremental-research',
            '--api-key-file', str(args.api_key_file), '--budget-ledger', str(args.ledger),
            '--task-budget-id', path['task_budget_id'], '--llm-budget-rmb', str(fixed['allowance_rmb']),
            *extra]


def check(path, outcome, calls, delta, before_unknown, after_unknown):
    fits = outcome['classifier_fit_accounting']
    attempted, completed = fits['attempted_classifier_fits'], fits['completed_classifier_fits']
    problems = []
    if not outcome.get('requested_planner_completed'):
        problems.append('planner did not complete')
    if attempted != completed:
        problems.append('fit accounting incomplete')
    if attempts_over(attempted, path['fit_limit']):
        problems.append(f'attempted {attempted} exceeds {path["fit_limit"]}')
    if calls > 3:
        problems.append(f'{calls} calls exceed 3')
    if delta != calls:
        problems.append(f'ledger delta {delta} != reported {calls}')
    if after_unknown != before_unknown:
        problems.append(f'new unknown rows: {before_unknown} -> {after_unknown}')
    if not outcome.get('full_refit_completed'):
        problems.append('final refit did not complete')
    confirmed = (outcome.get('confirmation') or {}).get('independent_confirmation_observed')
    if path['require_independent_confirmation'] and not confirmed:
        problems.append('independent confirmation was not observed')
    return problems


def attempts_over(attempted, limit):
    return attempted > limit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=EVIDENCE / 'plan.json')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    fixed = plan['fixed']
    paths = [p for p in plan['paths'] if not args.only or p['name'] in args.only]
    out = (args.output or (EVIDENCE / 'attempt_01')).resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True, exist_ok=True)
    frozen = sources()
    commands = {p['name']: command_for(p, fixed, out, args) for p in paths}
    protocol = {'batch': plan['batch'], 'source_files': frozen,
                'runner_sha256': sha(Path(__file__)),
                'plan_doc_sha256': sha(REPO / plan['plan_doc']),
                'matrix_sha256': sha(args.plan),
                'fixed': fixed, 'paths': paths, 'commands': commands,
                'execute_requested': args.execute, 'automatic_retries': False,
                'maximum_total_requests': 3 * len(paths)}
    save(out / 'protocol.json', protocol)
    for name, command in commands.items():
        build_parser().parse_args(command)
    if not args.execute:
        save(out / 'completion.json', {'status': 'validated_not_executed', 'paths': len(paths),
                                       'real_API_requests': 0, 'classifier_fits': 0,
                                       'course_cost_rmb': 0})
        print(json.dumps({'status': 'validated_not_executed', 'paths': len(paths)}))
        return 0
    if not (args.ledger and args.authorization and args.api_key_file):
        raise SystemExit('--execute needs ledger, authorization and api-key-file')
    ledger = TaskBudgetLedger(args.ledger)
    before = ledger.project_summary()
    auth_sha = sha(args.authorization)
    rows, started = [], time.monotonic()
    try:
        for path in paths:
            if sources() != frozen:
                raise RuntimeError('Source changed before ' + path['name'])
            requests_before = ledger.project_summary()['requests']
            unknown_before = ledger.project_summary()['unresolved_requests']
            parsed = build_parser().parse_args(commands[path['name']])
            with accepted_history(args.authorization, auth_sha), campus_resolution(CAMPUS_IP):
                parsed.handler(parsed)
            outcome = read(out / path['name'] / 'run' / 'run_outcome.json')
            calls = ledger.project_summary()['requests'] - requests_before
            unknown_after = ledger.project_summary()['unresolved_requests']
            problems = check(path, outcome, calls, outcome.get('new_api_requests'), unknown_before, unknown_after)
            prediction = out / path['name'] / 'predictions.csv'
            if path.get('predict', 'cli') == 'reload':
                record_path = out / path['name'] / 'delivery_check.json'
                done = subprocess.run([sys.executable, str(REPO / 'scripts' / 'a151_reload_check.py'),
                                       '--run-dir', str(out / path['name'] / 'run'),
                                       '--package', path['task'],
                                       '--output', str(record_path)],
                                      cwd=REPO, capture_output=True, text=True)
                prediction_rows = json.loads(record_path.read_text(encoding='utf-8'))['rows'] if record_path.is_file() else None
            else:
                done = subprocess.run([sys.executable, '-m', 'superagent', 'predict', path['task'],
                                       '--run-dir', str(out / path['name'] / 'run'), '--output', str(prediction)],
                                      cwd=REPO, capture_output=True, text=True)
                prediction_rows = len(prediction.read_text(encoding='utf-8').splitlines()) - 1 if prediction.is_file() else None
            rows.append({'name': path['name'], 'task': Path(path['task']).name,
                         'scheduler': path['scheduler'], 'contract': path['contract'],
                         'candidate': outcome.get('selected_candidate'),
                         'fits': outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                         'calls': calls,
                         'independent_confirmation_observed': (outcome.get('confirmation') or {}).get('independent_confirmation_observed'),
                         'prediction_rows': prediction_rows,
                         'prediction_channel': path.get('predict', 'cli'),
                         'prediction_ok': done.returncode == 0,
                         'problems': problems})
            save(out / 'completion.json', {'status': 'running', 'rows': rows,
                                           'elapsed_seconds': round(time.monotonic() - started, 3)})
            if problems:
                raise RuntimeError(f"{path['name']} failed: " + '; '.join(problems))
            if not done.returncode == 0:
                raise RuntimeError(f"{path['name']} prediction failed")
        after = ledger.project_summary()
        completion = {'status': 'complete', 'rows': rows,
                      'paths': len(rows),
                      'actual_classifier_fits': sum(r['fits'] for r in rows),
                      'real_API_requests': sum(r['calls'] for r in rows),
                      'course_cost_rmb': (after['known_course_cost_rmb'] or 0) - (before['known_course_cost_rmb'] or 0),
                      'new_unknown_requests': after['unresolved_requests'] - before['unresolved_requests'],
                      'pending_requests': after['pending_requests'],
                      'ledger_sha256_after': sha(args.ledger),
                      'elapsed_seconds': round(time.monotonic() - started, 3), 'automatic_retries': False}
        save(out / 'completion.json', completion)
        print(json.dumps(completion, ensure_ascii=False, indent=1))
        return 0
    except Exception as error:
        detail = f'{type(error).__name__}: {error}'.replace(
            Path(args.api_key_file).read_text(encoding='utf-8').strip(), '[redacted]')
        save(out / 'completion.json', {'status': 'stopped', 'error': detail, 'rows': rows,
                                       'elapsed_seconds': round(time.monotonic() - started, 3),
                                       'automatic_retries': False})
        print(json.dumps({'status': 'stopped', 'error': detail}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
