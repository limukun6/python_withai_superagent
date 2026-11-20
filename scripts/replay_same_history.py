"""Replay one recorded decision state under both feedback contexts.

Two modes:

    --verify   rebuild every historical request payload from the recorded
               decision states and compare it with events.jsonl. Costs nothing.
    --run      ask the model the same state under both contexts, optionally
               repeated, to measure per-context stability and paired divergence.

The verified payload is produced by the same `decision_context` function the
experiment used, so a successful --verify means the replay is the historical
prompt, not a lookalike.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / 'runs' / 'integration'
if not (INTEGRATION / 'superagent').is_dir():
    INTEGRATION = REPO
sys.path.insert(0, str(INTEGRATION))

from superagent.checked_history_policy import CheckedHistoryPolicy, decision_context
from superagent.contracts import DecisionState
from superagent.events import RunLogger


def default_key_file():
    """The key sits at the main checkout, which is the parent of this worktree."""
    for candidate in (REPO / 'api_key.txt', INTEGRATION.parent.parent / 'api_key.txt'):
        if candidate.is_file():
            return str(candidate)
    return str(REPO / 'api_key.txt')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def events(run_dir):
    rows = [json.loads(line) for line in
            (Path(run_dir) / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    requests = [r for r in rows if 'messages' in r]
    answers = [r for r in rows if 'content' in r]
    payloads = [json.loads(r['messages'][1]['content']) for r in requests]
    replies = []
    for row in answers:
        try:
            replies.append(json.loads(row['content']))
        except ValueError:
            replies.append(None)
    return [r['role'] for r in requests], payloads, replies


def plan(run_dir):
    """(task, arm, decision_number) for each recorded request, in recording order."""
    order = read(Path(run_dir) / 'protocol.json')['planned_order']
    return [(task, arm, step) for task in order for arm in order[task] if arm != 'local'
            for step in (1, 2, 3)]


def states(run_dir, task, arm):
    return read(Path(run_dir) / task / arm / 'decision_states.json')


def context(run_dir, task, arm, step, checked):
    """The context for decision `step` (1-based) of one recorded arm.

    State 0 is the shared anchor, which is decided locally, so the k-th model
    request reads state index k rather than k-1.
    """
    return decision_context(DecisionState(states(run_dir, task, arm)[step]), checked)


def diff_keys(left, right):
    """Top-level keys that differ, plus the keys present in only one side."""
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    changed = sorted(k for k in set(left) & set(right) if left[k] != right[k])
    return only_left, only_right, changed


def verify(run_dir):
    roles, payloads, _ = events(run_dir)
    expected = plan(run_dir)
    if len(expected) != len(payloads):
        raise SystemExit(f'protocol plans {len(expected)} requests, events.jsonl has {len(payloads)}')
    failures = 0
    for index, ((task, arm, step), role, recorded) in enumerate(zip(expected, roles, payloads)):
        checked = arm == 'checked'
        if role != ('checked_history' if checked else 'raw_history'):
            print(f'request {index}: planned {task}/{arm} but recorded role is {role}')
            failures += 1
            continue
        rebuilt = context(run_dir, task, arm, step, checked)
        only_recorded, only_rebuilt, changed = diff_keys(recorded, rebuilt)
        # The arm is its own counterfactual, so per-trial wall clock is identical here.
        if only_recorded or only_rebuilt or changed:
            print(f'request {index} {task}/{arm}/d{step}: MISMATCH '
                  f'only_recorded={only_recorded} only_rebuilt={only_rebuilt} changed={changed}')
            failures += 1
        else:
            print(f'request {index} {task}/{arm}/d{step}: identical ({len(json.dumps(rebuilt))} bytes)')
    # Show what the other context would have added or removed on the same state.
    print('\ncounterfactual delta on the same recorded state')
    for task, arm, step in expected:
        here = context(run_dir, task, arm, step, arm == 'checked')
        other = context(run_dir, task, arm, step, arm != 'checked')
        only_here, only_other, changed = diff_keys(here, other)
        print(f'  {task}/{arm}/d{step}: -{only_here} +{only_other} changed={changed}')
    return 1 if failures else 0


def run(run_dir, output, repeats, api_key_file, budget):
    from superagent.cli import CAMPUS_CHAT_ENDPOINT, CAMPUS_FLASH_MODEL
    from superagent.experiment_llm import ExperimentLLM
    plan_rows = plan(run_dir)
    # Fail before creating a directory or spending anything if an index is off:
    # state k is the decision for trial k+1 because trial 1 is the shared anchor.
    for task, arm, step in plan_rows:
        record = states(run_dir, task, arm)[step]
        if record['next_trial'] != step + 1:
            raise SystemExit(f'{task}/{arm}/d{step}: state has next_trial={record["next_trial"]}, '
                             f'expected {step + 1}; refusing to spend on the wrong state')
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(destination)
    logger.initialize_usage_ledger()
    client = ExperimentLLM(CAMPUS_CHAT_ENDPOINT, str(Path(api_key_file).resolve()), CAMPUS_FLASH_MODEL,
                           batch_budget=budget, max_requests=len(plan_rows) * 2 * repeats, streaming=True,
                           allowed_served_models=['DeepSeek-V4.1-Flash', 'DeepSeek-V4.1-Flash-1017', 'DSF'])
    results = []
    try:
        for task, arm, step in plan_rows:
            row = {'task': task, 'recorded_arm': arm, 'decision': step, 'history': [
                h['candidate'] for h in states(run_dir, task, arm)[step]['history']], 'contexts': {}}
            for checked in (False, True):
                name = 'checked' if checked else 'raw'
                answers, errors = [], []
                for attempt in range(repeats):
                    policy = CheckedHistoryPolicy(client, logger, checked)
                    state = DecisionState(states(run_dir, task, arm)[step])
                    try:
                        proposal = policy.propose(state)
                        answers.append(proposal.candidate)
                    except Exception as error:
                        errors.append(f'{type(error).__name__}: {error}'.replace(client.key, '[redacted]'))
                        break
                    print(f'{task}/{arm}/d{step} {name} #{attempt + 1}: {answers[-1]}', flush=True)
                row['contexts'][name] = {'answers': answers, 'errors': errors,
                                         'stable': len(set(answers)) == 1 and not errors}
            raw, checked_row = row['contexts']['raw'], row['contexts']['checked']
            row['paired_same'] = bool(raw['answers'] and checked_row['answers']
                                      and raw['answers'][0] == checked_row['answers'][0])
            results.append(row)
            (destination / 'replay.json').write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                                     encoding='utf-8')
            if any(row['contexts'][name]['errors'] for name in ('raw', 'checked')):
                raise RuntimeError('A replay request failed; stopping instead of retrying')
        summary = {'status': 'complete', 'requests': client.request_count, 'course_cost_rmb': client.spent,
                   'repeats': repeats, 'states': len(results),
                   'stable_both': sum(r['contexts']['raw']['stable'] and r['contexts']['checked']['stable']
                                      for r in results),
                   'paired_same': sum(r['paired_same'] for r in results)}
        (destination / 'completion.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)
    except Exception as error:
        detail = f'{type(error).__name__}: {error}'.replace(client.key, '[redacted]')
        (destination / 'completion.json').write_text(json.dumps(
            {'status': 'stopped', 'error': detail, 'requests_attempted': client.request_count,
             'known_course_cost_rmb': client.spent}, indent=2), encoding='utf-8')
        print(detail, flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run-dir', required=True, help='recorded batch directory')
    parser.add_argument('--verify', action='store_true', help='rebuild payloads offline, no API calls')
    parser.add_argument('--run', action='store_true', help='send the replay requests')
    parser.add_argument('--output', help='new directory for --run')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--api-key-file', default=default_key_file())
    parser.add_argument('--budget', type=float, default=1.0,
                        help='course-cost ceiling; the client reserves an over-estimate per call, '
                             'so actual spend stays well below this')
    args = parser.parse_args()
    if args.verify:
        return verify(args.run_dir)
    if not args.output:
        parser.error('--run needs --output pointing at a new directory')
    if not 1 <= args.repeats <= 5:
        parser.error('--repeats must be 1..5')
    return run(args.run_dir, args.output, args.repeats, args.api_key_file, args.budget)


if __name__ == '__main__':
    raise SystemExit(main())
