"""A151 menu-probe: one Flash request over the compressed menu representation.

Default is `--validate-only`: it rebuilds the payload, checks that decoding
restores it exactly, and stops. Only `--execute` makes one real request, with
zero classifier fits, and never retries.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.a_batch_common import save
from scripts.campus_direct import CAMPUS_IP, campus_resolution
from superagent.ab_flash import validate_proposal
from superagent.ab_run import registered_capabilities, sources
from superagent.events import RunLogger
from superagent.menu_view import compact_menu, decode_menu, recorded_size
from superagent.research_budget import IncrementalResearchAccount
from superagent.task_budget import TaskBudgetLedger
from tools.accepted_usage_snapshot import accepted_history

BATCH = 'a151_menu_probe'
EVIDENCE = REPO / 'research' / BATCH
SCOPE = 'a151_menu_probe'
ALLOWANCE = .15
MAX_REQUESTS = 1


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_request(events_path):
    for line in Path(events_path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('kind') == 'llm_request':
            return row
    raise SystemExit('No llm_request in the supplied events file')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--output', type=Path, default=EVIDENCE)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True)
    began = time.monotonic()
    request = load_request(args.events)
    system, payload_text = request['messages'][0]['content'], request['messages'][1]['content']
    payload = json.loads(payload_text)
    compact = compact_menu(payload)
    restored = decode_menu(compact)
    if restored != payload:
        raise SystemExit('Compressed menu does not round-trip; refusing to probe')
    frozen = sources()
    protocol = {'batch': 'A151-menu-probe', 'source_files': frozen,
                'runner_sha256': sha(Path(__file__)),
                'plan_sha256': sha(REPO / 'docs/batches/A151_menu_probe_plan_20261124_zh.md'),
                'request_from': str(args.events.resolve()),
                'original_payload_sha256': sha_bytes(payload_text.encode('utf-8')),
                'original_bytes': recorded_size(payload), 'compact_bytes': recorded_size(compact),
                'menu_encoding': compact['menu_encoding']['version'],
                'round_trip_exact': True,
                'candidates': len(payload['available_candidates']),
                'model': request['model'], 'contract': 'calibrated-v2',
                'maximum_requests': MAX_REQUESTS, 'maximum_fits': 0,
                'allowance_rmb': ALLOWANCE, 'execute_requested': args.execute,
                'automatic_retries': False}
    save(out / 'protocol.json', protocol)
    save(out / 'compact_payload.json', compact)
    if not args.execute:
        save(out / 'completion.json', {'status': 'validated_not_executed', 'real_API_requests': 0,
                                       'classifier_fits': 0, 'course_cost_rmb': 0,
                                       'elapsed_seconds': round(time.monotonic() - began, 3)})
        print(json.dumps({'status': 'validated_not_executed',
                          'original_bytes': protocol['original_bytes'],
                          'compact_bytes': protocol['compact_bytes'],
                          'reduction_pct': round(100 * (protocol['original_bytes'] - protocol['compact_bytes'])
                                                 / protocol['original_bytes'], 3)}))
        return 0
    if not (args.ledger and args.authorization and args.api_key_file):
        raise SystemExit('--execute needs ledger, authorization and api-key-file')
    from superagent.experiment_llm import ExperimentLLM
    ledger = TaskBudgetLedger(args.ledger)
    before = ledger.project_summary()
    auth_sha = sha(args.authorization)
    (out / 'run').mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out / 'run')
    logger.initialize_usage_ledger()
    account = IncrementalResearchAccount(ledger, SCOPE, str(out / 'run'))
    client = ExperimentLLM('https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions',
                           str(args.api_key_file), request['model'], ALLOWANCE,
                           max_requests=MAX_REQUESTS, allowed_served_models={request['model']},
                           account=account, streaming=True)
    client.set_deadline(began + 600)
    answer, error = None, None
    try:
        with accepted_history(args.authorization, auth_sha), campus_resolution(CAMPUS_IP):
            answer = client.ask(compact, logger, 'ab_menu_probe')
    except Exception as failure:  # transport/parse/usage: stop, no retry
        error = f'{type(failure).__name__}: {failure}'.replace(
            Path(args.api_key_file).read_text(encoding='utf-8').strip(), '[redacted]')
    result = {'error': error, 'answer': answer}
    if answer is not None:
        task_spec = payload['task_spec']
        capabilities = registered_capabilities(task_spec, 8301)
        result['capabilities_resolved'] = len(capabilities)
        ids = [c['id'] for c in payload['available_candidates']]
        try:
            proposal = validate_proposal(answer, {'task': task_spec, 'history': payload['history'],
                                                  'available_candidates': payload['available_candidates']},
                                         capabilities, ids, 'calibrated-v2')
            result['accepted'] = True
            result['candidate'] = proposal.candidate
            result['candidate_in_original_menu'] = proposal.candidate in ids
            result['params_match_catalogue'] = proposal.params == capabilities[proposal.candidate]['params']
        except Exception as invalid:
            result['accepted'] = False
            result['invalid_reason'] = f'{type(invalid).__name__}: {invalid}'
    save(out / 'results.json', result)
    after = ledger.project_summary()
    completion = {'status': 'complete' if error is None else 'stopped',
                  'error': error, 'accepted': result.get('accepted'),
                  'candidate': result.get('candidate'),
                  'real_API_requests': after['requests'] - before['requests'],
                  'classifier_fits': 0,
                  'course_cost_rmb': (after['known_course_cost_rmb'] or 0) - (before['known_course_cost_rmb'] or 0),
                  'new_unknown_requests': after['unresolved_requests'] - before['unresolved_requests'],
                  'ledger_sha256_after': sha(args.ledger),
                  'elapsed_seconds': round(time.monotonic() - began, 3), 'automatic_retries': False}
    save(out / 'completion.json', completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0 if completion['status'] == 'complete' and result.get('accepted') else 1


if __name__ == '__main__':
    raise SystemExit(main())
