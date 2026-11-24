"""A151: zero-fit, zero-API acceptance for knowledge/execution issue layering.

Two independent checks, both offline:

1. Program-oracle cases over constructed claims, with the expected channel and
   closure written down before running.
2. Replay of the twelve sealed A146 answers under `layering=off` and
   `layering=by-class`, comparing accepted candidate and issue status.

Nothing here trains a classifier or calls the school API.
"""
from copy import deepcopy
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from superagent.fact_registry import check_claim
from superagent.grounded_proposal import GroundingError, assess
from superagent.grounding_issues import (CHANNEL, EXECUTION, KNOWLEDGE, MEASUREMENT, address_issues,
                                         claim_class, confirm_execution, issue_for)

BATCH = 'a151_issue_layering'
EVIDENCE = REPO / 'research' / BATCH
SEALED = REPO / 'research' / 'a145_a146_grounded_pilot'
TASKS = ('coupon', 'sms', 'flowers')
CONDITIONS = ('natural', 'controlled')
MATCHED_ROW = {'trial': 4, 'status': 'completed',
               'metric_identity': {'status': 'confirmed'},
               'configuration_fact_check': {'status': 'confirmed'}}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform: the recorded sha256 must match the committed bytes.
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8') + b'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry_for(state, capabilities):
    from superagent.grounded_proposal import registry_for as build
    return build(state, capabilities)


def load_case(task, condition):
    root = SEALED / 'inputs' / task
    state = json.loads((root / f'{condition}_state.json').read_text(encoding='utf-8'))
    capabilities = json.loads((root / 'capabilities.json').read_text(encoding='utf-8'))
    folder = SEALED / 'paid' / task / f'{condition}_grounded'
    answer = json.loads((folder / 'answer.json').read_text(encoding='utf-8'))
    report = json.loads((folder / 'grounding_report.json').read_text(encoding='utf-8'))
    return state, capabilities, answer, report


def assess_once(state, capabilities, answer, layering):
    ids = [c['id'] for c in state['available_candidates']]
    try:
        proposal, report = assess(deepcopy(answer), deepcopy(state), capabilities, ids, layering)
        return {'accepted': True, 'candidate': proposal.candidate, 'report': report}
    except GroundingError as error:
        return {'accepted': False, 'candidate': None, 'report': error.report}


# ---------------------------------------------------------------- oracle cases
def oracle_cases(state, capabilities):
    registry = registry_for(state, capabilities)
    name = 'catboost_depth6_reg3.0'
    classifier = 'c:' + name + '.classifier'
    c_fact = 'c:' + name + '.C'
    have_c = c_fact in registry['facts']
    wrong_kind = {'kind': 'value', 'ref': 'task.kind', 'value': 'image'}
    right_kind = {'kind': 'value', 'ref': 'task.kind', 'value': state['task']['kind']}
    rows = []

    def run(label, claim, expected_class, expected_closure, correction=None):
        issue = issue_for(claim, registry, 1, 'by-class')
        observed_class = issue['issue_class'] if issue else None
        status_before = issue['status'] if issue else None
        new_fit = issue.get('new_fit_required') if issue else None
        closed = None
        if issue and correction is not None:
            addressed, errors = address_issues([issue], [correction], [issue['issue_id']], registry, 'by-class')
            closed = addressed[0]['status']
            reopened = confirm_execution(addressed, {'trial': 4, 'status': 'failed'}, 'by-class')[0]['status']
            settled = confirm_execution(addressed, MATCHED_ROW, 'by-class')[0]['status']
        elif issue:
            reopened = settled = None
        else:
            reopened = settled = None
        rows.append({'case': label,
                     'expected_issue_class': expected_class,
                     'observed_issue_class': observed_class,
                     'expected_closure': expected_closure,
                     'observed_closure': closed,
                     'status_on_issue_creation': status_before,
                     'new_fit_required': new_fit,
                     'after_failed_trial': reopened,
                     'after_matched_trial': settled,
                     'pass': (observed_class == expected_class and closed == expected_closure)})

    run('wrong task kind', wrong_kind, KNOWLEDGE, 'resolved_semantic', right_kind)
    if have_c:
        wrong_c = {'kind': 'value', 'ref': c_fact, 'value': -1.0}
        right_c = {'kind': 'value', 'ref': c_fact, 'value': registry['facts'][c_fact]['value']}
        run('wrong recorded C', wrong_c, KNOWLEDGE, 'resolved_semantic', right_c)
    # A claim about an effect that only exists after a run: no producer emits it yet.
    reserved = issue_for(wrong_kind, registry, 1, 'by-class')
    if reserved:
        execution = deepcopy(reserved)
        execution.update(issue_class=EXECUTION, resolution_channel=CHANNEL[EXECUTION], new_fit_required=True)
        addressed, _ = address_issues([execution], [right_kind], [execution['issue_id']], registry, 'by-class')
        settled = confirm_execution(addressed, MATCHED_ROW, 'by-class')[0]['status']
        reopened = confirm_execution(addressed, {'trial': 4, 'status': 'failed'}, 'by-class')[0]['status']
        rows.append({'case': 'reserved execution class',
                     'expected_issue_class': EXECUTION, 'observed_issue_class': execution['issue_class'],
                     'expected_closure': 'resolved', 'observed_closure': settled,
                     'after_failed_trial': reopened,
                     'new_fit_required': execution['new_fit_required'],
                     'pass': settled == 'resolved' and reopened == 'open'})
        measurement = deepcopy(reserved)
        measurement.update(issue_class=MEASUREMENT, resolution_channel=CHANNEL[MEASUREMENT], new_fit_required=True)
        addressed, _ = address_issues([measurement], [right_kind], [measurement['issue_id']], registry, 'by-class')
        settled = confirm_execution(addressed, MATCHED_ROW, 'by-class')[0]['status']
        rows.append({'case': 'reserved measurement class',
                     'expected_issue_class': MEASUREMENT, 'observed_issue_class': measurement['issue_class'],
                     'expected_closure': 'addressed_pending_check', 'observed_closure': settled,
                     'new_fit_required': measurement['new_fit_required'],
                     'pass': settled == 'addressed_pending_check'})
    unknown = issue_for({'kind': 'value', 'ref': 't9.search.accuracy', 'value': .5}, registry, 1, 'by-class')
    rows.append({'case': 'unknown fact reference', 'expected_issue_class': None,
                 'observed_issue_class': unknown['issue_class'] if unknown else None,
                 'expected_closure': None, 'observed_closure': None, 'pass': unknown is None})
    off_issue = issue_for(wrong_kind, registry, 1, 'off')
    rows.append({'case': 'layering off keeps the old postcondition',
                 'expected_issue_class': None,
                 'observed_issue_class': off_issue.get('issue_class'),
                 'expected_closure': 'addressed_pending_check', 'observed_closure': None,
                 'new_fit_required': off_issue.get('new_fit_required'),
                 'pass': 'issue_class' not in off_issue and 'new_fit_required' not in off_issue})
    return rows


# --------------------------------------------------------------------- replay
def replay_row(task, condition):
    state, capabilities, answer, sealed_report = load_case(task, condition)
    off = assess_once(state, capabilities, answer, 'off')
    layered = assess_once(state, capabilities, answer, 'by-class')

    def issues(payload):
        return [{k: i.get(k) for k in ('issue_id', 'status', 'issue_class', 'resolution_channel',
                                       'new_fit_required', 'closed_by')}
                | {'reason': (i.get('verification') or {}).get('reason')}
                for i in payload['report'].get('issues', [])]

    sealed_statuses = [i.get('status') for i in sealed_report.get('issues', [])]
    built = issues(layered)
    knowledge_open = sum(1 for i in built if i['issue_class'] == KNOWLEDGE and i['status'] != 'resolved_semantic')
    return {'task': task, 'condition': condition,
            'sealed_candidate': answer['candidate'], 'sealed_purpose': answer['purpose'],
            'sealed_issue_statuses': sealed_statuses,
            'sealed_issue_reasons': [(i.get('verification') or {}).get('reason')
                                     for i in sealed_report.get('issues', [])],
            'off': {'accepted': off['accepted'], 'candidate': off['candidate'], 'issues': issues(off)},
            'by_class': {'accepted': layered['accepted'], 'candidate': layered['candidate'], 'issues': built,
                         'knowledge_issues_not_closed': knowledge_open},
            'candidate_unchanged': off['candidate'] == layered['candidate'] == answer['candidate']}


def main():
    started = time.monotonic()
    if EVIDENCE.exists():
        print(f'{EVIDENCE} exists; refusing to overwrite', flush=True)
        return 1
    state, capabilities, _, _ = load_case('coupon', 'natural')
    cases = oracle_cases(state, capabilities)
    replay = [replay_row(task, condition) for task in TASKS for condition in CONDITIONS]
    sources = ['superagent/grounding_issues.py', 'superagent/grounded_proposal.py',
               'superagent/ab_flash.py', 'superagent/ab_run.py', 'superagent/cli.py',
               'tests/test_issue_layering.py', 'scripts/run_a151_issue_layering.py']
    protocol = {'batch': 'A151', 'title': 'knowledge and execution issue layering',
                'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'budget': {'classifier_fits': 0, 'school_api_calls': 0, 'cost_rmb': 0},
                'inputs': {'sealed_batch': 'research/a145_a146_grounded_pilot',
                           'sealed_answers': [f'{t}/{c}_grounded' for t in TASKS for c in CONDITIONS],
                           'matched_row_fixture': MATCHED_ROW},
                'expected': {'oracle_cases': len(cases), 'replay_cases': len(replay),
                             'all_oracle_pass': True, 'all_candidates_unchanged': True,
                             'knowledge_issues_closed': 3,
                             'measurement_closed_by_trial': False},
                'source_sha256': {p: sha(REPO / p) for p in sources},
                'note': 'Replay of sealed responses is a mechanism check, not a new model experiment.'}
    save(EVIDENCE / 'protocol.json', protocol)
    summary = {'oracle_cases': len(cases), 'oracle_passed': sum(r['pass'] for r in cases),
               'replay_cases': len(replay),
               'candidates_unchanged': sum(r['candidate_unchanged'] for r in replay),
               'sealed_knowledge_issues': sum(1 for r in replay for i in r['sealed_issue_statuses']
                                              if i == 'addressed_pending_check'),
               'closed_without_fit': sum(1 for r in replay for i in r['by_class']['issues']
                                         if i['status'] == 'resolved_semantic'),
               'knowledge_still_open': sum(r['by_class']['knowledge_issues_not_closed'] for r in replay)}
    results = {'summary': summary, 'oracle': cases, 'replay': replay}
    save(EVIDENCE / 'results.json', results)
    completion = {'status': 'complete' if summary['oracle_passed'] == summary['oracle_cases']
                  else 'oracle_failures', 'actual_classifier_fits': 0, 'real_API_requests': 0,
                  'course_cost_rmb': 0, 'elapsed_seconds': round(time.monotonic() - started, 3),
                  'protocol_sha256': sha(EVIDENCE / 'protocol.json'),
                  'results_sha256': sha(EVIDENCE / 'results.json'), 'automatic_retry': False}
    save(EVIDENCE / 'completion.json', completion)
    print(json.dumps({'summary': summary, 'completion': completion}, ensure_ascii=False, indent=1))
    return 0 if completion['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
