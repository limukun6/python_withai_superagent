"""No-fit, no-network development replay of historical versus current classifier mentions."""
import argparse
import json
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from superagent.ab_run import sources
from superagent.prose_claims import check_prose

CASES = [
    ('natural_A89', 'reason', 'It is a reasonable next primary-split probe after a strongly regularized one-hot logistic baseline.', {'logistic':'informational'}),
    ('following', 'reason', 'Try CatBoost following a logistic regression baseline.', {'catboost':'confirmed','logistic':'informational'}),
    ('linear_reference', 'reason', 'Try CatBoost after a linear SVM baseline.', {'catboost':'confirmed','linear_svm':'informational'}),
    ('current_wrong', 'reason', 'After the logistic baseline, the selected model is an SVM.', {'logistic':'informational','svm':'contradicted'}),
    ('current_true', 'reason', 'Following the SVM baseline, this model is CatBoost.', {'svm':'informational','catboost':'confirmed'}),
    ('current_then_history', 'reason', 'This model is SVM after a logistic baseline.', {'svm':'contradicted','logistic':'informational'}),
    ('simple_wrong', 'reason', 'This model is logistic regression.', {'logistic':'contradicted'}),
    ('simple_true', 'reason', 'This model is CatBoost.', {'catboost':'confirmed'}),
    ('no_reference_marker', 'reason', 'the logistic regression baseline', {'logistic':'contradicted'}),
    ('uncertain', 'reason', 'We might try SVM after a logistic baseline.', {'svm':'unverifiable','logistic':'informational'}),
    ('explicit_trial', 'reason', 'Trial 1 logistic baseline, current classifier is CatBoost.', {'logistic':'informational','catboost':'confirmed'}),
    ('sentence_boundary', 'reason', 'Previous SVM was poor; this model is CatBoost.', {'svm':'informational','catboost':'confirmed'}),
    ('comparison', 'reason', 'CatBoost is stronger than the logistic baseline.', {'catboost':'confirmed','logistic':'informational'}),
    ('next_binding', 'reason', 'After a logistic baseline, the next model is CatBoost.', {'logistic':'informational','catboost':'confirmed'}),
    ('proposed_wrong', 'reason', 'Following a CatBoost baseline, the proposed candidate is SVM.', {'catboost':'informational','svm':'contradicted'}),
    ('unbound_review', 'review', 'We tested logistic regression yesterday.', {'logistic':'unverifiable'}),
]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coupon-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-pass', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def offline(event, args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('A98 is a no-network analysis')
    sys.addaudithook(offline)
    save(a.output/'protocol.json', dict(source_files=sources(), runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A98_historical_baseline_scope_plan.md'),
        cases=CASES, actual_classifier_fits=0, real_API_requests=0,
        scope='Developer-labeled regression cases, not independent semantic validation'))
    rows = []
    for name, field, text, expected in CASES:
        result = check_prose({field:text}, {'classifier':'catboost'}, [dict(trial=1)])
        observed = {r['claim']:r['status'] for r in result['items'] if r['kind']=='classifier'}
        rows.append(dict(id=name, expected=expected, observed=observed, passed=observed==expected))
    replay = []
    for arm in ('R0','R1','E0','E1'):
        h = read(a.coupon_run/arm/'history.json')
        for i, row in enumerate(h):
            if not row.get('proposal'):
                continue
            old = row['explanation_audit']['prose_claims']
            fields = row['explanation_audit']['original_prose']['fields']
            facts = dict(classifier=row['configuration_fact_check']['observed'])
            new = check_prose(fields, facts, h[:i], kind='categorical_tabular')
            replay.append(dict(arm=arm, trial=row['trial'], old=old, audited=new))
    save(a.output/'results.json', dict(development_cases=rows, natural_proposal_replay=replay))
    failed = [r['id'] for r in rows if not r['passed']]
    summary = dict(status='passed' if not failed else 'observed_known_failures',
        development_passed=len(rows)-len(failed), development_total=len(rows), failed=failed,
        natural_proposals=len(replay), old_machine_contradictions=sum(r['old']['contradicted'] for r in replay),
        audited_machine_contradictions=sum(r['audited']['contradicted'] for r in replay),
        actual_classifier_fits=0, real_API_requests=0, elapsed_seconds=time.monotonic()-began)
    save(a.output/'completion.json', summary)
    print(summary)
    if failed and a.require_pass:
        raise ValueError('Development scope checks failed; original expectations retained')
