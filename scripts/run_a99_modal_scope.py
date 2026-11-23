"""Bounded modal-hypothesis replay, no training or provider calls."""
import argparse
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from superagent.ab_run import sources
from superagent.prose_claims import check_prose

NATURAL = ('Among the available raw-pixel candidates, increasing logistic C is unlikely to help given the observed overfitting, '
    'while extra trees with min_samples_leaf=3 offers a nonlinear model with some regularization and is a stronger next PRIMARY '
    'experiment than unregularized leaf1 or raw-pixel RBF SVM.')
CASES = [
    ('natural_A97', NATURAL, dict(unverifiable=2, informational=2)),
    ('unlikely_intervention', 'Increasing logistic C is unlikely to improve generalization.', dict(unverifiable=1)),
    ('likely_intervention', 'A logistic model is likely to help.', dict(unverifiable=1)),
    ('future_trial', 'Trial 9 is likely to improve.', dict(unverifiable=1)),
    ('current_wrong', 'This model is SVM.', dict(contradicted=1)),
    ('current_true', 'This model is CatBoost.', dict(confirmed=1)),
    ('measured_true', 'Trial 1 search accuracy was 75%.', dict(confirmed=2)),
    ('measured_false', 'Trial 1 search accuracy was 99%.', dict(confirmed=1, contradicted=1)),
    ('distinct_sentences', 'This model is CatBoost. Increasing logistic C is unlikely to help.', dict(confirmed=1, unverifiable=1)),
    ('bare_likely', 'The current model is an SVM with likely measurement noise.', dict(contradicted=1)),
]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flower-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-pass', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def offline(event, args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('A99 no-network audit')
    sys.addaudithook(offline)
    save(a.output/'protocol.json', dict(source_files=sources(), runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A99_modal_hypothesis_scope_plan.md'),
        cases=CASES, actual_classifier_fits=0, real_API_requests=0,
        scope='Developer-labeled regression cases, not independent semantic validation'))
    history = [dict(trial=1, status='completed', primary_metric='accuracy', search_score=.75,
        search_accuracy=.75, split_id='primary')]
    rows = []
    for name, text, expected in CASES:
        report = check_prose(dict(reason=text), dict(classifier='catboost'), history)
        observed = {k:report[k] for k in ('confirmed','contradicted','unverifiable','informational') if report[k]}
        rows.append(dict(id=name, expected=expected, observed=observed, passed=observed==expected))
    replay = []
    for arm in ('R0','R1','E0','E1'):
        h = read(a.flower_run/arm/'history.json')
        for i, row in enumerate(h):
            if row.get('proposal'):
                audit = row['explanation_audit']
                new = check_prose(audit['original_prose']['fields'],
                    dict(classifier=row['configuration_fact_check']['observed']), h[:i], kind='image')
                replay.append(dict(arm=arm, trial=row['trial'], old=audit['prose_claims'], audited=new))
    save(a.output/'results.json', dict(development_cases=rows, natural_proposal_replay=replay))
    failed = [r['id'] for r in rows if not r['passed']]
    result = dict(status='passed' if not failed else 'observed_known_failures', development_passed=10-len(failed),
        development_total=10, failed=failed, natural_proposals=len(replay),
        original_machine_contradictions=sum(r['old']['contradicted'] for r in replay),
        audited_machine_contradictions=sum(r['audited']['contradicted'] for r in replay),
        actual_classifier_fits=0, real_API_requests=0, elapsed_seconds=time.monotonic()-start)
    save(a.output/'completion.json', result)
    print(result)
    if failed and a.require_pass:
        raise ValueError('Modal scope expectations failed; original expectations retained')
