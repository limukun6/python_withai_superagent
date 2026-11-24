"""Controlled projection check and no-fit related regressions, not a real LLM correction experiment."""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import time
import unittest
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A121-transport-regression',source_files=source,runner_sha256=sha(Path(__file__)),
        controlled_projection_fixture=True,not_real_next_request=True,maximum_classifier_fits=0,real_API_requests=0,
        excluded_fit_test='tests.test_task_semantics.SwapBehaviorTests.test_a_symmetric_model_is_detected_by_the_probe'))
    try:
        from scripts import a38_predict_guard
        from superagent.ab_flash import context_for
        from superagent.explanation_audit import audit_explanation
        records=read(ROOT/'research/a70_independent_review/blinded_reviews.json')
        row=next(r for r in records if r['review_id']=='29491f1bd1d1')
        proposal=row['proposal'];candidate=row['selected_registered_configuration'];history=row['history']
        facts={'classifier':candidate['estimator'],'source':'shared_ab_catalogue'}
        reports=[audit_explanation(proposal,history,candidate,parameter_semantics=flag,selected_facts=facts) for flag in (False,True)]
        states=[]
        for report in reports:
            state=dict(task=row['task_contract'],history=deepcopy(history),available_candidates=[candidate],
                budget=dict(fit_limit=11,attempted_classifier_fits=2,completed_classifier_fits=2,
                            remaining_fit_allowance=9,reserved_final_refit=1,llm_request_limit=3,used_llm_requests=1))
            # This attachment tests serialization only, not a claim about what trial 2 proposed.
            state['history'][-1]['explanation_audit']=report;states.append(state)
        caps={candidate['id']:{'estimator':candidate['estimator']}};eligible=[candidate['id']]
        raw=[context_for(s,caps,eligible,False,'calibrated-v2') for s in states]
        checked=[context_for(s,caps,eligible,True,'calibrated-v2') for s in states]
        if raw[0]!=raw[1]:raise ValueError('Diagnostic leaked into raw history input')
        for k in ('task_spec','history','available_candidates','candidate_facts','budget','instruction'):
            if checked[0][k]!=checked[1][k]:raise ValueError('Shared context changed: '+k)
        lengths=[len(json.dumps(c,ensure_ascii=False,separators=(',',':')).encode()) for c in checked]
        def flatten(suite):
            for item in suite:
                if isinstance(item,unittest.TestSuite):yield from flatten(item)
                else:yield item
        modules=['tests.test_a120_parameter_semantics','tests.test_a112_native_scope','tests.test_prose_claims',
                 'tests.test_explanation_audit','tests.test_configuration_facts','tests.test_task_semantics','tests.test_ab_flash']
        skipped='tests.test_task_semantics.SwapBehaviorTests.test_a_symmetric_model_is_detected_by_the_probe'
        tests=list(flatten(unittest.defaultTestLoader.loadTestsFromNames(modules)))
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(t for t in tests if t.id()!=skipped))
        (a.output/'regression.log').write_text(stream.getvalue(),encoding='utf-8')
        summary=dict(tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),
            intentionally_excluded_training_test=skipped,raw_context_unchanged=True,checked_common_fields_unchanged=True,
            checked_serialized_utf8_bytes_off_on=lengths,additional_utf8_bytes=lengths[1]-lengths[0],
            actual_token_increment=None,actual_cost_increment=None,real_API_requests=0,
            scope='One controlled transport fixture; bytes are not tokens or billed cost',
            default_diagnostic_enabled=False,source_unchanged=hashes(ROOT)==source)
        save(a.output/'results.json',summary)
        save(a.output/'completion.json',dict(status='passed' if result.wasSuccessful() else 'failed',actual_classifier_fits=0,
            real_API_requests=0,source_unchanged=hashes(ROOT)==source,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(summary),flush=True)
        if not result.wasSuccessful():raise RuntimeError('Related regression failed')
    except Exception as error:
        if not (a.output/'completion.json').exists():save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
