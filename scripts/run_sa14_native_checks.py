"""SA14: native semantic contracts and production entry checks, no fitting or API."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    modules=['tests.test_native_semantics','tests.test_inspection_source','tests.test_live_entry','tests.test_task_budget','tests.test_ab_flash','tests.test_semantic_artifact','tests.test_csv_label_injectivity','tests.test_submission_equivalence','tests.test_multiturn_panel','tests.test_real_delivery_cases','tests.test_semantic_tool_repair','tests.test_semantic_panel','tests.test_semantic_autonomy_merge','tests.test_proactive_probe','tests.test_output_recipe',
             'tests.test_semantic_contract','tests.test_adaptive_formats','tests.test_anonymous_policy',
             'tests.test_task_readme','tests.test_metric_and_interruption',
             'tests.test_task_opening','tests.test_feedback_reflection','tests.test_planner_review',
             'tests.test_cross_task_exploration','tests.test_conditional_review',
             'tests.test_proposal_value','tests.test_scheduled_review','tests.test_protected_stop']
    # These old B fixtures call DummyClassifier.fit directly. They are outside
    # this zero-fit batch, not silently counted as passing or as new fits=0.
    excluded={
        'tests.test_proposal_value.ProposalValueTests.test_protected_pointer_survives_interruption_during_final_refit',
        'tests.test_proposal_value.ProposalValueTests.test_independent_confirmation_binds_the_frozen_models_and_can_go_either_way'}
    paths=['superagent/data.py','superagent/native_semantics.py','tests/test_native_semantics.py','superagent/schema.py','superagent/folder_images.py','superagent/image_files.py','tests/test_inspection_source.py','superagent/accepted_history.py','superagent/campus_transport.py','tests/test_live_entry.py','tools/accepted_usage_snapshot.py','superagent/semantic_contract.py','superagent/output_recipe.py','superagent/format_intake.py',
           'superagent/autonomous_run.py','superagent/unified.py','superagent/cli.py','superagent/task_opening.py',
           'superagent/autonomous_checkpoint.py','superagent/experiment_llm.py','superagent/task_budget.py',
           'unified_agent.py','tests/test_semantic_autonomy_merge.py','tests/test_semantic_tool_repair.py','tests/test_semantic_panel.py','scripts/run_sa14_native_checks.py','superagent/semantic_artifact.py','tests/test_semantic_artifact.py','tools/submission_equivalence.py','tests/test_csv_label_injectivity.py','tests/test_submission_equivalence.py','tests/test_multiturn_panel.py','tests/test_real_delivery_cases.py']
    identity={p:sha(root/p) for p in paths}
    save('protocol.json',dict(batch='SA14-native-semantics-checks',A='0d09760',B='b0944ae',source_sha256=identity,
        test_modules=modules,excluded_fit_fixtures=sorted(excluded),
        plan_sha256=sha(root/'docs/batches/SA14_native_semantics_plan_20261127_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0,
        scope='simulated clients, real readers, fixed untrained prediction fixtures and mocked workers'))
    def flatten(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite): yield from flatten(item)
            else: yield item
    selected=[]; removed=[]
    for test in flatten(unittest.TestLoader().loadTestsFromNames(modules)):
        if test.id() in excluded: removed.append(test.id())
        else: selected.append(test)
    if set(removed)!=excluded:
        raise RuntimeError('Declared exclusions did not match actual fixtures: '+repr(removed))
    start=time.monotonic()
    from superagent.ab_run import Executor
    with (args.output/'tests.log').open('w',encoding='utf-8') as stream:
        with patch.object(Executor,'worker',side_effect=AssertionError('SA14 forbids fitting')):
            with patch.object(socket,'create_connection',side_effect=AssertionError('SA14 forbids networking')):
                result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(selected))
    unchanged=identity=={p:sha(root/p) for p in paths}
    passed=result.wasSuccessful() and unchanged
    receipt=dict(status='passed' if passed else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_unchanged=unchanged,
        excluded_fit_fixtures=removed,actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        elapsed_seconds=time.monotonic()-start,scope='production accepted-history transactions, per-client campus HTTPS, interpretation and CLI checks; no real requests')
    save('results.json',receipt); save('completion.json',receipt)
    print(json.dumps(receipt)); return not passed


if __name__=='__main__': raise SystemExit(main())
