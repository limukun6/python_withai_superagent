"""Run the source-bound SA3 no-fit/no-network protocol in a fresh evidence directory."""
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
    modules=['tests.test_proactive_probe','tests.test_output_recipe','tests.test_semantic_contract',
             'tests.test_adaptive_formats','tests.test_anonymous_policy','tests.test_task_readme',
             'tests.test_metric_and_interruption']
    paths=['superagent/semantic_contract.py','superagent/output_recipe.py','superagent/format_intake.py',
           'superagent/autonomous_run.py','superagent/unified.py','superagent/cli.py','unified_agent.py',
           'tests/test_proactive_probe.py','scripts/run_sa3_proactive_probe.py']
    identity={p:sha(root/p) for p in paths}
    save('protocol.json',dict(batch='SA3-proactive-probe',source_sha256=identity,test_modules=modules,
        plan_sha256=sha(root/'docs/batches/SA3_proactive_probe_plan_20261126_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0,
        scope='simulated LLM; real readers and pre-fit probes; mocked training child; fixed untrained export fixture'))
    start=time.monotonic()
    from superagent.ab_run import Executor
    with (args.output/'tests.log').open('w',encoding='utf-8') as stream:
        with patch.object(Executor,'worker',side_effect=AssertionError('SA3 forbids fitting')):
            with patch.object(socket,'create_connection',side_effect=AssertionError('SA3 forbids networking')):
                suite=unittest.TestLoader().loadTestsFromNames(modules)
                result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    unchanged=identity=={p:sha(root/p) for p in paths}
    passed=result.wasSuccessful() and unchanged
    receipt=dict(status='passed' if passed else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_unchanged=unchanged,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        elapsed_seconds=time.monotonic()-start,
        scope='engineering protocol only; no real-LLM success-rate, quality, or cost-saving claim')
    save('results.json',receipt); save('completion.json',receipt)
    print(json.dumps(receipt)); return not passed


if __name__=='__main__': raise SystemExit(main())
