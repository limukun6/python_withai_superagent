"""No-network/no-fit regression and source-bound SA1 receipt; never calls a model API."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    modules = ['tests.test_semantic_contract', 'tests.test_adaptive_formats',
               'tests.test_anonymous_policy', 'tests.test_task_readme']
    files = [root/'superagent/semantic_contract.py', root/'superagent/autonomous_run.py',
             root/'tests/test_semantic_contract.py', root/'tests/test_adaptive_formats.py', Path(__file__)]
    identity = {str(f.relative_to(root)): digest(f) for f in files}
    save('protocol.json', dict(batch='SA1-semantic-contract', test_modules=modules, source_sha256=identity,
         maximum_fits=0, maximum_API_requests=0, model='scripted fixture, not a live LLM',
         plan_sha256=digest(root/'docs/batches/SA1_semantic_contract_plan_20261126_zh.md')))
    started = time.monotonic()
    from superagent.ab_run import Executor
    with (a.output/'tests.log').open('w', encoding='utf-8') as stream:
        with patch.object(Executor, 'worker', side_effect=AssertionError('SA1 forbids physical fitting')):
            with patch.object(socket, 'create_connection', side_effect=AssertionError('SA1 forbids networking')):
                suite = unittest.TestLoader().loadTestsFromNames(modules)
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    unchanged = identity == {str(f.relative_to(root)): digest(f) for f in files}
    passed = result.wasSuccessful() and unchanged
    receipt = dict(status='passed' if passed else 'failed', tests_run=result.testsRun,
                   errors=len(result.errors), failures=len(result.failures), source_unchanged=unchanged,
                   actual_classifier_fits=0, real_API_requests=0, course_cost_rmb=0,
                   elapsed_seconds=time.monotonic()-started,
                   scope='simulated LLM; real decoding and official pre-fit entry; no performance/semantic accuracy claim')
    save('results.json', receipt)
    save('completion.json', receipt)
    print(json.dumps(receipt))
    return not passed


if __name__ == '__main__': raise SystemExit(main())
