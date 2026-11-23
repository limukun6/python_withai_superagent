"""No-network policy regression with temporary test ledgers, never the real project ledger."""
import argparse
import io
from pathlib import Path
import time
import unittest
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    modules=['tests.test_a130_accepted_history','tests.test_task_budget','tests.test_a126_registered_options']
    save(a.output/'protocol.json',dict(batch='A130-test',runner_sha256=sha(Path(__file__)),modules=modules,
        maximum_classifier_fits=0,real_API_requests=0,real_ledger_used_for_tests=False))
    from scripts import a38_predict_guard
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    (a.output/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    save(a.output/'results.json',dict(tests=result.testsRun,failures=[t.id() for t,_ in result.failures],errors=[t.id() for t,_ in result.errors]))
    save(a.output/'completion.json',dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        actual_classifier_fits=0,real_API_requests=0,elapsed_seconds=time.monotonic()-start))
    print(dict(tests=result.testsRun,success=result.wasSuccessful(),API=0),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
