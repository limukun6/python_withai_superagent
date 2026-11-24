"""No-fit regression subset while the live model queue runs; record actual observed fit calls."""
import argparse
import contextlib
from pathlib import Path
import sys
import time
import unittest

from scripts.a_batch_common import save
from scripts.classifier_fit_observer import observe_classifier_fits
from scripts.run_a10_real_task_pilot import sha
from superagent.ab_run import sources


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources()
    suites=['tests.test_training_profile','tests.test_calibrated_proposal','tests.test_ab_flash','tests.test_unattended_runtime','tests.test_campus_direct']
    save(args.output/'protocol.json',dict(batch='A50-no-fit-contract-checks',source_files=frozen,runner_sha256=sha(Path(__file__)),suites=suites))
    with (args.output/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log),observe_classifier_fits() as calls:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    save(args.output/'fit_observations.json',calls)
    record=dict(status='passed' if result.wasSuccessful() and not calls else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),actual_classifier_fits=len(calls),real_api_requests=0,
        source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start)
    save(args.output/'completion.json',record);print(record)
    if record['status']!='passed':raise SystemExit(1)
