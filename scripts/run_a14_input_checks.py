"""Persist symmetric-input and goal/semantics checks; zero real fits or API."""
import argparse
import contextlib
from pathlib import Path
import unittest
from unittest.mock import patch
from sklearn.dummy import DummyClassifier
from scripts.a_batch_common import Journal

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    suites=['tests.test_symmetric_input','tests.test_task_goal','tests.test_task_semantics']
    journal=Journal(args.output,Path(__file__).resolve().parents[1],dict(batch='A14',suites=suites,
        regression_dummy_fit_limit=1,real_api_requests=0,scope='symmetric task input and pre-fit rejection; one existing dummy fixture fit'))
    calls=[];original=DummyClassifier.fit
    def counted(model,*fit_args,**kwargs):
        if calls:raise RuntimeError('Unexpected extra dummy regression fit')
        calls.append('DummyClassifier');return original(model,*fit_args,**kwargs)
    with (journal.out/'tests.log').open('w',encoding='utf-8') as file:
        with contextlib.redirect_stdout(file),contextlib.redirect_stderr(file),patch.object(DummyClassifier,'fit',counted):
            result=unittest.TextTestRunner(stream=file,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(suites))
    journal.finish([],tests=result.testsRun,all_tests_passed=result.wasSuccessful(),failures=len(result.failures),
        errors=len(result.errors),actual_classifier_fits=len(calls),classifier_fit_calls=calls)
    if not result.wasSuccessful():raise SystemExit(1)
