"""Measured regression resource journal; selected suites, 24 real worker-fit cap."""
import argparse
import contextlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from a_batch_common import Journal,save

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();sys.path.insert(0,str(ROOT))
    import superagent.benchmark as benchmark
    suites=['tests.test_task_goal','tests.test_architecture_contracts','tests.test_general_extensions','tests.test_confirmation','tests.test_image_files']
    j=Journal(a.output,ROOT,{'batch':'A-window-regression','suites':suites,'real_worker_fit_limit':24,
                           'expected_worker_fits':21,'validate_only':a.validate_only,'note':'Preprocessor-only fit in general_extensions is separate'})
    rows=[];original=benchmark.train_candidate
    def counted(payload,candidate,destination,timeout):
        if len(rows)>=24:raise RuntimeError('Regression fit cap reached')
        row={'candidate':candidate,'status':'started'};rows.append(row)
        try:
            model,metrics=original(payload,candidate,destination,timeout)
            row.update(status=metrics['status'],classifier_fits=metrics.get('classifier_fits') if metrics['status']=='completed' else None)
            save(j.out/'worker_calls.json',rows)
            return model,metrics
        except Exception as error:row.update(status='exception',error=repr(error));save(j.out/'worker_calls.json',rows);raise
    if a.validate_only:j.finish([],classifier_fits=0);return
    with patch.object(benchmark,'train_candidate',side_effect=counted):
        suite=unittest.defaultTestLoader.loadTestsFromNames(suites)
        with (j.out/'tests.log').open('w',encoding='utf-8') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    known=all(r.get('classifier_fits') is not None for r in rows)
    j.finish(rows,tests=result.testsRun,all_tests_passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
             worker_attempts=len(rows),completed_classifier_fits=sum(r.get('classifier_fits') or 0 for r in rows),complete_count_known=known,
             preprocessing_only_fit_calls=1,simulated_http_requests=0)
    print({'tests':result.testsRun,'passed':result.wasSuccessful(),'worker_attempts':len(rows),'fits':sum(r.get('classifier_fits') or 0 for r in rows)})


if __name__=='__main__':main()
