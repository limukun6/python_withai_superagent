"""Run declared no-fit integration tests under network/training guards and retain exact failures."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import unittest
from scripts.a_batch_common import save,hashes

ROOT=Path(__file__).resolve().parents[1]
MODULES=['test_ab_adaptive_opening','test_ab_complement','test_ab_encoder_coverage',
         'test_native_familiarity_integration','test_ab_single_conflict','test_a112_native_scope',
         'test_a120_parameter_semantics','test_ab_text_kernel','test_ab_vision','test_external_assets',
         'test_prose_claims','test_configuration_facts','test_explanation_audit','test_ab_flash',
         'test_a126_registered_options']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    missing=[m for m in MODULES if not (ROOT/'tests'/ (m+'.py')).exists()]
    save(a.output/'protocol.json',dict(batch='A126',source_files=source,modules=MODULES,missing_modules=missing,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),maximum_classifier_fits=0,real_API_requests=0))
    try:
        from scripts import a38_predict_guard
        present=['tests.'+m for m in MODULES if m not in missing]
        suite=unittest.defaultTestLoader.loadTestsFromNames(present)
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (a.output/'regression.log').write_text(stream.getvalue(),encoding='utf-8')
        stats=dict(tests_run=result.testsRun,failures=[t.id() for t,_ in result.failures],
            errors=[t.id() for t,_ in result.errors],missing_modules=missing,source_unchanged=hashes(ROOT)==source,
            actual_classifier_fits=0,real_API_requests=0)
        save(a.output/'results.json',stats)
        save(a.output/'completion.json',dict(**stats,status='passed' if result.wasSuccessful() and not missing else 'failed',
            elapsed_seconds=time.monotonic()-start))
        print(json.dumps(stats),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
