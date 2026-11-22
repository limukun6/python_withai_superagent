"""Contract-only checks against an isolated merged checkout, all classifier fits and network denied."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha,read

SUITES=['tests.test_score_evidence','tests.test_ab_finalists','tests.test_comparison_scope','tests.test_review_referent',
    'tests.test_ab_scheduler','tests.test_ab_fixed_schedule','tests.test_ab_confirmation_budget','tests.test_ab_fixed_allocation',
    'tests.test_ab_registered_rows','tests.test_ab_deployment_criteria','tests.test_ab_vision',
    'tests.test_ab_ordered_relation.OrderedRelationTests.test_catalogue_exposes_interactions_only_for_the_ordered_two_text_contract',
    'tests.test_ab_symmetric.TestInvariantPairMethods.test_frozen_features_encode_the_same_unique_text_batch_in_either_order',
    'tests.test_ab_symmetric.TestInvariantPairMethods.test_resolution_uses_fitted_code_and_rejects_other_semantics',
    'tests.test_ab_symmetric.TestInvariantPairMethods.test_equivalence_manifest_and_relation_preference_separation']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(batch='A73',A_base='67ba67f',B_base='fa58e85',runner_sha256=sha(Path(__file__)),
        merged_checkout=str(a.checkout),suites=SUITES,classifier_fitting_forbidden=True,network_forbidden=True))
    program='''import json,pathlib,sys,unittest
from scripts import a38_predict_guard
import superagent
root=pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(superagent.__file__).resolve().is_relative_to(root)
from superagent.ab_run import sources
before=sources()
suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[3:])
result=unittest.TextTestRunner(verbosity=1).run(suite)
summary=dict(status='passed' if result.wasSuccessful() else 'failed',cases=result.testsRun,failures=len(result.failures),errors=len(result.errors),
 actual_classifier_fits=0,real_api_requests=0,source_unchanged=before==sources(),source_files=before)
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
raise SystemExit(not result.wasSuccessful())'''
    env={**os.environ,'SUPERAGENT_OFFLINE':'1','PYTHONPATH':str(a.checkout),'PYTHONIOENCODING':'utf-8',
         'SUPERAGENT_MINILM_WEIGHTS':'','SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':''}
    with (a.output/'checks.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-c',program,str(a.checkout),str(a.output/'completion.json'),*SUITES],cwd=a.checkout,
                              env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    summary=read(a.output/'completion.json') if (a.output/'completion.json').exists() else dict(status='process_failed',returncode=result.returncode)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_files'}))
    raise SystemExit(result.returncode)
