"""No-fit/no-network asset contract verification and unchanged legacy frozen-bundle inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    save(out/'protocol.json',dict(batch='A80',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A80_encoder_delivery_contract_plan.md'),actual_classifier_fits=0,real_api_requests=0))
    program='''import json,pathlib,sys,unittest
from scripts import a38_predict_guard
from superagent.ab_run import sources
before=sources()
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_external_assets']))
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(cases=result.testsRun,failures=len(result.failures),errors=len(result.errors),source_unchanged=sources()==before)))
raise SystemExit(not result.wasSuccessful())'''
    with (out/'checks.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-c',program,str(out/'checks.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:raise RuntimeError('External resource contract tests failed')
    legacy=a.reference/'sms_auc/export/bundle';features=a.reference/'sms_auc/test.zip'
    with (out/'legacy.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-I',str(legacy/'predict.py'),'--features',str(features),'--output',str(out/'legacy.csv')],
                              cwd=out,env={**os.environ,'SUPERAGENT_MINILM_WEIGHTS':'','SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':''},
                              stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode or (out/'legacy.csv').read_bytes()!=(a.reference/'sms_auc/isolated.csv').read_bytes():
        raise ValueError('Legacy lexical bundle no longer works without encoder assets')
    save(out/'completion.json',dict(status='passed',contract_tests=json.loads((out/'checks.json').read_text()),
        actual_classifier_fits=0,real_api_requests=0,A_cumulative_fits_after=1239,legacy_bundle_CSV_unchanged=True,
        controlled_gate_not_real_encoder_prediction=True,elapsed_seconds=time.monotonic()-began))
