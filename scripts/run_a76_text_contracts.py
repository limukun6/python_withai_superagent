"""Run merged text and current A contract suites with classifier fitting/network forbidden."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha
from scripts.run_a73_review_contracts import SUITES

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    suites=[*SUITES,'tests.test_ab_text_kernel','tests.test_ab_text_recipes',
            'tests.test_ab_canonical_preference_output','tests.test_ab_local_text_entry']
    save(out/'protocol.json',dict(batch='A76',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A76_text_incremental_integration_plan.md'),
        B_commits=['07a3706','1453394','d0b9fb3'],suites=suites,actual_classifier_fits=0,real_api_requests=0))
    code='''import json,pathlib,sys,time,unittest
from scripts import a38_predict_guard
from superagent.ab_run import sources
start=time.monotonic();before=sources()
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(sys.argv[2:]))
summary=dict(status='passed' if result.wasSuccessful() else 'failed',cases=result.testsRun,failures=len(result.failures),errors=len(result.errors),
actual_classifier_fits=0,real_api_requests=0,source_unchanged=before==sources(),source_files=before,elapsed_seconds=time.monotonic()-start)
pathlib.Path(sys.argv[1]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
raise SystemExit(not result.wasSuccessful())'''
    env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONIOENCODING':'utf-8','SUPERAGENT_OFFLINE':'1',
         'SUPERAGENT_MINILM_WEIGHTS':'','SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':''}
    with (out/'checks.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-c',code,str(out/'completion.json'),*suites],cwd=ROOT,env=env,
                              stdout=log,stderr=subprocess.STDOUT,timeout=120)
    print(json.dumps({k:v for k,v in json.loads((out/'completion.json').read_text()).items() if k!='source_files'}))
    raise SystemExit(result.returncode)
