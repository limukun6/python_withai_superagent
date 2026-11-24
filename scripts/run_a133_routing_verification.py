"""Replay B's sealed inference receipts and routing contracts without training or network."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import sha

ROOT = Path(__file__).resolve().parents[1]
GUARD = '''import sys, runpy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
def no_fit(*args, **kwargs): raise RuntimeError('A133 forbids new classifier fits')
for cls in (LogisticRegression, ExtraTreesClassifier, SVC, LinearSVC, DummyClassifier): cls.fit = no_fit
try:
 from catboost import CatBoostClassifier
 CatBoostClassifier.fit = no_fit
except ImportError: pass
def offline(event,args):
 if event in ('socket.connect','socket.getaddrinfo','socket.sendto'): raise PermissionError('A133 forbids network')
sys.addaudithook(offline)
target=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(target,run_name='__main__')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); frozen=hashes(ROOT)
    tests=['test_ab_encoder_routing','test_ab_complement','test_ab_registered_rows',
           'test_ab_confirmation_budget','test_a126_registered_options','test_a120_parameter_semantics',
           'test_task_budget','test_a130_accepted_history']
    save(out/'protocol.json',dict(batch='A133',source_files=frozen,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A133_A135_next_integration_and_task1_plan_20261123_zh.md'),
         release=str(a.release),fit_cap=0,model_api_cap=0,tests=tests))
    results=[]
    try:
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SUPERAGENT_OFFLINE':'1',
             'PYTHONIOENCODING':'utf-8','PYTHONPATH':str(ROOT)}
        for name in ('verify_b_encoder_cascade','verify_b_native_encoder_routing'):
            path=a.release/'scripts'/(name+'.py'); before=time.monotonic()
            with (out/(name+'.log')).open('x',encoding='utf-8') as stream:
                r=subprocess.run([sys.executable,'-c',GUARD,str(path)],cwd=a.release,env=env,
                                 stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            if r.returncode: raise RuntimeError('B receipt verifier failed: '+name)
            results.append(dict(verifier=name,runner_sha256=sha(path),seconds=time.monotonic()-before,
                                report=json.loads((out/(name+'.log')).read_text(encoding='utf-8').splitlines()[-1])))
        testcode=GUARD.split('target=sys.argv[1]')[0]+"\nimport unittest\nsuite=unittest.TestLoader().loadTestsFromNames("+repr(['tests.'+t for t in tests])+")\nr=unittest.TextTestRunner(verbosity=2).run(suite)\nraise SystemExit(not r.wasSuccessful())\n"
        with (out/'targeted_tests.log').open('x',encoding='utf-8') as stream:
            r=subprocess.run([sys.executable,'-c',testcode],cwd=ROOT,env=env,stdout=stream,
                             stderr=subprocess.STDOUT,timeout=180)
        if r.returncode: raise RuntimeError('Targeted routing/regression tests failed')
        if frozen!=hashes(ROOT): raise ValueError('Production source changed during verification')
        save(out/'results.json',dict(receipt_checks=results,targeted_tests_passed=True,
             evidence_scope='B historical receipts reused; no A performance rerun'))
        status='passed'
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0))
        raise
    save(out/'completion.json',dict(status=status,actual_classifier_fits=0,real_API_requests=0,
         elapsed_seconds=time.monotonic()-started,source_unchanged=True,
         results_sha256=sha(out/'results.json'),protocol_sha256=sha(out/'protocol.json')))
    print(json.dumps(dict(status=status,actual_classifier_fits=0,real_API_requests=0)))


if __name__=='__main__': main()
