"""No-fit/no-network development and default menu verification, before retained freeze."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import sha
from scripts.run_a133_routing_verification import GUARD
from scripts.run_a137_numeric_acceptance import MENU_CODE

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','baseline','cases'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();source=hashes(ROOT)
    tests=['tests.'+n for n in ('test_grounded_proposal','test_ab_flash','test_a139_compact_feedback',
        'test_a137_numeric_kernel','test_task_budget','test_ab_registered_rows','test_a130_accepted_history',
        'test_ab_confirmation_budget','test_a120_parameter_semantics')]
    save(out/'protocol.json',dict(batch='A145-development-verification',source_files=source,
        runner_sha256=sha(Path(__file__)),tests=tests,maximum_classifier_fits=0,maximum_API_requests=0,
        development_cases_sha256=sha(a.cases/'development.json'),retained_evaluated=False))
    env={**os.environ,'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'SUPERAGENT_OFFLINE':'1','PYTHONPATH':str(ROOT)}
    guard=GUARD.split('target=sys.argv[1]')[0]
    try:
        code=guard+'\nimport unittest\ns=unittest.TestLoader().loadTestsFromNames('+repr(tests)+')\nr=unittest.TextTestRunner(verbosity=2).run(s)\nraise SystemExit(not r.wasSuccessful())\n'
        with (out/'tests.log').open('x',encoding='utf-8') as stream:
            proc=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        if proc.returncode:raise RuntimeError('Grounding regression failed; see tests.log')
        menus=[]
        for folder in (a.baseline.resolve(),ROOT):
            raw=subprocess.check_output([sys.executable,'-c',guard+'\n'+MENU_CODE],cwd=folder,
                env={**env,'PYTHONPATH':str(folder)},text=True,timeout=120)
            menus.append(json.loads(raw))
        if menus[0]!=menus[1]:raise ValueError('Default candidate menu drift')
        save(out/'menus.json',dict(default_menus_unchanged=True,menus=menus[1]))
        with (out/'development.log').open('x',encoding='utf-8') as stream:
            proc=subprocess.run([sys.executable,'-m','scripts.evaluate_a145_grounding_cases','--cases',str(a.cases),
                 '--split','development','--output',str(out/'development')],cwd=ROOT,env=env,
                 stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        if proc.returncode:raise RuntimeError('Development evaluation failed')
        if source!=hashes(ROOT):raise ValueError('Production source changed')
        save(out/'completion.json',dict(status='passed',source_files=source,source_unchanged=True,
             actual_classifier_fits=0,real_API_requests=0,default_menus_unchanged=True,
             elapsed_seconds=time.monotonic()-start,tests_log_sha256=sha(out/'tests.log'),
             protocol_sha256=sha(out/'protocol.json')))
        print(json.dumps(dict(status='passed',actual_classifier_fits=0,real_API_requests=0)))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
