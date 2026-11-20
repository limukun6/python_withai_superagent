"""Persist A1 regression logs for handoff; no live API credentials are read."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    suites=['tests.test_task_semantics','tests.test_estimator_facts','tests.test_targeted_feedback',
            'tests.test_general_extensions','tests.test_confirmation',
            'tests.test_architecture_contracts','tests.test_hypothesis_agent']
    command=[sys.executable,'-m','unittest',*suites,'-q']
    (out/'protocol.json').write_text(json.dumps(dict(command=command,cwd=str(repo),
        timeout_seconds=600,scope='regression fixtures, not task performance',
        live_api_disabled=True),indent=2),encoding='utf-8')
    start=time.monotonic()
    status='failed';returncode=1
    try:
        with (out/'tests.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=repo,env={**os.environ,'SUPERAGENT_OFFLINE':'1',
                'PYTHONIOENCODING':'utf-8'},stdout=log,stderr=subprocess.STDOUT,timeout=600)
        returncode=result.returncode
        status='complete' if returncode==0 else 'failed'
    finally:
        summary=dict(status=status,returncode=returncode,elapsed_seconds=time.monotonic()-start,
            api_requests=0,experimental_classifier_fits=0,
            fixture_classifier_fits='not independently counted; see tests.log; not charged to task experiments')
        (out/'completion.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary))
    return returncode

if __name__=='__main__':raise SystemExit(main())
