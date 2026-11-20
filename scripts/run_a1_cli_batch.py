"""Offline HTTP-fixture CLI verification; persists logs and source identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['baseline','fixed','regression'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=lambda:{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('superagent','tests') for p in sorted((REPO/folder).glob('*.py'))}
    frozen=source()
    suites=['tests.test_checked_cli'] if args.phase!='regression' else [
        'tests.test_unified_cli','tests.test_task_budget','tests.test_feedback_loop','tests.test_prose_claims',
        'tests.test_measurement_claims','tests.test_estimator_facts','tests.test_targeted_feedback',
        'tests.test_architecture_contracts','tests.test_hypothesis_agent','tests.test_task_semantics',
        'tests.test_general_extensions','tests.test_confirmation']
    command=[sys.executable,'-m','unittest',*suites,'-q']
    def save(name,x):(out/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('protocol.json',dict(phase=args.phase,source_sha256=frozen,command=command,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Mock HTTP with real API client/accounting; no live service'))
    start=time.monotonic()
    try:
        with (out/'tests.log').open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=REPO,env={**os.environ,'PYTHONIOENCODING':'utf-8',
                'SUPERAGENT_OFFLINE':'1','A1_04_ARTIFACT_DIR':str(out/'cases')},stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        log=(out/'tests.log').read_text(encoding='utf-8');match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
        stats=[json.loads(p.read_text(encoding='utf-8')) for p in (out/'cases').glob('*/stats.json')]
        unchanged=source()==frozen
        save('completion.json',dict(status='complete' if result.returncode==0 and unchanged else 'failed',
            phase=args.phase,returncode=result.returncode,tests=int(match[1]) if match else None,
            elapsed_seconds=time.monotonic()-start,source_unchanged=unchanged,real_api_requests=0,
            mock_http_requests=sum(r['mock_http_requests'] for r in stats),
            cli_experiment_classifier_fits=sum(r['classifier_fits'] for r in stats),
            regression_fixture_fits='not independently counted' if args.phase=='regression' else None))
        print((out/'completion.json').read_text(encoding='utf-8'))
        return 0 if args.phase=='baseline' else result.returncode or int(not unchanged)
    except Exception as error:
        save('completion.json',dict(status='failed',error=f'{type(error).__name__}: {error}',real_api_requests=0))
        raise

if __name__=='__main__':raise SystemExit(main())
