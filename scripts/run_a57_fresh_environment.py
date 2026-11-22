"""New venv, pinned binary dependencies and six offline predictions; original environment untouched."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


def launch(command,log,cwd,env,timeout):
    with log.open('w',encoding='utf-8') as file:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=file,stderr=subprocess.STDOUT)
        try:return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait(timeout=15)
            raise TimeoutError('Fresh-environment stage exceeded its registered time limit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundles','lock','window-config','output','real-ledger'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for name in ('bundles','lock','window_config','real_ledger'):setattr(args,name,getattr(args,name).resolve())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();results=[]
    deadline=datetime.fromisoformat(json.loads(args.window_config.read_text(encoding='utf-8-sig'))['deadline'])
    def left():return (deadline-datetime.now(timezone.utc)).total_seconds()
    ledger_hash=sha(args.real_ledger)
    save(root/'protocol.json',dict(batch='A57-fresh-env',runner_sha256=sha(Path(__file__)),lock_sha256=sha(args.lock),
        model_bundle_study_sha256=sha(args.bundles/'completion.json'),actual_classifier_fit_limit=0,real_API_limit=0,
        original_environment_modified=False,system_site_packages=False,automatic_retries=False,index='https://pypi.org/simple'))
    try:
        if read(args.bundles/'completion.json')['status']!='passed':raise ValueError('A54 must pass before dependency installation')
        if left()<1200:raise ValueError('Less than 20 minutes remain; no environment installation started')
        target=root/'venv';python=target/'Scripts/python.exe'
        if launch([sys.executable,'-m','venv','--without-pip',str(target)],root/'create_env.log',root,os.environ.copy(),60):raise RuntimeError('Fresh venv creation failed')
        env={**os.environ,'PIP_CONFIG_FILE':os.devnull,'PIP_DISABLE_PIP_VERSION_CHECK':'1','PYTHONIOENCODING':'utf-8',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        install_start=time.monotonic()
        command=[sys.executable,'-m','pip','--python',str(python),'install','--index-url','https://pypi.org/simple',
            '--only-binary=:all:','--no-cache-dir','--retries','0','--timeout','30','-r',str(args.lock.resolve())]
        code=launch(command,root/'install.log',root,env,min(600,max(30,left()-420)))
        install_seconds=time.monotonic()-install_start
        if code:raise RuntimeError('Pinned binary dependency installation failed; versions and source were not changed')
        cwd=root/'isolated_workdir';cwd.mkdir()
        for name in ('coupon','preference','sms','photo','bean','coupon_partial'):
            if left()<90:raise RuntimeError('No time remains for another isolated inference')
            bundle=args.bundles/name/'bundle';expected=args.bundles/name/'isolated_predictions/rows.csv'
            # The A54 output manifest records exactly which feature package was used.
            old_manifest=read(args.bundles/name/'isolated_predictions/prediction_manifest.json')
            destination=root/name/'rows.csv';destination.parent.mkdir()
            before=time.monotonic()
            code=launch([str(python),'-I',str(bundle/'predict.py'),'--features',old_manifest['input_source'],'--output',str(destination)],
                root/(name+'.log'),cwd,env,60)
            if code:raise RuntimeError('Fresh-environment bundle prediction failed: '+name)
            if destination.read_bytes()!=expected.read_bytes():raise ValueError('Fresh-environment prediction differs: '+name)
            results.append(dict(task=name,predictions_identical=True,rows=len(destination.read_text(encoding='utf-8').splitlines())-1,
                prediction_sha256=sha(destination),elapsed_seconds=time.monotonic()-before))
            save(root/'results.json',results)
        if sha(args.real_ledger)!=ledger_hash:raise ValueError('Unexpected real ledger modification during offline prediction')
        save(root/'completion.json',dict(status='passed',packages=len(results),fresh_dependency_install_tested=True,
            system_site_packages=False,original_environment_modified=False,actual_classifier_fits=0,real_api_requests=0,
            real_ledger_unchanged=True,dependency_install_seconds=install_seconds,elapsed_seconds=time.monotonic()-start))
        print(read(root/'completion.json'))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error_type=type(error).__name__,error=str(error),verified_packages=len(results),
            actual_classifier_fits=0,real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise
