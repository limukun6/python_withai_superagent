"""A63 optional clean-environment check, reusing A57 venv without network or training."""
import argparse
from pathlib import Path
import subprocess
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('delivery','features','python','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    save(a.output/'protocol.json',dict(batch='A63-existing-clean-venv',runner_sha256=sha(Path(__file__)),
        python_executable=str(a.python),environment_reused=True,installation_performed=False,actual_classifier_fits=0,real_api_requests=0))
    try:
        for name in ('full','partial'):
            bundle=a.delivery/(name+'_package/bundle');out=a.output/(name+'.csv')
            with (a.output/(name+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run([str(a.python),'-I',str(bundle/'predict.py'),'--features',str(a.features),
                    '--output',str(out)],cwd=a.output,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            if result.returncode:raise ValueError('Clean environment prediction failed: '+name)
            if out.read_bytes()!=(a.delivery/(name+'_isolated.csv')).read_bytes():raise ValueError('Probability CSV differs: '+name)
            rows.append(dict(case=name,probability_csv_identical=True,project_source_from_bundle=True))
        save(a.output/'results.json',rows)
        save(a.output/'completion.json',dict(status='passed',cases=len(rows),actual_classifier_fits=0,real_api_requests=0,
            existing_independent_venv=True,new_installation=False,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_api_requests=0));raise
