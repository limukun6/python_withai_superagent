"""Meter at most four supervised fits for real native L12 CLI and portable delivery coverage."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_b1_05_pair_pilot import pair_package

ROOT = Path(__file__).resolve().parents[1]
TRAIN = '''
import json,sys
from unittest.mock import patch
from scripts import a38_predict_guard
from superagent import ab_run
from superagent.cli import build_parser
candidate=sys.argv[1]
command=json.loads(sys.argv[2])
original=ab_run.registered_capabilities
def one(spec,rows,scope='full'):
    full=original(spec,rows,scope)
    if candidate not in full: raise ValueError('Native route unavailable')
    return {candidate:full[candidate]}
with patch.object(ab_run,'registered_capabilities',one):
    args=build_parser().parse_args(command)
    raise SystemExit(args.handler(args))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target','base-assets','deep-assets','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    frozen = hashes(a.target)
    save(a.output/'protocol.json',dict(batch='A113-delivery',source_files=frozen,
         runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A113_native_delivery_probe_plan.md'),
         engineering_single_candidate_projection=True, autonomous_performance_claim_allowed=False,
         maximum_classifier_fits=4,real_API_requests=0,starting_A_fits=1501))
    fits = a.output/'physical_fits.jsonl'
    env = {**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(a.base_assets/'minilm'),
           'SUPERAGENT_DEEP_MINILM_WEIGHTS':str(a.deep_assets/'minilm_l12'),
           'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','SUPERAGENT_OFFLINE':'1',
           'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(fits),
           'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a29_worker_observer'),str(a.target)]),
           'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    results = {}

    def trace():
        return [json.loads(line) for line in fits.read_text(encoding='utf-8').splitlines()] if fits.exists() else []

    def child(command,log,mode,timeout=600,allow_failure=False,extra_env=None):
        if time.monotonic()-start > 900:
            raise TimeoutError('Native delivery probe window elapsed')
        local = {**env,'AB_A29_RUN_NAME':mode,**(extra_env or {})}
        with log.open('x',encoding='utf-8') as stream:
            process = subprocess.run(command,cwd=a.target,env=local,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        if process.returncode and not allow_failure:
            raise RuntimeError('Native stage failed without retry: '+log.name)
        return process.returncode

    try:
        if read(a.deep_assets/'completion.json')['status'] != 'ready':
            raise ValueError('Deep assets not ready')
        rows,labels = [],[]
        for i in range(40):
            pair = (f'item {i} bright clear useful account about a river', f'item {i} vague dull brief account about a river')
            rows.append(pair if i%2 else pair[::-1])
            labels.append(1 if i%2 else 0)
        queries = [(f'new case {i} detailed story about a bridge',f'new case {i} short story about a bridge') for i in range(12)]
        pair_package(a.output/'anonymous.zip',rows,labels,'preference','train')
        pair_package(a.output/'queries.zip',queries,None,'preference','test')
        for mode,candidate in [('deep','minilm_l12_text_with_lexical_features'),('deep-edit','minilm_l12_with_edit_context')]:
            directory = a.output/mode
            directory.mkdir()
            command = ['run',str(a.output/'anonymous.zip'),'--run-dir',str(directory/'run'),
                       '--planner','local','--scheduler','ab-v1','--portfolio','ab-v1','--preference-catalog',mode,
                       '--catalog-scope','full','--fit-limit','2','--seed','707','--time-limit','600','--trial-timeout','180']
            child([sys.executable,'-c',TRAIN,candidate,json.dumps(command)],directory/'train.log',mode)
            outcome = read(directory/'run/run_outcome.json')
            actual = [r for r in trace() if r['run_name']==mode]
            if not outcome['requested_planner_completed'] or len(actual)!=2 or outcome['new_api_requests']:
                raise ValueError('Expected exactly two fits and zero requests: '+mode)
            count = outcome['classifier_fit_accounting']
            if count['attempted_classifier_fits']!=2 or count['completed_classifier_fits']!=2:
                raise ValueError('Physical/outcome fit counts differ')
            child([sys.executable,'-m','scripts.a38_predict_guard',str(a.output/'queries.zip'),
                   '--run-dir',str(directory/'run'),'--output',str(directory/'native.csv')],directory/'native.log',mode)
            child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(directory/'run'),
                   '--output',str(directory/'portable')],directory/'bundle.log',mode)
            bundle = directory/'portable/bundle'
            child([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(a.output/'queries.zip'),
                   '--output',str(directory/'isolated.csv')],directory/'isolated.log',mode)
            with (directory/'native.csv').open(encoding='utf-8',newline='') as stream:
                predicted = list(csv.DictReader(stream))
            if len(predicted)!=12 or (directory/'native.csv').read_bytes()!=(directory/'isolated.csv').read_bytes():
                raise ValueError('Native and portable output differ')
            rejection = child([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(a.output/'queries.zip'),
                               '--output',str(directory/'missing.csv')],directory/'missing.log',mode,
                              allow_failure=True,extra_env={'SUPERAGENT_DEEP_MINILM_WEIGHTS':''})
            if not rejection or (directory/'missing.csv').exists():
                raise ValueError('Missing deep assets did not reject before output')
            resources = read(bundle/'external_resources.json')
            if [r['name'] for r in resources]!=['minilm_l12']:
                raise ValueError('Wrong exported resource dependency')
            results[mode] = dict(candidate=candidate,fits=2,rows=12,native_portable_identical=True,
                                 missing_assets_rejected_before_output=True,external_resources=resources,
                                 outcome=outcome,model_sha256=read(directory/'run/final_selection.json')['model_sha256'])
            save(a.output/'checkpoint.json',results)
            print(dict(mode=mode,fits=2,rows=12,portable_identical=True),flush=True)
        if len(trace())!=4 or hashes(a.target)!=frozen:
            raise ValueError('Unexpected fit or source change')
        save(a.output/'results.json',results)
        save(a.output/'completion.json',dict(status='completed_engineering_probe',actual_classifier_fits=4,
             real_API_requests=0,A_cumulative_fits_after=1505,source_unchanged=True,
             H_or_official_validation_accessed=False,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(trace()),
             real_API_requests=0,A_cumulative_fits_after=1501+len(trace()),automatic_retry=False,
             elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    main()
