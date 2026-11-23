"""Reuse one real CLI training-boundary probe for registered H1 and H2 inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import read, sha

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'L0':('local','ab-fixed'),'L1':('local','ab-v1'),'R0':('raw-history','ab-fixed'),
        'R1':('raw-history','ab-v1'),'E0':('checked-history','ab-fixed'),'E1':('checked-history','ab-v1')}
PROBE = '''
import json,sys
from pathlib import Path
from unittest.mock import patch
from scripts import a38_predict_guard
from superagent import ab_run
from tools.ab_registered_run import run
class Boundary(BaseException): pass
info=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
original=ab_run.freeze_partitions
def intercept(task,folder,seed,metric):
    if task.X_val is not None or task.y_val is not None: raise ValueError('Unexpected validation')
    if len(task.y_train)!=info['rows']: raise ValueError('Training rows changed')
    _,actual=original(task,folder,seed,metric)
    if actual!=info['splits']: raise ValueError('Actual worker identities differ')
    manifest=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['capabilities']!=info['capabilities']: raise ValueError('Actual menu differs')
    if list(folder.glob('trial_*.joblib')): raise ValueError('Model produced during entry probe')
    raise Boundary()
with patch.object(ab_run,'freeze_partitions',intercept):
    try: run(Path(info['manifest']),info['manifest_sha256'],info['command'])
    except Boundary: print('Reached registered partitions; stopped before Executor')
    else: raise ValueError('Did not reach the registered training boundary')
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target','h1','h2','base-assets','deep-assets','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    frozen=hashes(a.target)
    save(a.output/'protocol.json',dict(batch='A115',source_files=frozen,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A115_registered_H_entry_plan.md'),
         H1_receipt_sha256=sha(a.h1/'completion.json'),H2_receipt_sha256=sha(a.h2/'completion.json'),
         maximum_classifier_fits=0,real_API_requests=0,final_H_freeze=False))
    try:
        h1=read(a.h1/'results.json')['training_inputs']
        h2=read(a.h2/'results.json')
        entries={}
        for seed,row in h1.items():
            entries['H1_'+seed]=dict(folder=a.h1/seed,rows=row['train_rows'],train_sha=row['train_archive_sha256'],
                manifest_sha=row['inner_manifest_sha256'],splits=row['registered_splits'],capabilities=row['capabilities'])
        for seed,row in h2['registered_inputs'].items():
            entries['H2_'+seed]=dict(folder=a.h2/'training'/seed,rows=row['train_rows'],train_sha=row['train_archive_sha256'],
                manifest_sha=row['inner_manifest_sha256'],
                splits={s:{k:v for k,v in r.items() if k not in ('train_rows','internal_eval_rows')} for s,r in row['splits'].items()},
                capabilities=h2['menus'][seed]['capabilities'])
        env={**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(a.base_assets/'minilm'),
             'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(a.base_assets/'mobilenet_v3_small-047dcff4.pth'),
             'SUPERAGENT_DEEP_MINILM_WEIGHTS':str(a.deep_assets/'minilm_l12'),
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONIOENCODING':'utf-8'}
        env.pop('PYTHONPATH',None)
        templates={}
        for name,entry in entries.items():
            folder=entry['folder']
            if sha(folder/'train.zip')!=entry['train_sha'] or sha(folder/'inner_splits.json')!=entry['manifest_sha']:
                raise ValueError('Registered input changed: '+name)
            templates[name]={}
            for arm,(planner,scheduler) in ARMS.items():
                command=['run',str(folder/'train.zip'),'--run-dir',str(a.output/(name+'_'+arm)),
                    '--planner',planner,'--scheduler',scheduler,'--portfolio','ab-v1','--catalog-scope','full',
                    '--fit-limit','11','--seed','707','--time-limit','1200','--trial-timeout','300']
                if planner!='local':
                    command+=['--model','DeepSeek-V4.1-Flash','--api-key-file','LOCAL_KEY_PATH',
                              '--budget-ledger','EXISTING_SHARED_LEDGER_PATH','--task-budget-id',name[:2]+'_transfer',
                              '--llm-budget-rmb','.30','--history-policy','incremental-research','--proposal-contract','calibrated-v2']
                info={k:v for k,v in entry.items() if k!='folder'}
                info.update(manifest=str(folder/'inner_splits.json'),manifest_sha256=entry['manifest_sha'],command=command)
                config=a.output/(name+'_'+arm+'_probe.json')
                save(config,info)
                if planner=='local':
                    child=[sys.executable,'-c',PROBE,str(config)]
                    with (a.output/(name+'_'+arm+'.log')).open('x',encoding='utf-8') as log:
                        result=subprocess.run(child,cwd=a.target,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    if result.returncode: raise ValueError('Registered CLI probe failed: '+name+'/'+arm)
                else:
                    code='from superagent.cli import build_parser; from superagent.ab_run import validate_options; import json,sys; validate_options(build_parser().parse_args(json.loads(sys.argv[1])))'
                    result=subprocess.run([sys.executable,'-c',code,json.dumps(command)],cwd=a.target,env=env,capture_output=True,text=True,timeout=60)
                    if result.returncode: raise ValueError('Flash option validation failed: '+result.stderr[-1000:])
                templates[name][arm]=[sys.executable,'-m','tools.ab_registered_run','--split-manifest',str(folder/'inner_splits.json'),
                                      '--split-sha256',entry['manifest_sha'],'--',*command]
            print(dict(task=name,local_boundaries=2,paid_option_sets=4),flush=True)
        if frozen!=hashes(a.target): raise ValueError('Source changed')
        save(a.output/'results.json',dict(command_templates=templates,
             limitation='No H policy freeze or performance; paid handler/funding/network not invoked'))
        save(a.output/'completion.json',dict(status='entry_checked_pending_joint_freeze',local_CLI_boundaries=8,
             flash_option_sets=16,actual_classifier_fits=0,real_API_requests=0,source_unchanged=True,
             A_cumulative_fits_after=1505,H_outer_scores_computed=0,elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,
             real_API_requests=0,elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__': main()
