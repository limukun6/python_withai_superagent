"""Controlled disk-state recovery tests using already-trained A138 components; no new fits/API."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
OLD=BASE/'runs/a137_numeric_checkout'
ORIGINAL=BASE/'runs/a138_goal_execute/accuracy/run'
TRAIN=BASE/'runs/a138_goal_prepared/accuracy.zip'
QUERY=BASE/'runs/a138_goal_prepared/accuracy_query.zip'


def file_hashes(root):return {p.name:sha(p) for p in root.iterdir() if p.is_file()}


def snapshot(folder,completed,full=False):
    folder.mkdir();names=['run_manifest.json']
    for split in ('primary','confirmation_1','confirmation_2'):names.extend([split+'.joblib',split+'_split.json'])
    names.extend('state_'+str(i)+'.json' for i in range(1,completed+2))
    for i in range(1,completed+1):names.extend('trial_'+str(i)+suffix for suffix in ('.joblib','.fits.json','.admission.json'))
    if full:
        names.extend(['final_selection.json','selected_model.joblib','task_spec.json','refit_component_0.fits.json','refit_component_0.admission.json'])
    for name in names:shutil.copy2(ORIGINAL/name,folder/name)
    old=read(ORIGINAL/'classifier_fits.json');workers=old['workers'] if full else old['workers'][:completed]
    save(folder/'classifier_fits.json',dict(attempted_classifier_fits=sum(w['attempted_classifier_fits'] for w in workers),
         completed_classifier_fits=sum(w['completed_classifier_fits'] for w in workers),unknown_failed_attempts=0,workers=workers))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic();source=hashes(ROOT)
    original_before=file_hashes(ORIGINAL)
    ledger=Path(read(ROOT/'research/a140_flash_projection/paid/current_ledger_owner.json')['ledger_path']);ledger_before=sha(ledger)
    save(out/'protocol.json',dict(batch='A141',source_files=source,runner_sha256=sha(Path(__file__)),
         recovery_tool_sha256=sha(ROOT/'tools/recover_run_checkpoint.py'),
         plan_sha256=sha(ROOT/'docs/batches/A141_interruption_recovery_plan_20261124_zh.md'),
         original_run=str(ORIGINAL),original_files=original_before,training_source_root=str(OLD),
         fit_cap=0,API_cap=0,starting_A_fits=1614,scope='Controlled disk snapshots, not real power interruption',
         ledger_sha256_before=ledger_before))
    env={**os.environ,'SUPERAGENT_OFFLINE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PYTHONPATH':str(OLD),'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    def child(command,log,expected=0):
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=OLD,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        if (expected==0 and result.returncode) or (expected!=0 and not result.returncode):raise RuntimeError('Unexpected child result: '+str(log))
        return result.returncode
    results=[]
    try:
        if hashes(OLD)!={'superagent/'+k:v for k,v in read(ORIGINAL/'run_manifest.json')['source_files'].items()}:
            raise ValueError('Pinned old source differs before recovery test')
        cases=[('after_trial',1,False),('before_refit',2,False),('unknown_worker',2,False),
               ('torn_state',2,False),('corrupt_model',2,False),('wrong_task',2,False),
               ('full_export',2,True),('repeat_restore',1,False),('wrong_source',1,False)]
        for name,n,full in cases:
            folder=out/name;folder.mkdir();disk=folder/'interrupted';snapshot(disk,n,full)
            if name=='unknown_worker':
                save(disk/'trial_3.admission.json',dict(maximum_classifier_fits=1,ledger_path='interrupted_worker_unknown'))
            if name=='torn_state':(disk/'state_3.json').write_bytes(b'{"history":[')
            if name=='corrupt_model':(disk/'trial_2.joblib').write_bytes(b'interrupted write: not a valid model')
            if name=='full_export':(disk/'predictions.csv').write_text('row_id,prediction\n0,',encoding='utf-8')
            wrong_archive=folder/'wrong.zip'
            if name=='wrong_task':wrong_archive.write_bytes(b'wrong registered training packet')
            before=file_hashes(disk);recovered=folder/'recovered';reject=name in ('corrupt_model','wrong_task','wrong_source')
            command=[sys.executable,'-I',str(ROOT/'tools/recover_run_checkpoint.py'),'--run-dir',str(disk),
                 '--output',str(recovered),'--source-root',str(ROOT if name=='wrong_source' else OLD),
                 '--task-archive',str(wrong_archive if name=='wrong_task' else TRAIN)]
            child(command,folder/'recover.log',2 if reject else 0)
            receipt=read(recovered/'recovery_completion.json')
            if reject:
                if receipt['status']!='rejected' or (recovered/'final_selection.json').exists():raise ValueError('Invalid recovery accepted')
                result=dict(case=name,status='expected_rejection',error=receipt['error'])
            else:
                if receipt['status']!='recovered' or receipt['training_resume_supported']:raise ValueError('Recovery claimed unsupported training resume')
                selection=read(recovered/'final_selection.json')
                if full:
                    if not selection['refit'] or receipt['mode']!='existing_frozen_selection':raise ValueError('Original full-model status changed')
                elif selection['refit'] or selection['delivery_status']!='partial':raise ValueError('Checkpoint was relabelled full')
                if name=='torn_state' and receipt['ignored_incomplete_states']!=['state_3.json']:raise ValueError('Torn snapshot was not reported')
                resources=read(recovered/'inherited_resources.json')
                if name=='unknown_worker' and not resources['additional_past_fit_cost_unknown']:raise ValueError('In-flight cost was silently zeroed')
                native=folder/'native.csv'
                child([sys.executable,'-m','scripts.a38_predict_guard',str(QUERY),'--run-dir',str(recovered),'--output',str(native)],folder/'native.log')
                child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(recovered),'--output',str(folder/'portable')],folder/'bundle.log')
                child([sys.executable,'-I',str(folder/'portable/bundle/predict.py'),'--features',str(QUERY),'--output',str(folder/'isolated.csv')],folder/'isolated.log')
                if native.read_bytes()!=(folder/'isolated.csv').read_bytes():raise ValueError('Recovered independent predictions differ')
                with native.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
                if [int(r['row_id']) for r in rows]!=list(range(32)):raise ValueError('Recovered row order differs')
                if full and native.read_bytes()!=(ORIGINAL.parent/'native.csv').read_bytes():raise ValueError('Full re-export changed predictions')
                if name=='repeat_restore':
                    second=folder/'recovered_again';command[command.index('--output')+1]=str(second)
                    child(command,folder/'recover_again.log')
                    child([sys.executable,'-m','scripts.a38_predict_guard',str(QUERY),'--run-dir',str(second),
                          '--output',str(folder/'again.csv')],folder/'again.log')
                    if native.read_bytes()!=(folder/'again.csv').read_bytes():raise ValueError('Repeated recovery differs')
                result=dict(case=name,status='recovered',mode=receipt['mode'],delivery_status=selection.get('delivery_status','complete'),
                            rows=32,isolated_identical=True,inherited_resources=resources,
                            ignored_incomplete_states=receipt.get('ignored_incomplete_states',[]))
            if file_hashes(disk)!=before:raise ValueError('Controlled source disk modified')
            results.append(result);save(out/'checkpoint.json',results);print(json.dumps(dict(case=name,status=result['status'])),flush=True)
        if file_hashes(ORIGINAL)!=original_before or hashes(ROOT)!=source or sha(ledger)!=ledger_before:
            raise ValueError('Original model/source/API ledger changed')
        save(out/'results.json',dict(cases=results,passed=len(results),total=9,source_run_unchanged=True,
             API_ledger_unchanged=True,actual_power_loss_tested=False,training_resume_supported=False))
        save(out/'completion.json',dict(status='passed',cases=9,actual_classifier_fits=0,real_API_requests=0,
             A_cumulative_fits_after=1614,elapsed_seconds=time.monotonic()-began,source_unchanged=True,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
    except Exception as e:
        save(out/'results.json',dict(cases=results))
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':main()
