"""Frozen SA25 local integration acceptance; existing public data, zero campus calls."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows
from scripts.run_sa21_context_live import clean_env
from scripts.sa8_real_delivery_cases import partition

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/SA25B_JOINT_CONTINUATION_PLAN_20261127_zh.md'
HOOK=r'''import json,os,sys,time,subprocess
from pathlib import Path
base=Path(os.environ['SA25_INTERRUPT_DIRECTORY'])
if 'unified_agent' in sys.orig_argv:
    (base/'controller.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid())),encoding='utf-8')
original_popen=subprocess.Popen
def observed_popen(*args,**kwargs):
    p=original_popen(*args,**kwargs)
    command=args[0] if args else kwargs.get('args',[])
    if isinstance(command,list):
        role='trainer_launch' if 'superagent' in command and 'unified' in command else 'worker_launch' if 'superagent.experiment_worker' in command else None
        if role:(base/(role+'.json')).write_text(json.dumps(dict(pid=p.pid,parent=os.getpid())),encoding='utf-8')
    return p
subprocess.Popen=observed_popen
if 'superagent' in sys.orig_argv and 'unified' in sys.orig_argv:
    (base/'trainer.json').write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid())),encoding='utf-8')
if 'superagent.experiment_worker' in sys.orig_argv:
    receipt=json.loads((base/'delivery/prediction_manifest.json').read_text(encoding='utf-8'))
    assert receipt['delivery_phase']=='first' and receipt['delivery_status']=='partial'
    marker=base/'worker.tmp'
    marker.write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid(),stage='before_worker_module_and_fit',argv=sys.orig_argv)),encoding='utf-8')
    marker.replace(base/'worker.json')
    time.sleep(90)
'''


def identity():
    paths=list((ROOT/'superagent').glob('*.py'))+[ROOT/'unified_agent.py',Path(__file__),ROOT/'tools/import_semantic_artifact.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def npy(value):
    stream=io.BytesIO();np.save(stream,value,allow_pickle=False);return stream.getvalue()


def execute(command,log,env=None,timeout=210):
    with log.open('w',encoding='utf-8') as output:
        return subprocess.run(command,cwd=ROOT,env=env or clean_env(),stdin=subprocess.DEVNULL,
            stdout=output,stderr=subprocess.STDOUT,timeout=timeout).returncode


def prepare(a):
    from sklearn.datasets import load_iris
    from tools.import_semantic_artifact import migrate
    checks=read(a.checks/'completion.json')
    if checks['status']!='passed' or not checks['source_unchanged']:raise ValueError('Joint checks must pass')
    if any(sha(ROOT/n)!=v for n,v in read(a.checks/'protocol.json')['source_sha256'].items()):raise ValueError('Checked source changed')
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    cases=a.private_output/'cases';cases.mkdir();data=load_iris();fit,held=partition(data.data,data.target,96,24)
    base=dict(schema_version=1,modality='tabular',objective='classification',target='kind',metric='balanced_accuracy',
        restrictions=dict(linear_only=True,no_pretrained_assets=True,no_external_data=True))
    train=dict(base,splits=dict(train=dict(format='npy',path='observations.npy',target_path='classes.npy')))
    query=dict(base,splits=dict(test=dict(format='npy',path='new.npy')))
    for name,members in [('ordinary.zip',{'manifest.json':json.dumps(train),'observations.npy':npy(data.data[fit]),'classes.npy':npy(data.target[fit]),
            'README.md':'Only linear classifiers are permitted. No external data or pretrained encoders. Evaluate balanced accuracy.'}),
            ('ordinary.query.zip',{'manifest.json':json.dumps(query),'new.npy':npy(data.data[held])})]:
        with zipfile.ZipFile(cases/name,'w') as z:
            for member,body in members.items():z.writestr(member,body)
    sources=a.source/'research/sa23_delivery_prepared_02/cases'
    row=read(sources/'registry.json')[0]
    for key,name in [('package','custom.zip'),('reference','custom.query.zip'),('query','custom.late.zip')]:
        original=sources/row[key]
        if sha(original)!=row[key+'_sha256']:raise ValueError('Original SA23 input changed')
        shutil.copyfile(original,cases/name)
    artifact=a.private_output/'custom.semantic_artifact.json'
    original=a.source/'runs/sa21_context_live_01/C1/run/agent/semantic_contract.json'
    metadata=migrate(cases/'custom.zip',cases/'custom.query.zip',original,a.ledger,artifact)
    save(a.output/'custom.migration.json',metadata)
    old_adapter=a.source/'research/sa23_delivery_live_01/V1/prediction_adapter.json'
    shutil.copyfile(old_adapter,a.private_output/'original_prediction_adapter.json')
    save(a.output/'protocol.json',dict(batch='SA25B-local-joint-delivery',A='7e9d142',B='4d4222a',source_sha256=identity(),plan_sha256=sha(PLAN),
        inputs={p.name:sha(p) for p in cases.iterdir()},cases_root=str(cases.resolve()),artifact=str(artifact.resolve()),artifact_sha256=sha(artifact),
        original_adapter_sha256=sha(a.private_output/'original_prediction_adapter.json'),private_prepared=str(a.private_output.resolve()),
        checks_completion_sha256=sha(a.checks/'completion.json'),per_complete_path_fit_limit=2,maximum_total_classifier_fits=4,
        custom_expected_ids=read(a.source/'research/sa21_context_prepared_01/cases/C1.oracle.json')['prediction_ids'],
        maximum_new_campus_API=0,order=['ordinary','custom','late_replay','windows_interruption'],
        labels_access='training labels only; query answers not exported or scored',
        interpretation_sources=dict(ordinary='explicit manifest',custom='explicit migration of original real SA21 response',
            late_replay='explicit revalidation of original real SA23 adapter; not fresh LLM recovery',
            windows_interruption='same explicit ordinary package; controlled process failure before first fit'),
        source_note='Iris/Breast Cancer previously used public development data; compatibility, not independent generality or quality'))
    save(a.output/'completion.json',dict(status='prepared',actual_classifier_fits=0,campus_API_requests=0))
    print(dict(status='prepared',cases=['ordinary','custom','late_replay','windows_interruption']))


def alive(pid):
    import ctypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.WaitForSingleObject.restype=ctypes.c_uint32
    k.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=k.OpenProcess(0x100000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return False
        raise OSError('Could not inspect owned process '+str(pid))
    try:return k.WaitForSingleObject(handle,0)==258
    finally:k.CloseHandle(handle)


def wait_for(predicate,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.1)
    raise TimeoutError('Expected integration checkpoint was not reached')


def command(action,package,run,output,*,query=None,artifact=None):
    args=[sys.executable,'-m','unified_agent',action,str(package),'--run-dir',str(run),'--output',str(output)]
    if action=='solve':args+=['--prediction-input',str(query),'--fit-limit','2','--time-limit','150']
    if artifact:args+=['--semantic-artifact-from',str(artifact)]
    return args


def complete_path(name,cases,private,public,artifact=None,expected_ids=None):
    private.mkdir();public.mkdir();run=private/'run';output=private/'delivery/predictions.csv'
    args=command('solve',cases/(name+'.zip'),run,output,query=cases/(name+'.query.zip'),artifact=artifact)
    save(public/'invocation.json',dict(command=args,stdin='closed',campus_client_configured=False))
    if execute(args,private/'solve.log'):raise RuntimeError(name+' solve failed')
    model=run/'agent';selection=read(model/'final_selection.json');spec=read(model/'task_spec.json')
    current=read(output.parent/'prediction_manifest.json');first=read(run/'first_delivery/prediction_manifest.json')
    if current['delivery_status']!='completed' or current['delivery_phase']!='selected' or not selection['refit']:raise ValueError('No full selected delivery')
    if first['delivery_status']!='partial' or first['delivery_phase']!='first':raise ValueError('No disclosed early prediction')
    replay=private/'replay/predictions.csv'
    if execute(command('predict',cases/(name+'.query.zip'),run,replay),private/'replay.log',timeout=60):raise RuntimeError('New-process prediction failed')
    if sha(output)!=sha(replay):raise ValueError('New-process predictions differ')
    save(public/'final_selection.json',selection);save(public/'prediction_manifest.json',current);save(public/'first_delivery_manifest.json',first)
    fits=read(model/'classifier_fits.json');save(public/'classifier_fits.json',fits)
    if fits['attempted_classifier_fits']!=2:raise ValueError('Expected one search fit and full refit')
    if name=='ordinary':
        from superagent.task_constraints import permitted
        from superagent.ab_catalog import capabilities_for
        from superagent.open_methods import capability
        if set(spec['method_restrictions'])!={'linear_only','no_external_data','no_pretrained_assets'}:raise ValueError('Lost task restrictions')
        history=read(model/'history.json')
        caps=capabilities_for(read(model/'task_contract.json'),train_rows=96)
        caps.update({n:capability(entry) for n,entry in read(model/'candidate_registry.json')['entries'].items()})
        if any(not permitted(caps[h['candidate']],spec['method_restrictions']) for h in history):raise ValueError('Forbidden executed method')
        rows=list(csv.DictReader(output.open(encoding='utf-8',newline='')))
        if len(rows)!=24 or [r['row_id'] for r in rows]!=list(map(str,range(24))):raise ValueError('Ordinary rows/index mismatch')
    else:
        with output.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
        if len(rows)!=40 or list(rows[0])!=['Case','Decision','BenignScore']:raise ValueError('Custom CSV shape mismatch')
        if not output.read_bytes().startswith(b'\xef\xbb\xbf'):raise ValueError('Missing custom BOM')
        if [r['Case'] for r in rows]!=expected_ids:raise ValueError('Custom IDs not preserved')
        if not all(0<=float(r['BenignScore'])<=1 and r['Decision'] in ('B','M') for r in rows):raise ValueError('Invalid class/probability')
    return dict(id=name,status='passed',actual_classifier_fits=2,rows=len(rows),early_partial_and_final_completed=True,
        cross_process_identical=True,prediction_sha256=sha(output),model_sha256=selection['model_sha256'],campus_API_requests=0)


def late_replay(cases,custom,private,public,prepared):
    from superagent.prediction_adapter import apply_proposal,preflight,write_adapter,load_adapter
    from superagent.data import load_prediction_features
    from superagent.predict import create_predictions,PredictionAdaptationError
    from scripts.sa7_multiturn_cases import array_fingerprint
    private.mkdir();public.mkdir();run=custom/'run';model=run/'agent';spec=read(model/'task_spec.json');selection=read(model/'final_selection.json')
    frozen={str(p):sha(p) for p in [model/selection['model_file'],model/'task_spec.json',model/'final_selection.json',model/'classifier_fits.json']}
    query=cases/'custom.late.zip';old=read(prepared/'original_prediction_adapter.json');raw=old['proposal'];refs=set(old['known_document_refs'])
    try:create_predictions(query,model,private/'should_not_exist/p.csv')
    except PredictionAdaptationError:pass
    else:raise ValueError('Late input did not actually require adaptation')
    adapted=apply_proposal(spec,raw,refs);checks=preflight(query,adapted)
    path=run/'delivery_recovery/sa25_revalidated/prediction_adapter.json';path.parent.mkdir(parents=True)
    write_adapter(path,query,model,raw,refs,checks)
    original=load_prediction_features(cases/'custom.query.zip',spec['model_kind'],spec)
    transformed=load_prediction_features(query,spec['model_kind'],load_adapter(path,query,model,spec,selection))
    if array_fingerprint(original)!=array_fingerprint(transformed):raise ValueError('Late feature meanings differ')
    hashes=[]
    for tag in ('first','second'):
        output=private/tag/'predictions.csv'
        if execute(command('predict',query,run,output),private/(tag+'.log'),timeout=60):raise RuntimeError('Staged cached late replay failed')
        if read(output.parent/'prediction_manifest.json')['delivery_phase']!='selected':raise ValueError('Late input silently downgraded')
        hashes.append(sha(output))
    if hashes!=[sha(custom/'delivery/predictions.csv')]*2 or frozen!={n:sha(n) for n in frozen}:raise ValueError('Late export or frozen model changed')
    save(public/'prediction_adapter.json',read(path));save(public/'frozen_files.json',frozen)
    receipts=list((path.parent/'deliveries').glob('*.json'))
    if len(receipts)!=2:raise ValueError('Repeated adapted publication receipts collide')
    return dict(id='late_replay',status='passed',actual_classifier_fits=0,rows=len(original),cached_replay_API=0,
        frozen_files_unchanged=True,features_equal=True,two_processes_equal=True,unique_receipts=len(receipts),
        adapter_source='explicitly revalidated original SA23 real reply; no new LLM request',campus_API_requests=0)


def interruption(cases,private,public):
    if os.name!='nt':raise ValueError('This registered acceptance requires Windows')
    private.mkdir();public.mkdir();hook=private/'hook';hook.mkdir();(hook/'sitecustomize.py').write_text(HOOK,encoding='utf-8')
    run=private/'run';output=private/'delivery/predictions.csv';env=clean_env()
    env.update(PYTHONPATH=os.pathsep.join((str(hook),str(ROOT))),SA25_INTERRUPT_DIRECTORY=str(private))
    args=command('solve',cases/'ordinary.zip',run,output,query=cases/'ordinary.query.zip')
    save(public/'invocation.json',dict(command=args,hook_sha256=sha(hook/'sitecustomize.py'),controlled_fault='controller killed after first delivery and before worker module/fit',
        hook_changes_model_or_selection=False))
    trainer=worker=None;owned=[];start=time.monotonic()
    with (private/'solve.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            wait_for(lambda:all((private/(n+'.json')).is_file() for n in ('worker','trainer','controller','trainer_launch','worker_launch')) or process.poll() is not None,80)
            trainer=read(private/'trainer.json');worker=read(private/'worker.json')
            controller=read(private/'controller.json');trainer_launch=read(private/'trainer_launch.json');worker_launch=read(private/'worker_launch.json')
            links=(controller['parent']==process.pid and trainer_launch['parent']==controller['pid'] and
                trainer['parent']==trainer_launch['pid'] and worker_launch['parent']==trainer['pid'] and worker['parent']==worker_launch['pid'])
            if not links:raise ValueError('Not our created Windows launcher/process tree')
            owned=[process.pid,controller['pid'],trainer_launch['pid'],trainer['pid'],worker_launch['pid'],worker['pid']]
            if worker['stage']!='before_worker_module_and_fit' or not alive(worker['pid']):raise ValueError('No paused pre-fit worker')
            receipt=read(output.parent/'prediction_manifest.json');early=sha(output);first_seconds=time.monotonic()-start
            if receipt['delivery_status']!='partial' or receipt['delivery_phase']!='first':raise ValueError('Early output unavailable')
            subprocess.run(['taskkill','/PID',str(controller['pid']),'/F'],capture_output=True,timeout=10,check=True)
            process.wait(timeout=15)
            wait_for(lambda:not any(alive(pid) for pid in owned),15)
            if sha(output)!=early:raise ValueError('Controller loss changed early output')
        finally:
            if process.poll() is None:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=15);process.wait(timeout=15)
            for row in (trainer,worker):
                if row and alive(row['pid']):subprocess.run(['taskkill','/PID',str(row['pid']),'/T','/F'],capture_output=True,timeout=15)
    replay=private/'replay/predictions.csv'
    if execute(command('predict',cases/'ordinary.query.zip',run/'first_delivery',replay),private/'replay.log',timeout=60):raise RuntimeError('Interrupted early model failed reload')
    if sha(replay)!=early:raise ValueError('Interrupted snapshot replay differs')
    save(public/'prediction_manifest.json',receipt);save(public/'owned_processes.json',dict(controller_launcher=process.pid,controller=controller,
        trainer_launch=trainer_launch,trainer=trainer,worker_launch=worker_launch,worker=worker,owned_pids=owned,all_exited=True))
    return dict(id='windows_interruption',status='passed',actual_classifier_fits=0,rows=24,early_delivery_seconds=first_seconds,
        paused_worker_before_any_classifier_fit=True,created_worker_processes=1,controller_trainer_worker_exited=True,
        early_prediction_unchanged=True,new_process_early_replay_identical=True,campus_API_requests=0,
        scope='controlled pre-fit controller loss; does not test power loss or interruption during estimator.fit')


def run(a):
    from superagent.task_budget import TaskBudgetLedger
    protocol=read(a.prepared/'protocol.json');prepared=Path(protocol['private_prepared']);cases=Path(protocol['cases_root'])
    if protocol['source_sha256']!=identity() or protocol['plan_sha256']!=sha(PLAN):raise ValueError('Frozen source/plan changed')
    if any(sha(cases/n)!=v for n,v in protocol['inputs'].items()):raise ValueError('Frozen inputs changed')
    if sha(protocol['artifact'])!=protocol['artifact_sha256'] or sha(prepared/'original_prediction_adapter.json')!=protocol['original_adapter_sha256']:raise ValueError('Historical interpretation changed')
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',protocol);ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger);rows=[];clock=time.monotonic()
    save(a.output/'historical_rows_before.json',old)
    for name in protocol['order']:
        start=time.monotonic()
        try:
            if name in ('ordinary','custom'):result=complete_path(name,cases,a.private_output/name,a.output/name,Path(protocol['artifact']) if name=='custom' else None,protocol['custom_expected_ids'])
            elif name=='late_replay':result=late_replay(cases,a.private_output/'custom',a.private_output/name,a.output/name,prepared)
            else:result=interruption(cases,a.private_output/name,a.output/name)
        except Exception as issue:
            result=dict(id=name,status='failed',error=dict(type=type(issue).__name__,message=str(issue)[:800]),campus_API_requests=0)
            fit=a.private_output/name/'run/agent/classifier_fits.json'
            result['actual_classifier_fits']=read(fit)['attempted_classifier_fits'] if fit.is_file() else 0
        result['elapsed_seconds']=time.monotonic()-start;rows.append(result)
        folder=a.output/name;folder.mkdir(exist_ok=True);save(folder/'completion.json',result);save(a.output/'partial_results.json',rows)
        print(json.dumps(result),flush=True)
        if result['status']!='passed':break
    current=ledger_rows(ledger);unchanged=old==current
    result=dict(status='passed' if len(rows)==4 and all(r['status']=='passed' for r in rows) and unchanged else 'failed',cases=rows,
        actual_classifier_fits=sum(r['actual_classifier_fits'] for r in rows),new_campus_API_requests=0,new_course_cost_rmb=0,
        elapsed_seconds=time.monotonic()-clock,ledger_before=before,ledger_after=ledger.project_summary(),entire_canonical_ledger_unchanged=unchanged,
        source_unchanged=protocol['source_sha256']==identity(),evidence_boundary=protocol['source_note'])
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare')
    for key in ('output','private-output','checks','source','ledger'):q.add_argument('--'+key,type=Path,required=True)
    q=sub.add_parser('run')
    for key in ('prepared','output','private-output','ledger'):q.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)


if __name__=='__main__':main()
