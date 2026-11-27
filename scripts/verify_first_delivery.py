"""Interrupt processes created by this harness and exercise ordinary solve/predict recovery on POSIX."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from scripts.run_planner_self_review import packet, source_identity
from scripts.unified_released_study import ROOT, read, save, sha

# This hook never changes a prediction or selection. A real fit worker pauses at a
# reproducible interrupt point. It must already see the requested early CSV at startup.
HOOK = r"""import fcntl, json, os, sys, time
from pathlib import Path
p=Path(os.environ['FIRST_DELIVERY_TEST_DIRECTORY'])
if 'superagent' in sys.orig_argv and 'unified' in sys.orig_argv:
    trainer_lock=(p/'trainer.lock').open('a')
    fcntl.flock(trainer_lock,fcntl.LOCK_EX)
    (p/'trainer.json').write_text(json.dumps({'pid':os.getpid(),'parent':os.getppid()}))
if 'superagent.experiment_worker' in sys.orig_argv:
    dest=next((Path(v).name for v in sys.orig_argv if v.endswith('.joblib') and Path(v).name.startswith('trial_')),None)
    attempted=p/'delivery'/'prediction_attempt.json'
    assert attempted.is_file() and json.loads(attempted.read_text())['delivery_phase']=='first'
    if dest==os.environ.get('FIRST_DELIVERY_PAUSE_AT'):
        worker_lock=(p/'worker.lock').open('a')
        fcntl.flock(worker_lock,fcntl.LOCK_EX)
        marker=p/'worker-started.tmp'
        marker.write_text(json.dumps({'pid':os.getpid(),'trainer':os.getppid(),'group':os.getpgrp()}))
        marker.replace(p/'worker-started.json')
        time.sleep(90)
"""


def invoke(args, log, env, success=True):
    with log.open('x') as f:
        run=subprocess.run([sys.executable,'-m','unified_agent',*map(str,args)],cwd=ROOT,env=env,
                           stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=100)
    if (run.returncode==0)!=success:
        raise RuntimeError('Unexpected entry status. See '+str(log))


def rows(path,count):
    with path.open(encoding='utf-8',newline='') as f:
        records=list(csv.DictReader(f))
    assert [r['row_id'] for r in records]==list(map(str,range(count)))
    for r in records:
        assert int(r['prediction']) in (0,1,2)
        assert all(0<=float(r[f'score_{i}'])<=1 for i in range(3))
        assert abs(sum(float(r[f'score_{i}']) for i in range(3))-1)<1e-6
    return records


def wait_for(predicate,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.05)
    raise TimeoutError('Controlled run did not reach its expected checkpoint')


def interrupt(root,tag,environment,train,query,hook):
    p=root/tag;p.mkdir()
    out=p/'delivery'/'p.csv';run=p/'run'
    which='trial_1.joblib' if tag=='controller_loss' else 'trial_2.joblib'
    options=dict(environment,PYTHONPATH=os.pathsep.join((str(hook),str(ROOT))),FIRST_DELIVERY_TEST_DIRECTORY=str(p),
                 FIRST_DELIVERY_PAUSE_AT=which)
    start=time.monotonic()
    with (p/'solve.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-m','unified_agent','solve',str(train),'--prediction-input',str(query),
                '--run-dir',str(run),'--output',str(out),'--fit-limit','5','--time-limit','65'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=options,start_new_session=True)
        try:
            wait_for(lambda:(p/'worker-started.json').is_file() or process.poll() is not None,65)
            worker=read(p/'worker-started.json'); trainer=read(p/'trainer.json')
            assert trainer['parent']==process.pid and worker['trainer']==trainer['pid']==worker['group']
            first=sha(out); first_seconds=time.monotonic()-start
            assert read(out.parent/'prediction_attempt.json')['delivery_status']=='partial'
            assert read(run/'first_delivery'/'prediction_manifest.json')['prediction_sha256']==first
            assert not (run/'agent'/'run_outcome.json').exists()
            rows(out,57)
            import fcntl
            def exited(name):
                with (p/(name+'.lock')).open('a') as handle:
                    try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: return False
                    return True
            assert not exited('trainer') and not exited('worker')
            if tag=='controller_loss':
                process.kill()
            else:
                os.killpg(trainer['pid'],signal.SIGKILL)
            code=process.wait(timeout=30)
            wait_for(lambda:exited('trainer') and exited('worker'),12)
        finally:
            if process.poll() is None: process.kill();process.wait()
    assert code==(-signal.SIGKILL if tag=='controller_loss' else 0)
    dest=p/'reload'/'p.csv'
    invoke(['predict',query,'--run-dir',run/'first_delivery','--output',dest],p/'reload.log',environment)
    assert sha(dest)==first
    settled=read(out.parent/'prediction_manifest.json')
    rows(out,57)
    if tag=='controller_loss':
        assert first==sha(out) and settled['delivery_phase']=='first'
    else:
        assert settled['delivery_phase']=='selected' and settled['source_trial']==1
        later=p/'selected-reload'/'p.csv'
        invoke(['predict',query,'--run-dir',run,'--output',later],p/'selected-reload.log',environment)
        assert sha(later)==sha(out)
    fits=[read(f) for f in (run/'agent').glob('*.fits.json')]
    return dict(first_available_by_seconds=round(first_seconds,3),interrupted_stage=which,supervisor_exit=code,
                rows=57,score_columns=3,first_sha256=first,first_reloaded=True,trainer_and_worker_stopped=True,
                remaining_output_sha256=sha(out),delivered_family=settled['selected_experiment'],
                remaining_delivery_phase=settled['delivery_phase'],observed_fit_starts=sum(f['attempted_classifier_fits'] for f in fits))


def verify(root):
    if os.name!='posix': raise RuntimeError('The process group termination probe requires POSIX')
    root.mkdir(parents=True)
    rng=np.random.default_rng(271127)
    data=rng.normal(size=(270,5));labels=np.argmax(data[:,:3],axis=1)
    train=root/'train.zip';query=root/'features.zip'
    packet(train,'tabular','roc_auc',data,labels);packet(query,'tabular','roc_auc',rng.normal(size=(57,5)))
    guard=root/'guard';guard.mkdir();(guard/'sitecustomize.py').write_text(HOOK)
    env=dict(os.environ,SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for key in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID',
                'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX','SUPERAGENT_LESSON_STORE'):
        env.pop(key,None)
    save(root/'protocol.json',dict(base_plan=sha(ROOT/'research/reliable_delivery_1127/protocol.json'),
         runtime=source_identity(ROOT),train=sha(train),features=sha(query),runner=sha(Path(__file__)),api_requests=0))
    report={tag:interrupt(root,tag,env,train,query,guard) for tag in ('controller_loss','trainer_loss')}
    # Corrupt only a disposable copy of the selected pointer, then exercise the ordinary prediction entry.
    recovery=root/'fault-test';(recovery/'agent').mkdir(parents=True)
    base=root/'trainer_loss'/'run'
    shutil.copytree(base/'first_delivery',recovery/'first_delivery')
    shutil.copyfile(base/'agent/task_spec.json',recovery/'agent/task_spec.json')
    selected=read(base/'agent/final_selection.json');selected['model_file']='missing-fault-fixture.joblib'
    save(recovery/'agent/final_selection.json',selected)
    dest=root/'fault-output'/'p.csv'
    invoke(['predict',query,'--run-dir',recovery,'--output',dest],root/'fault.log',env)
    assert read(dest.parent/'prediction_manifest.json')['delivery_phase']=='recovered_first'
    assert sha(dest)==report['trainer_loss']['first_sha256']
    invalid=root/'broken.zip';invalid.write_bytes(b'not a feature archive')
    invoke(['predict',invalid,'--run-dir',recovery,'--output',dest],root/'invalid.log',env,False)
    assert read(dest.parent/'prediction_attempt.json')['generated_by_this_attempt'] is False
    assert sha(dest)==report['trainer_loss']['first_sha256']
    report.update(damaged_final_recovered=True,invalid_new_input_not_claimed=True,additional_recovery_fits=0,api_requests=0)
    save(root/'results.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    verify(parser.parse_args().output.resolve())
