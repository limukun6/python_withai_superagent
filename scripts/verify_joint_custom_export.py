"""Interrupt our own solve process after a verified custom export, then reload it independently."""
import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from scripts.compare_joint_feedback import CONFIG
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_first_delivery import HOOK, invoke, wait_for
from superagent.events import RunLogger
from superagent.semantic_contract import resolve_training_task
from tests.test_output_recipe import package, recipe
from tests.test_proactive_probe import output_proposal
from tests.test_semantic_contract import Client


def verify(root):
    root.mkdir(parents=True)
    fixture=package(root/'source.zip',probability=True)
    # Give the trainer enough rows to start its first genuine worker. Only the two unlabeled IDs are exported.
    train=root/'package.zip'
    with zipfile.ZipFile(fixture) as old, zipfile.ZipFile(train,'w') as archive:
        for name in old.namelist():
            blob=old.read(name)
            if name=='ledger.data':
                blob='signal,decision_code,cohort\n'+''.join(
                    f'{i%11},{"accept" if i%3 else "decline"},learn\n' for i in range(150))
            elif name=='02_dictionary.md':
                blob='Train on signal. Output a semicolon CSV with UTF-8 BOM. Ticket must preserve event_id including leading zeros. Chance is the probability of accept. Use sample.csv row order.'
            archive.writestr(name,blob)
    reply=output_proposal()
    reply['plan']['output']=recipe(probability=True,template=True)
    cached=root/'contract'
    cached.mkdir()
    client=Client([reply])
    resolve_training_task(train,client,RunLogger(cached),mode='proactive',prediction_package=train)
    require(client.request_count==1,'Contract fixture did not compile')
    hook=root/'guard'
    hook.mkdir()
    (hook/'sitecustomize.py').write_text(HOOK)
    settings=dict(os.environ,SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name in CONFIG:settings.pop(name,None)
    environment=dict(settings,PYTHONPATH=os.pathsep.join(map(str,(hook,ROOT))),FIRST_DELIVERY_TEST_DIRECTORY=str(root),
                     FIRST_DELIVERY_PAUSE_AT='trial_1.joblib')
    run=root/'run'
    output=root/'delivery/pred.csv'
    start=time.monotonic()
    with (root/'solve.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-m','unified_agent','solve',str(train),'--prediction-input',str(train),
                '--semantic-artifact-from',str(cached/'semantic_artifact.json'),'--run-dir',str(run),'--output',str(output),
                '--fit-limit','4','--time-limit','75'],cwd=ROOT,start_new_session=True,env=environment,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            wait_for(lambda:(root/'worker-started.json').is_file() or process.poll() is not None,70)
            require((root/'worker-started.json').is_file(),'Worker did not reach the interrupt point')
            owner=read(root/'trainer.json')
            child=read(root/'worker-started.json')
            require(owner['parent']==process.pid and owner['pid']==child['trainer']==child['group'],'Unexpected process ownership')
            receipt=read(output.parent/'prediction_manifest.json')
            require(receipt['delivery_phase']=='first' and receipt['delivery_status']=='partial','No early recovery export')
            require(output.read_bytes().startswith(b'\xef\xbb\xbf'),'BOM lost')
            with output.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.reader(handle,delimiter=';'))
            require(rows[0]==['Ticket','Chance'] and [r[0] for r in rows[1:]]==['0010','0001'],'Template order or IDs changed')
            require(all(abs(float(r[1])-2/3)<1e-12 for r in rows[1:]),'Probability lost original class identity')
            elapsed=time.monotonic()-start
            initial=sha(output)
            process.kill()
            require(process.wait(timeout=20)==-signal.SIGKILL,'Controller did not receive the controlled interruption')
            import fcntl
            def all_exited():
                for name in ('trainer','worker'):
                    with (root/(name+'.lock')).open('a') as f:
                        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:return False
                return True
            wait_for(all_exited,12)
        finally:
            if process.poll() is None:process.kill();process.wait()
    replay=root/'replay/answers.csv'
    invoke(['predict',train,'--run-dir',run/'first_delivery','--output',replay],root/'replay.log',settings)
    require(initial==sha(output)==sha(replay),'Crash or independent reload changed verified predictions')
    save(root/'verified.json',dict(fixture='scripted cited contract with two reordered prediction IDs',
         handler='production solve and predict',custom_format='UTF-8 BOM, semicolon, original ID, class probability, template order',
         early_phase=receipt['delivery_phase'],rows=len(rows)-1,early_seconds=round(elapsed,3),
         prediction_sha256=initial,owner_killed=True,trainer_and_worker_exited=True,independent_reload=True,
         fitted_classifiers=0,real_model_requests=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=lambda s:Path(s).resolve())
    verify(parser.parse_args().directory)
