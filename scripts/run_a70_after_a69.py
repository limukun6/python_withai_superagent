"""Bounded serial continuation: wait for A69, then freeze/ask/evaluate A70 once."""
import argparse
from datetime import datetime,timedelta,timezone
from pathlib import Path
import subprocess
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from tools.unattended_runtime import prevent_idle_sleep

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('study','reference','ledger','api-key-file','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();phase='wait_A69';start=time.monotonic()
    save(root/'protocol.json',dict(batch='A70-serial-queue',source_files=frozen,runner_sha256=sha(Path(__file__)),
        maximum_requests=24,maximum_new_classifier_fits=48,automatic_retry=False,history_unknown_retained=True))
    try:
        with prevent_idle_sleep():
            while not (a.study/'completion.json').exists():
                if time.monotonic()-start>10800:raise TimeoutError('A69 waiting window ended; A70 not started')
                time.sleep(5)
            if read(a.study/'completion.json')['status']!='completed':raise RuntimeError('A69 failed; no A70 request started')
            save(root/'window.json',dict(deadline=(datetime.now(timezone.utc)+timedelta(minutes=45)).isoformat()))
            prepared=root/'prepared';answers=root/'answers';evaluation=root/'evaluation'
            commands=[('prepare',['--study',str(a.study),'--reference',str(a.reference),'--output',str(prepared)]),
                ('ask',['--prepared',str(prepared),'--window-config',str(root/'window.json'),'--ledger',str(a.ledger),
                        '--api-key-file',str(a.api_key_file),'--execute','--output',str(answers)]),
                ('evaluate',['--prepared',str(prepared),'--answers',str(answers),'--window-config',str(root/'window.json'),'--output',str(evaluation)])]
            for phase,argv in commands:
                if sources()!=frozen:raise ValueError('Source changed before '+phase)
                with (root/(phase+'.log')).open('w',encoding='utf-8') as log:
                    result=subprocess.run([sys.executable,'-m','scripts.run_a70_four_task_feedback',phase,*argv],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,timeout=2800)
                if result.returncode:raise RuntimeError('Stopped without retry in '+phase)
                print('A70 '+phase+' completed',flush=True)
            save(root/'completion.json',dict(status='completed',prepare=read(prepared/'completion.json'),
                answers=read(answers/'completion.json'),evaluation=read(evaluation/'completion.json'),elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',phase=phase,error=str(error),automatic_retry=False,elapsed_seconds=time.monotonic()-start))
        raise
