"""Serial A50 then conditionally A51, under the already registered recovery deadline."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources


ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parents[1]/'runs'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','window-config','ledger','api-key-file'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    window=json.loads(args.window_config.read_text(encoding='utf-8-sig'));deadline=datetime.fromisoformat(window['deadline'])
    frozen=sources();runner=ROOT/'scripts/run_a50_facts_calibration.py';runner_hash=sha(runner);stages=[]
    save(root/'protocol.json',dict(batch='A50-A51-serial-live-queue',source_files=frozen,runner_sha256=runner_hash,
        window=window,maximum_new_real_requests=36,automatic_retry=False,training_queues=1))
    def left():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def execute(label,task_set,output,paid):
        if sources()!=frozen or sha(runner)!=runner_hash:raise ValueError('Frozen source changed')
        command=[sys.executable,'-m','scripts.run_a50_facts_calibration','--prepared',str(RUNS/'a34_transfer_prepared'),
            '--models-probe',str(RUNS/'a50_models_direct'),'--ledger',str(args.ledger),'--api-key-file',str(args.api_key_file),
            '--window-config',str(args.window_config),'--task-set',task_set,'--output',str(output)]
        if paid:command.append('--execute')
        record=dict(stage=label,status='running',started_at=datetime.now(timezone.utc).isoformat());stages.append(record);save(root/'stages.json',stages)
        print(json.dumps(dict(stage=label,status='started',seconds_left=round(left()))),flush=True)
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        record.update(status='completed' if code==0 else 'stopped',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat());save(root/'stages.json',stages)
        if code:raise RuntimeError('Stopped without retry: '+label)
        print(json.dumps(dict(stage=label,status='completed',seconds_left=round(left()))),flush=True)
    try:
        execute('A50','course',RUNS/'a50_facts_calibration',True)
        if left()>=3000:
            execute('A51_preflight','multimodal',RUNS/'a51_preflight',False)
            execute('A51','multimodal',RUNS/'a51_joint_transfer',True)
        else:
            stages.append(dict(stage='A51',status='not_started',reason='less_than_50_minutes_left'));save(root/'stages.json',stages)
        save(root/'completion.json',dict(status='completed_registered_stages',stages=stages,seconds_left=left(),source_unchanged=sources()==frozen,automatic_retry=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),stages=stages,seconds_left=left(),automatic_retry=False));raise
