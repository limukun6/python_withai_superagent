"""Durable serial continuation of already registered stages; stops on any failed stage."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from superagent.ab_run import sources
from scripts.status_a29_choice_matrix import safe_read, completion_failure


ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parents[1]/'runs'


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    config=read(args.window_config);deadline=datetime.fromisoformat(config['deadline']);frozen=sources();stages=[]
    scripts=['run_a34_transfer_coverage.py','run_a35_fixed_state_study.py','summarize_a29_choice_matrix.py']
    files={name:sha(ROOT/'scripts'/name) for name in scripts}
    save(root/'protocol.json',dict(batch='three-hour-durable-queue',window=config,source_files=frozen,
        stage_script_sha256=files,automatic_retries=False,one_training_queue=True,
        parent_matrix=str(args.after_matrix),maximum_new_requests_after_A32=48))
    def left():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def execute(name,module,arguments):
        if sources()!=frozen or any(sha(ROOT/'scripts'/n)!=v for n,v in files.items()):
            raise RuntimeError('Frozen source or a queued script changed')
        command=[sys.executable,'-m','scripts.'+module,*map(str,arguments)]
        record=dict(stage=name,status='running',command=command,started_at=datetime.now(timezone.utc).isoformat())
        stages.append(record);save(root/'stages.json',stages)
        print(json.dumps(dict(stage=name,status='started',seconds_left=round(left()))),flush=True)
        with (root/(name+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        record.update(status='completed' if code==0 else 'stopped',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
        save(root/'stages.json',stages)
        if code:raise RuntimeError('Stage stopped without retry: '+name)
        print(json.dumps(dict(stage=name,status='completed',seconds_left=round(left()))),flush=True)
    common=['--prepared',args.training_prepared,'--after-matrix',args.after_matrix,'--ledger',args.ledger,
            '--api-key-file',args.api_key_file,'--window-deadline',config['deadline']]
    def fixed_study(label,task_set):
        prepared=RUNS/(label+'_states');answers=RUNS/(label+'_answers');evaluation=RUNS/(label+'_evaluation')
        execute(label+'_prepare','run_a35_fixed_state_study',['prepare','--matrix',args.after_matrix,
            '--transfer',RUNS/'a34_transfer_coverage','--training-prepared',args.training_prepared,
            '--task-set',task_set,'--output',prepared])
        execute(label+'_preflight','run_a35_fixed_state_study',['ask','--prepared',prepared,'--ledger',args.ledger,
            '--api-key-file',args.api_key_file,'--output',RUNS/(label+'_preflight'),'--window-deadline',config['deadline']])
        execute(label+'_ask','run_a35_fixed_state_study',['ask','--prepared',prepared,'--ledger',args.ledger,
            '--api-key-file',args.api_key_file,'--output',answers,'--window-deadline',config['deadline'],'--execute'])
        execute(label+'_evaluate','run_a35_fixed_state_study',['evaluate','--prepared',prepared,'--answers',answers,
            '--output',evaluation,'--window-deadline',config['deadline']])
    try:
        print(json.dumps(dict(status='waiting_for_main_matrix',seconds_left=round(left()))),flush=True)
        while not (args.after_matrix/'completion.json').exists():
            for outcome_path in args.after_matrix.glob('*/run_outcome.json'):
                outcome = safe_read(outcome_path)
                if outcome is not None and not outcome.get('requested_planner_completed'):
                    raise RuntimeError('Main path failed; no further stages started: '+outcome_path.parent.name+': '+str(completion_failure(outcome)))
            if left()<=0:raise RuntimeError('Window ended while waiting for the main matrix')
            time.sleep(3)
        completed=read(args.after_matrix/'completion.json')
        if completed['status']!='completed':raise RuntimeError('Main matrix did not complete; no further paid stages started')
        if read(args.after_matrix/'protocol.json')['source_files']!=frozen:raise RuntimeError('Main study source differs')
        execute('A32_summary','summarize_a29_choice_matrix',['--matrix',args.after_matrix,'--ledger',args.ledger,
            '--output',ROOT/'research/a32_coverage_matrix','--study','route-coverage'])
        transfer_done=False
        if left()>=3000:
            execute('A34_preflight','run_a34_transfer_coverage',['execute',*common,'--output',RUNS/'a34_transfer_preflight'])
            execute('A34_execute','run_a34_transfer_coverage',['execute',*common,'--output',RUNS/'a34_transfer_coverage','--execute'])
            transfer_done=True
        else:stages.append(dict(stage='A34',status='not_started_time_limit'));save(root/'stages.json',stages)
        if left()>=6000 and transfer_done:
            fixed_study('a35_all','all-four')
        elif left()>=3600:
            fixed_study('a35_course','course')
            if transfer_done and left()>=3600:fixed_study('a35_multimodal','multimodal')
        else:stages.append(dict(stage='A35',status='not_started_time_limit'));save(root/'stages.json',stages)
        summary=read(ROOT/'research/a32_coverage_matrix/summary.json')
        coverage_increased=any(len(task['arms'][base+'_routes']['explored_routes'])>
            len(task['arms'][base+'_off']['explored_routes']) for task in summary['tasks'].values()
            for base in ('R0','E0','R1','E1'))
        if transfer_done and coverage_increased and left()>=2700:
            execute('A36_preflight','run_a34_transfer_coverage',['execute',*common,'--task-set','all-four','--fit-limit','5',
                '--output',RUNS/'a36_budget_preflight'])
            execute('A36_execute','run_a34_transfer_coverage',['execute',*common,'--task-set','all-four','--fit-limit','5',
                '--output',RUNS/'a36_budget_coverage','--execute'])
        else:stages.append(dict(stage='A36',status='not_started_condition_or_time_limit'));save(root/'stages.json',stages)
        save(root/'completion.json',dict(status='completed_registered_queue',stages=stages,seconds_left=left(),
            source_unchanged=sources()==frozen,automatic_retries=False,
            note='Queue completion is not a claim that every conditional stage ran or that the full three-hour window elapsed'))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),stages=stages,seconds_left=left(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','window-config','after-matrix','training-prepared','ledger','api-key-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
