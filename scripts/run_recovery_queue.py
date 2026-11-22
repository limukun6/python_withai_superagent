"""Bounded serial continuation after standby; preserves partial runs and never retries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources


ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parents[1]/'runs'


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    window=json.loads(args.window_config.read_text(encoding='utf-8-sig'));deadline=datetime.fromisoformat(window['deadline'])
    frozen=sources();stages=[]
    scripts=['run_a42_recovery_supplement.py','run_a29_choice_matrix.py','run_a34_transfer_coverage.py','run_a35_fixed_state_study.py']
    frozen_scripts={name:sha(ROOT/'scripts'/name) for name in scripts}
    save(root/'protocol.json',dict(batch='recovery-queue',window=window,source_files=frozen,
        script_hashes=frozen_scripts,automatic_retry=False,training_queues=1))
    def left():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def execute(label,module,arguments):
        if sources()!=frozen or any(sha(ROOT/'scripts'/n)!=h for n,h in frozen_scripts.items()):raise ValueError('Frozen source changed')
        command=[sys.executable,'-m','scripts.'+module,*map(str,arguments)]
        record=dict(stage=label,status='running',started_at=datetime.now(timezone.utc).isoformat(),command=command)
        stages.append(record);save(root/'stages.json',stages)
        print(json.dumps(dict(stage=label,status='started',seconds_left=round(left()))),flush=True)
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        record.update(status='completed' if code==0 else 'stopped',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
        save(root/'stages.json',stages)
        if code:raise RuntimeError('Stage stopped without retry: '+label)
        print(json.dumps(dict(stage=label,status='completed',seconds_left=round(left()))),flush=True)
    def skip(stage,reason):
        stages.append(dict(stage=stage,status='not_started',reason=reason));save(root/'stages.json',stages)
    matrix=RUNS/'a42_recovery_supplement';transfer=RUNS/'a34_transfer_coverage'
    shared=['--ledger',args.ledger,'--api-key-file',args.api_key_file,'--window-deadline',window['deadline']]
    def fixed(label,task_set):
        prepared=RUNS/(label+'_states');answers=RUNS/(label+'_answers')
        execute(label+'_prepare','run_a35_fixed_state_study',['prepare','--matrix',matrix,'--transfer',transfer,
            '--training-prepared',RUNS/'a34_transfer_prepared','--task-set',task_set,'--output',prepared])
        execute(label+'_preflight','run_a35_fixed_state_study',['ask','--prepared',prepared,*shared,'--output',RUNS/(label+'_preflight')])
        execute(label+'_ask','run_a35_fixed_state_study',['ask','--prepared',prepared,*shared,'--output',answers,'--execute'])
        execute(label+'_evaluate','run_a35_fixed_state_study',['evaluate','--prepared',prepared,'--answers',answers,
            '--window-deadline',window['deadline'],'--output',RUNS/(label+'_evaluation')])
    try:
        execute('A42','run_a42_recovery_supplement',['--original',RUNS/'a32_coverage_matrix','--prepared',RUNS/'a32_coverage_prepared',
            *shared,'--output',matrix,'--execute'])
        transfer_done=False
        common=['execute','--prepared',RUNS/'a34_transfer_prepared','--after-matrix',matrix,*shared]
        if left()>=3000:
            execute('A34_preflight','run_a34_transfer_coverage',[*common,'--output',RUNS/'a34_transfer_preflight'])
            execute('A34','run_a34_transfer_coverage',[*common,'--output',transfer,'--execute']);transfer_done=True
        else:skip('A34','less_than_50_minutes_left')
        if left()>=6000 and transfer_done:fixed('a35_all','all-four')
        elif left()>=3600:
            fixed('a35_course','course')
            if transfer_done and left()>=3600:fixed('a35_multimodal','multimodal')
        else:skip('A35','less_than_60_minutes_left')
        outcomes=read(matrix/'results.json');bykey={(r['task'],r['arm']):r for r in outcomes}
        increased=any(bykey[t,b+'_routes']['full_refit_comparison_eligible'] and
            len(bykey[t,b+'_routes']['explored_routes'])>len(bykey[t,b+'_off']['explored_routes'])
            for t in ('coupon','preference') for b in ('R0','E0','R1','E1'))
        if transfer_done and increased and left()>=2700:
            execute('A36_preflight','run_a34_transfer_coverage',[*common,'--task-set','all-four','--fit-limit','5','--output',RUNS/'a36_budget_preflight'])
            execute('A36','run_a34_transfer_coverage',[*common,'--task-set','all-four','--fit-limit','5','--output',RUNS/'a36_budget_coverage','--execute'])
        else:skip('A36','coverage_or_transfer_or_45_minute_condition_not_met')
        save(root/'completion.json',dict(status='completed_registered_recovery_queue',stages=stages,seconds_left=left(),
            source_unchanged=sources()==frozen,automatic_retry=False,original_A32_completed=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),stages=stages,seconds_left=left(),automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','window-config','ledger','api-key-file'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
