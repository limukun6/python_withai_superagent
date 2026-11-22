"""Wait for the one live queue, then run predeclared feedback/action/delivery checks serially."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources


ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT.parents[1]/'runs'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','window-config','ledger','api-key-file'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    window=json.loads(args.window_config.read_text(encoding='utf-8-sig'));deadline=datetime.fromisoformat(window['deadline'])
    frozen=sources();stages=[]
    paths=[ROOT/'scripts'/n for n in ('run_a55_fresh_feedback.py','run_a53_action_audit.py','run_a52_relocation.py','run_a54_export_acceptance.py','run_a26_symmetric.py')]
    paths.extend([ROOT/'tools/export_model_bundle.py',ROOT/'scripts/campus_direct.py'])
    immutable={str(p):sha(p) for p in paths}
    save(root/'protocol.json',dict(batch='A50-tail-queue',source_files=frozen,script_hashes=immutable,window=window,
        maximum_extra_requests=12,automatic_retries=False,single_training_queue=True))
    def left():return (deadline-datetime.now(timezone.utc)).total_seconds()
    def execute(label,module,argv):
        if sources()!=frozen or any(sha(Path(p))!=h for p,h in immutable.items()):raise ValueError('Frozen source/script changed')
        record=dict(stage=label,status='running',started_at=datetime.now(timezone.utc).isoformat());stages.append(record);save(root/'stages.json',stages)
        print(json.dumps(dict(stage=label,status='started',seconds_left=round(left()))),flush=True)
        with (root/(label+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run([sys.executable,'-m',module,*map(str,argv)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        record.update(status='completed' if code==0 else 'stopped',exit_code=code,finished_at=datetime.now(timezone.utc).isoformat());save(root/'stages.json',stages)
        print(json.dumps(dict(stage=label,status=record['status'],seconds_left=round(left()))),flush=True)
        return code==0
    def skip(stage,reason):
        stages.append(dict(stage=stage,status='not_started',reason=reason));save(root/'stages.json',stages)
    try:
        print(json.dumps(dict(status='waiting_for_live_queue',seconds_left=round(left()))),flush=True)
        live=RUNS/'a50_live_queue/completion.json'
        while not live.exists() and left()>0:time.sleep(5)
        live_ok=live.exists() and read(live)['status']=='completed_registered_stages'
        course=RUNS/'a50_facts_calibration';native=RUNS/'a51_joint_transfer'
        course_ok=(course/'completion.json').exists() and read(course/'completion.json')['status']=='completed'
        if live_ok and course_ok and left()>=1260:
            prefix=3 if left()>=2760 else 1
            prepared=RUNS/'a55_fresh_states';answers=RUNS/'a55_fresh_answers'
            base=['--window-config',args.window_config]
            if execute('A55_prepare','scripts.run_a55_fresh_feedback',['prepare',*base,'--study',course,'--prefix-count',prefix,'--output',prepared]):
                common=['--prepared',prepared,'--ledger',args.ledger,'--api-key-file',args.api_key_file,*base]
                if execute('A55_preflight','scripts.run_a55_fresh_feedback',['ask',*common,'--output',RUNS/'a55_fresh_preflight']):
                    if execute('A55_ask','scripts.run_a55_fresh_feedback',['ask',*common,'--output',answers,'--execute']):
                        execute('A55_evaluate','scripts.run_a55_fresh_feedback',['evaluate','--prepared',prepared,'--answers',answers,*base,'--output',RUNS/'a55_fresh_evaluation'])
        else:skip('A55','live_not_successful_or_less_than_21_minutes')
        if course_ok and left()>=1260:
            execute('A53','scripts.run_a53_action_audit',['--study',course,'--window-config',args.window_config,'--output',RUNS/'a53_action_audit'])
        else:skip('A53','course_incomplete_or_less_than_21_minutes')
        native_ok=(native/'completion.json').exists() and read(native/'completion.json')['status']=='completed'
        if course_ok and native_ok:
            execute('A52','scripts.run_a52_relocation',['--course',course,'--native',native,'--output',RUNS/'a52_relocation'])
        else:skip('A52','requires_both_live_studies_completed')
        execute('A54','scripts.run_a54_export_acceptance',['--catalog-study',RUNS/'a47_catalog_Q','--partial-study',RUNS/'a44_a46_offline_final','--output',RUNS/'a54_selfcontained_delivery'])
        if left()>=120:
            execute('final_regression','scripts.run_a26_symmetric',['checks','--include-coverage','--output',RUNS/'a50_final_regression'])
        else:skip('final_regression','less_than_two_minutes')
        save(root/'completion.json',dict(status='completed_tail_with_stage_statuses',stages=stages,seconds_left=left(),source_unchanged=sources()==frozen,
            automatic_retry=False,all_executed_stages_passed=all(r['status']!='stopped' for r in stages)))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),stages=stages,seconds_left=left(),automatic_retry=False));raise
