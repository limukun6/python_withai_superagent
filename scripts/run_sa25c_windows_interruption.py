"""Correct only the Windows observation fixture; reuse frozen input, no new training."""
import argparse
import json
from pathlib import Path
import time

from scripts import run_sa25_joint_delivery as base
from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows

HOOK=r'''import json,os,sys,time,subprocess
from pathlib import Path
try:
    base=Path(os.environ['SA25_INTERRUPT_DIRECTORY'])
    def marker(name,pid,parent):
        (base/(name+'.json')).write_text(json.dumps(dict(pid=pid,parent=parent)),encoding='utf-8')
    if 'unified_agent' in sys.orig_argv:marker('controller',os.getpid(),os.getppid())
    original_popen=subprocess.Popen
    class ObservedPopen(original_popen):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            command=args[0] if args else kwargs.get('args',[])
            if isinstance(command,list):
                role='trainer_launch' if 'superagent' in command and 'unified' in command else 'worker_launch' if 'superagent.experiment_worker' in command else None
                if role:marker(role,self.pid,os.getpid())
    subprocess.Popen=ObservedPopen
    if 'superagent' in sys.orig_argv and 'unified' in sys.orig_argv:marker('trainer',os.getpid(),os.getppid())
    if 'superagent.experiment_worker' in sys.orig_argv:
        receipt=json.loads((base/'delivery/prediction_manifest.json').read_text(encoding='utf-8'))
        assert receipt['delivery_phase']=='first' and receipt['delivery_status']=='partial'
        path=base/'worker.tmp'
        path.write_text(json.dumps(dict(pid=os.getpid(),parent=os.getppid(),stage='before_worker_module_and_fit',argv=sys.orig_argv)),encoding='utf-8')
        path.replace(base/'worker.json')
        time.sleep(90)
except BaseException:
    os._exit(86)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','output','private-output','ledger'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();protocol=read(a.prepared/'protocol.json');cases=Path(protocol['cases_root'])
    if protocol['source_sha256']!=base.identity():raise ValueError('SA25B source changed')
    if any(sha(cases/n)!=v for n,v in protocol['inputs'].items()):raise ValueError('Input changed')
    from superagent.task_budget import TaskBudgetLedger
    ledger=TaskBudgetLedger(a.ledger);old=ledger_rows(ledger);clock=time.monotonic()
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    plan=base.ROOT/'docs/batches/SA25C_WINDOWS_OBSERVER_PLAN_20261127_zh.md'
    save(a.output/'protocol.json',dict(batch='SA25C-Windows-fixture-only',source_sha256=base.identity(),script_sha256=sha(Path(__file__)),
        plan_sha256=sha(plan),prepared_protocol_sha256=sha(a.prepared/'protocol.json'),maximum_actual_fits=0,maximum_campus_API=0,
        observation='Popen subclass preserves asyncio subclassing; records process identities only'))
    base.HOOK=HOOK
    try:
        result=base.interruption(cases,a.private_output/'windows_interruption',a.output/'windows_interruption')
    except Exception as issue:
        result=dict(id='windows_interruption',status='failed',error=dict(type=type(issue).__name__,message=str(issue)[:800]))
    result.update(entire_canonical_ledger_unchanged=old==ledger_rows(ledger),source_unchanged=protocol['source_sha256']==base.identity(),
        actual_classifier_fits=0,new_campus_API_requests=0,new_course_cost_rmb=0,elapsed_seconds=time.monotonic()-clock)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
