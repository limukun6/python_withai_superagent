"""Replay one unchanged historical real failure; zero fitting and zero new API."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--events',type=Path,default=Path('runs/sa12_live_entry_01/V1/agent/events.jsonl'))
    p.add_argument('--invocation',type=Path,default=Path('research/sa12_live_entry_01/invocation.json'))
    p.add_argument('--ledger',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from superagent.semantic_contract import compile_plan,probe_prediction_input,evidence_workspace,_diagnostic,_output_overlay,resolve_training_task
    from superagent.format_sources import package_files
    from superagent.format_intake import DECODE_ERRORS
    from superagent.recovery import RecoveryController
    from superagent.task_budget import TaskBudgetLedger
    events=[json.loads(s) for s in a.events.read_text(encoding='utf-8').splitlines()]
    selected=None;previous=None;request_id=None
    for event in events:
        if event['kind']=='llm_request':request_id=event.get('budget_request_id')
        if event['kind']=='llm_response':previous=event['content']
        if event['kind']=='semantic_contract_rejected' and event.get('proposal',{}).get('version','').startswith('semantic-'):selected=(event,previous,request_id)
    if selected is None:raise ValueError('No historical contract failure to replay')
    event,content,request_id=selected
    parsed=content.strip()
    if parsed.startswith('```'):parsed='\n'.join(parsed.splitlines()[1:-1])
    if json.loads(parsed)!=event['proposal']:raise ValueError('Rejected proposal differs from actual reply')
    command=read(a.invocation)['command'];package=Path(command[command.index('solve')+1]);query=Path(command[command.index('--prediction-input')+1])
    paths=[a.events,package,query];before={str(path):sha(path) for path in paths}
    ledger=TaskBudgetLedger(a.ledger);ledger_before=ledger.project_summary()
    with ledger._db() as db:row=dict(db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone())
    if row['status']!='known':raise ValueError('Historical response is not settled')
    save(a.output/'protocol.json',dict(batch='SA21-historical-failure-replay',source_sha256=before,
        original_response_text_sha256=hashlib.sha256(content.encode()).hexdigest(),original_request_id=request_id,
        actual_classifier_fits=0,real_API_requests=0,scope='unchanged historical response, current compiler and summary; no new model continuation'))
    diagnostic=None
    try:
        with package_files(package) as files,package_files(query) as queries:
            if event['proposal']['version'].startswith('semantic-output'):
                task=_output_overlay(resolve_training_task(package),evidence_workspace(files),event['proposal'])
            else:task=compile_plan(files,evidence_workspace(files),event['proposal'],prediction_files=queries)
            probe_prediction_input(query,task)
    except DECODE_ERRORS as error:diagnostic=_diagnostic(error,'contract_probe')
    if diagnostic is None:raise ValueError('Historical failure no longer fails under current compiler')
    controller=RecoveryController();controller.failure('contract_probe',diagnostic,action=event['proposal'])
    record=controller.context()['failed_actions'][-1]
    ledger_unchanged=ledger_before==ledger.project_summary()
    unchanged=before=={str(path):sha(path) for path in paths} and ledger_unchanged
    result=dict(status='passed' if unchanged else 'failed',actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        original_request_id=request_id,source_unchanged=unchanged,old_ledger_unchanged=ledger_unchanged,record=record,
        scope='historical real failure replay; does not demonstrate a new live repair or accuracy improvement')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
