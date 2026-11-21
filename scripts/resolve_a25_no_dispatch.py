"""Audit a single demonstrably unsent DNS request, preserving the original failure transition."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.a_batch_common import save
from superagent.task_budget import TaskBudgetLedger,encoded,now


def resolve(ledger,run,output):
    output.mkdir(parents=True,exist_ok=False)
    usage_path=run/'llm_usage.jsonl';events_path=run/'events.jsonl'
    usage=[json.loads(s) for s in usage_path.read_text(encoding='utf-8').splitlines()]
    unknown=[u for u in usage if u.get('event')=='unresolved_call']
    if len(unknown)!=1:raise ValueError('Expected exactly one unresolved request')
    ticket=unknown[0]['budget_request_id'];role=unknown[0]['role']
    events=[json.loads(s) for s in events_path.read_text(encoding='utf-8').splitlines()]
    requests=[e for e in events if e.get('kind')=='llm_request' and e.get('role')==role]
    failures=[e for e in events if e.get('kind')=='llm_transport_failure' and e.get('role')==role]
    if len(requests)!=1 or len(failures)!=1:raise ValueError('Ambiguous request failure evidence')
    if requests[0].get('budget_request_id')!=ticket:raise ValueError('Request evidence belongs to a different ticket')
    failure=failures[0];error=failure.get('error','')
    if not ('NameResolutionError' in error and 'getaddrinfo failed' in error and
            failure.get('http_status') is None and failure.get('data_events',0)==0):
        raise ValueError('Failure does not prove no HTTP dispatch')
    if any(e.get('role')==role and e.get('kind') in ('llm_response','llm_http_failure') for e in events):
        raise ValueError('A response was observed; DNS-only classification invalid')
    proof=dict(ticket=ticket,classification='not_dispatched',rule='DNS name resolution failed before any HTTP status or stream data',
        original_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (usage_path,events_path)},
        provider_token_receipt_present=False,actual_account_debit_rmb=None,original_logs_modified=False)
    with ledger._db() as db:
        db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT * FROM requests WHERE id=?',(ticket,)).fetchone()
        if row is None or row['status']!='unknown':raise ValueError('Expected the original unknown ticket')
        payload=json.loads(row['payload'])
        if Path(payload.get('run_id','')).resolve()!=run.resolve():raise ValueError('Ledger run identity differs')
        payload['no_dispatch_review']=proof
        db.execute('UPDATE requests SET status=?,course_units=?,payload=? WHERE id=?',('not_dispatched',0,encoded(payload),ticket))
        db.execute('INSERT INTO transitions(request_id,event,time_utc,payload) VALUES (?,?,?,?)',
            (ticket,'reviewed_not_dispatched',now(),encoded(proof)));db.commit()
    save(output/'proof.json',proof)
    save(output/'completion.json',dict(status='verified_not_dispatched',request_attempts=1,model_requests_sent=0,
        real_new_model_usage_rmb=0,project_usage=ledger.project_summary(),historical_eleven_unknown_unchanged=True,
        actual_classifier_fits=0,new_requests=0))
    print(json.dumps(ledger.project_summary()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();resolve(TaskBudgetLedger(a.ledger),a.run,a.output)
