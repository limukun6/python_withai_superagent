"""Import real local usage events, preserving unknowns; no platform balance claims."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.a_batch_common import save
from superagent.pricing import resolve_pricing
from superagent.task_budget import TaskBudgetLedger
from superagent.usage_audit import fingerprint,money_units


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    calls={};sources=[];roots=['benchmarks','agent_architecture','v2_pilot','sse_live_validation','v2_pilot_local_followup']
    for root in roots:
        for usage in sorted((a.runs/root).rglob('llm_usage.jsonl')):
            rows=[json.loads(s) for s in usage.read_text(encoding='utf-8').splitlines() if json.loads(s).get('event')!='ledger_initialized']
            if not rows:continue
            events=usage.parent/'events.jsonl';records=[json.loads(s) for s in events.read_text(encoding='utf-8').splitlines()]
            requests=[r for r in records if r.get('kind')=='llm_request'];used=set()
            scope='task1' if 'categorical_tabular' in str(usage) else 'task2' if 'pair_text' in str(usage) else 'shared_diagnostics'
            sources.append(dict(usage=str(usage.relative_to(a.runs)),usage_sha256=hashlib.sha256(usage.read_bytes()).hexdigest(),
                events_sha256=hashlib.sha256(events.read_bytes()).hexdigest(),scope=scope))
            for row in rows:
                choices=[(i,r) for i,r in enumerate(requests) if i not in used and r.get('role')==row.get('role')
                    and r.get('time_utc','')<=row.get('time_utc','')]
                if choices:
                    index,request=choices[0];used.add(index);key=fingerprint(request);basis='original_request_event'
                elif row.get('event')=='nonstream_comparison':
                    request={};key=fingerprint(dict(standalone_observed_usage=row));basis='standalone_original_usage_event'
                else:raise ValueError('Cannot bind real usage to request: '+str(usage))
                model=row.get('served_model') or row.get('requested_model') or request.get('model')
                price=resolve_pricing(model);unknown=row.get('usage_known') is False
                inp,out_tokens=row.get('input_tokens'),row.get('output_tokens')
                cost=None if unknown else price.course_cost(inp,out_tokens)
                call=dict(request_id=key,scope=scope,cost_status='unknown' if unknown else 'known',
                    course_cost_units=None if unknown else money_units(cost),course_cost_rmb=cost,
                    input_tokens=None if unknown else inp,output_tokens=None if unknown else out_tokens,
                    requested_model=request.get('model') or row.get('requested_model'),served_model=row.get('served_model'),
                    pricing=price.metadata(),original_usage_event_sha256=fingerprint(row),identity_basis=basis,
                    actual_account_debit_rmb=None,verification_basis='local original receipts; no platform reconciliation')
                if key in calls and calls[key]!=call:raise ValueError('Conflicting copies of historical request')
                calls[key]=call
            if len(used)!=len(requests):raise ValueError('Historical request without terminal usage is not accounted')
    unknown=sum(r['cost_status']=='unknown' for r in calls.values())
    if unknown!=11:raise ValueError('Expected eleven preserved unknown real calls; inspect new inventory')
    audit=dict(calls=list(calls.values()),sources=sources,historical_unknown_preserved=11,
        complete_course_cost_rmb=None,platform_reconciled=False,fixture_sources_included=False,
        scope='Original local source roots only; external/teammate records not claimed complete')
    path=out/'history_audit.json';save(path,audit)
    ledger=TaskBudgetLedger.initialize(out/'ledger.sqlite',history=path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(out/'completion.json',dict(status='imported_with_unknowns',**ledger.project_summary(),
        source_files=len(sources),real_api_requests=0,classifier_fits=0))
    print(json.dumps(ledger.project_summary()))
