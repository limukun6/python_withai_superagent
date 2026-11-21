"""Read-only review of the exact eleven unknown calls; no guessed usage or credential access."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.a_batch_common import save
from superagent.usage_audit import fingerprint


def audit(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inventory=args.runs/'flash_history_unresolved.json'
    gaps=json.loads(inventory.read_text(encoding='utf-8'))
    save(out/'protocol.json',dict(batch='A21',inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),requests_reviewed=len(gaps),
        classifier_fits=0,real_api_requests=0,original_logs_modified=False))
    rows=[]
    for gap in gaps:
        usage=args.runs/gap['file'];events=usage.parent/'events.jsonl'
        row=dict(source=gap['file'],usage_line=gap['line'],requested_model=gap['requested_model'],
                 original_cost_status='unknown',recovered_actual_usage=None,upper_bound_status='unbounded')
        if not usage.exists() or not events.exists():
            row['reason']='original source unavailable';rows.append(row);continue
        row['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (usage,events)}
        records=[json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
        requests=[r for r in records if r.get('kind')=='llm_request' and r.get('time_utc')==gap['request_utc']
                  and r.get('role')==gap['role']]
        if len(requests)!=1:
            row['reason']='request identity not unique';rows.append(row);continue
        request=requests[0];row['request_id']=fingerprint(request);row['request_keys']=sorted(request)
        messages=request.get('messages')
        row['messages_utf8_bytes']=len(json.dumps(messages,ensure_ascii=False).encode('utf-8')) if messages is not None else None
        output_limit=request.get('max_tokens') or request.get('max_output_tokens') or (request.get('request_body') or {}).get('max_tokens')
        row['recorded_output_token_limit']=output_limit
        terminals=[r for r in records if r.get('role')==gap['role'] and gap['request_utc']<=r.get('time_utc','')<=gap['usage_utc']
                   and r.get('kind') in ('llm_response','llm_transport_failure','llm_http_failure')]
        row['terminal_events']=[dict(kind=r.get('kind'),http_status=r.get('http_status',r.get('status_code')),
            response_content_present=isinstance(r.get('content'),str),
            token_usage=r.get('usage') if isinstance(r.get('usage'),dict) else None) for r in terminals]
        # A transport error/response text is not a platform usage statement.
        row['reason']='No token-usage receipt bound to this request; cannot replace unknown with zero'
        if type(output_limit) is int and output_limit>0 and row['messages_utf8_bytes'] is not None:
            row['upper_bound_status']='candidate_limit_requires_request_enforcement_verification'
        rows.append(row)
    save(out/'results.json',rows)
    save(out/'completion.json',dict(status='reviewed',unknown_calls=len(rows),
        recovered_actual_usage=sum(r['recovered_actual_usage'] is not None for r in rows),
        recorded_output_limits=sum(r.get('recorded_output_token_limit') is not None for r in rows),
        exact_actual_cost_still_unknown=True,original_logs_modified=False,classifier_fits=0,real_api_requests=0))
    print(json.dumps(dict(rows=len(rows),output_limit_count=sum(r.get('recorded_output_token_limit') is not None for r in rows),
        request_key_sets=sorted({tuple(r.get('request_keys',[])) for r in rows}))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);audit(p.parse_args())
