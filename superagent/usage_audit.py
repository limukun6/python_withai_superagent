"""Normalize explicit historical request ledgers; no network or credential access."""
from collections import Counter,defaultdict
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

from .pricing import resolve_pricing


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def money_units(value):
    amount=Decimal(str(value))
    if not amount.is_finite() or amount<0:raise ValueError('Money must be finite and nonnegative')
    return int((amount*1_000_000_000).to_integral_value(rounding=ROUND_HALF_UP))


def read_source(descriptor):
    usage_path=Path(descriptor['usage_path']);event_path=Path(descriptor['events_path'])
    for path,key in ((usage_path,'usage_sha256'),(event_path,'events_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=descriptor[key]:raise ValueError('Historical source hash mismatch')
    usage=[(i,json.loads(line)) for i,line in enumerate(usage_path.read_text().splitlines(),1)
           if json.loads(line).get('event')!='ledger_initialized']
    requests=[(i,json.loads(line)) for i,line in enumerate(event_path.read_text().splitlines(),1)
              if json.loads(line).get('kind')=='llm_request']
    if len(usage)!=len(requests):raise ValueError('Historical requests and usage events do not pair one-to-one')
    result=[]
    for (request_line,request),(usage_line,record) in zip(requests,usage):
        if request.get('role')!=record.get('role'):raise ValueError('Request/usage role mismatch')
        if not request.get('time_utc') or record.get('time_utc','')<request['time_utc']:
            raise ValueError('Request/usage timestamps do not establish event order')
        request_id=fingerprint(request)
        unknown=record.get('usage_known') is False
        inp,out=record.get('input_tokens'),record.get('output_tokens')
        known_tokens=all(type(v) is int and v>=0 for v in (inp,out))
        cost=None
        if not unknown:
            if not known_tokens:raise ValueError('A supposedly metered call lacks valid token counts')
            price=resolve_pricing(record.get('served_model') or request['model'])
            cost=money_units(price.course_cost(inp,out))
            if record.get('course_cost_rmb') is None or money_units(record['course_cost_rmb'])!=cost:
                raise ValueError('Stored course cost does not agree with frozen official pricing')
            snapshot=record.get('pricing',{})
            if snapshot.get('price_version')!=price.price_version or snapshot.get('cache_discount_applied') is not False:
                raise ValueError('Unverified historical pricing policy')
        elif any(record.get(key) is not None for key in ('course_cost_rmb','cost_rmb')):
            raise ValueError('Unknown usage/cost must not be recorded as known zero')
        result.append({'request_id':request_id,'scope':descriptor['scope'],'stage':descriptor['stage'],
            'request_time_utc':request['time_utc'],'usage_time_utc':record['time_utc'],
            'requested_model':request.get('model'),'served_model':record.get('served_model'),
            'provider_response_id':record.get('response_id'),'event':record['event'],
            'token_usage_known':known_tokens,'input_tokens':inp if known_tokens else None,'output_tokens':out if known_tokens else None,
            'cost_status':'unknown' if unknown else 'known','course_cost_units':cost,
            'course_cost_rmb':cost/1e9 if cost is not None else None,
            'actual_account_debit_rmb':record.get('actual_cost_rmb'),
            'pricing':record.get('pricing'),'raw_usage':record.get('raw_usage'),
            'usage_event_fingerprint':fingerprint(record),
            'provenance':{'source_id':descriptor['id'],'usage_line':usage_line,'request_line':request_line}})
    return result


def aggregate(calls):
    unique={};copies=[]
    for call in calls:
        key=call['request_id']
        if key in unique:
            first=unique[key]
            if any(first[field]!=call[field] for field in ('usage_event_fingerprint','scope','stage')):
                raise ValueError('Conflicting records or allocations for one request event')
            copies.append({'request_id':key,'original':first['provenance'],'duplicate':call['provenance']})
        else:unique[key]=call
    rows=list(unique.values());scopes={};stages={}
    def group(selected):
        unknown=sum(row['cost_status']=='unknown' for row in selected)
        known=sum(row['course_cost_units'] or 0 for row in selected)
        return {'request_count':len(selected),'metered_count':len(selected)-unknown,'unresolved_count':unknown,
            'known_input_tokens':sum(row['input_tokens'] or 0 for row in selected),
            'known_output_tokens':sum(row['output_tokens'] or 0 for row in selected),
            'known_course_cost_units':known,'known_course_cost_rmb':known/1e9,
            'complete_course_cost_rmb':known/1e9 if not unknown else None,
            'actual_account_debit_rmb':sum(row['actual_account_debit_rmb'] for row in selected)
                if all(row['actual_account_debit_rmb'] is not None for row in selected) else None}
    for scope in sorted({row['scope'] for row in rows}):scopes[scope]=group([r for r in rows if r['scope']==scope])
    for stage in sorted({row['stage'] for row in rows}):stages[stage]=group([r for r in rows if r['stage']==stage])
    by_response=defaultdict(list)
    for row in rows:
        if row['provider_response_id']:by_response[row['provider_response_id']].append(row['request_id'])
    reused={key:value for key,value in by_response.items() if len(value)>1}
    return {'summary':group(rows),'scopes':scopes,'stages':stages,'calls':rows,'duplicate_copies_removed':copies,
            'reused_response_ids_retained_as_distinct_requests':reused,
            'policy':'Deduplicate identical request events with matching usage; never deduplicate distinct requests by response ID alone'}
