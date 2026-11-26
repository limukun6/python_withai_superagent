"""Read-only SA5 resource/provenance audit; no API, training or grading by another LLM."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--private-run',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--prior-rows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    receipt=read(args.run/'results.json');protocol=read(args.run/'protocol.json')
    root=Path(__file__).resolve().parents[1]
    unchanged=all(sha(root/name)==digest for name,digest in protocol['source_sha256'].items())
    prior=read(args.prior_rows)
    with closing(sqlite3.connect(args.ledger.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        actual={r['id']:dict(r) for r in db.execute('SELECT * FROM requests')}
    old_unchanged=all(actual.get(r['id'])==r for r in prior)
    new={key:r for key,r in actual.items() if key not in {r['id'] for r in prior}}
    usage=[];observations=[]
    for folder in sorted(args.private_run.iterdir()):
        if not folder.is_dir():continue
        events=folder/'events.jsonl'
        rows=[json.loads(s) for s in events.read_text(encoding='utf-8').splitlines()] if events.is_file() else []
        for event in rows:
            if event.get('kind')=='llm_response':
                try:
                    content=event['content'].strip()
                    if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
                    response=json.loads(content)
                    observations.append(dict(cell=folder.name,action=response.get('action','proposal'),
                        collection=response.get('collection'),member=response.get('member'),
                        response_sha256=hashlib.sha256(content.encode()).hexdigest()))
                except (KeyError,ValueError): observations.append(dict(cell=folder.name,action='invalid_json'))
        path=folder/'llm_usage.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines() if path.is_file() else []:
            event=json.loads(line);key=event.get('budget_request_id')
            if not key:continue
            native=new.get(key)
            if native is None:raise ValueError('Usage request absent from canonical ledger')
            row=dict(cell=folder.name,request_id=key,status=native['status'],
                     course_cost_rmb=native['course_units']/1e9 if native['course_units'] is not None else None)
            if native['status']=='known':
                inp,out=event['input_tokens'],event['output_tokens']
                expected=1000*(inp+4*out)
                resolution=json.loads(native['payload'])['resolution']
                if expected!=native['course_units'] or resolution['input_tokens']!=inp or resolution['output_tokens']!=out:
                    raise ValueError('Tokens/price/ledger settlement conflict')
                row.update(input_tokens=inp,output_tokens=out,served_model=event.get('served_model'),
                           latency_seconds=event.get('latency_seconds'))
            usage.append(row)
    if len({r['request_id'] for r in usage})!=len(usage):raise ValueError('Duplicate request in usage export')
    if set(new)!={r['request_id'] for r in usage}:raise ValueError('Unmatched new ledger requests')
    counts={}
    for arm in ['legacy','global']:
        cells=[r for r in receipt['cells'] if r['arm']==arm]
        counts[arm]=dict(cells=len(cells),accepted=sum(r['status']=='accepted' for r in cells),
            complete=sum(r.get('complete',False) for r in cells),rejected=sum(r['status']=='rejected' for r in cells),
            accepted_wrong=sum(r.get('accepted_wrong_contract',False) for r in cells),
            calls=sum(r['real_API_requests'] for r in cells))
    status='passed' if unchanged and old_unchanged and len(usage)==receipt['real_API_requests'] else 'failed'
    result=dict(status=status,source_unchanged=unchanged,all_prior_rows_unchanged=old_unchanged,
        prior_requests=len(prior),new_requests=len(new),new_unresolved=sum(r['status']!='known' for r in new.values()),
        input_tokens=sum(r.get('input_tokens',0) for r in usage),output_tokens=sum(r.get('output_tokens',0) for r in usage),
        known_incremental_course_cost_rmb=sum(r['course_units'] or 0 for r in new.values())/1e9,
        actual_school_debit_rmb=None,counts=counts,reply_actions=observations,usage=usage,
        actual_classifier_fits=0,real_API_requests_added_by_audit=0,team_total_verified=False)
    for name in ['results.json','completion.json']:
        (args.output/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','new_requests','new_unresolved','known_incremental_course_cost_rmb','counts']},ensure_ascii=False))
    return status!='passed'


if __name__=='__main__':raise SystemExit(main())
