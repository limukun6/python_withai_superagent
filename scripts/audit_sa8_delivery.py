"""Audit the first SA8 attempt before repairing the evaluation-only recipe comparison."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--private-run',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True);parser.add_argument('--prior-rows',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];receipt=read(args.run/'results.json');protocol=read(args.run/'protocol.json')
    unchanged=all(sha(root/name)==digest for name,digest in protocol['source_sha256'].items())
    prior=read(args.prior_rows);previous={r['id']:r for r in prior}
    with closing(sqlite3.connect(args.ledger.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row;current={r['id']:dict(r) for r in db.execute('SELECT * FROM requests')}
    old_unchanged=all(current.get(k)==r for k,r in previous.items());new={k:r for k,r in current.items() if k not in previous}
    usage=[];physical=0;case_records=[]
    for case in receipt['cases']:
        folder=args.private_run/case['case']/'agent'
        p=folder/'llm_usage.jsonl'
        for line in p.read_text(encoding='utf-8').splitlines() if p.is_file() else []:
            event=json.loads(line);key=event.get('budget_request_id')
            if not key:continue
            actual=new[key];resolution=json.loads(actual['payload']).get('resolution',{})
            if actual['status']!='known' or actual['course_units']!=1000*(event['input_tokens']+4*event['output_tokens']):
                raise ValueError('Metering conflict')
            if resolution.get('input_tokens')!=event['input_tokens'] or resolution.get('output_tokens')!=event['output_tokens']:
                raise ValueError('Token settlement conflict')
            usage.append(dict(case=case['case'],request_id=key,input_tokens=event['input_tokens'],output_tokens=event['output_tokens'],
                course_cost_rmb=actual['course_units']/1e9,served_model=event.get('served_model')))
        fits=read(folder/'classifier_fits.json') if (folder/'classifier_fits.json').is_file() else dict(attempted_classifier_fits=0,workers=[])
        physical+=fits['attempted_classifier_fits']
        target=args.output/case['case'];target.mkdir()
        save(target/'classifier_fits.json',fits)
        for name in ['semantic_contract.json','final_selection.json','prediction_preflight.json']:
            if (folder/name).is_file():save(target/name,read(folder/name))
        case_records.append(dict(case=case['case'],original_status=case['status'],physical_fits=fits['attempted_classifier_fits'],
            schema_checks=case.get('semantic_gate'),model_sha256=case.get('model_sha256'),prediction_sha256=case.get('prediction_sha256')))
    ok=(unchanged and old_unchanged and physical==receipt['actual_classifier_fits'] and
        len(usage)==len(new)==receipt['real_API_requests'] and len({r['request_id'] for r in usage})==len(usage))
    result=dict(status='passed' if ok else 'failed',source_unchanged=unchanged,all_prior_rows_unchanged=old_unchanged,
        original_cases=case_records,actual_classifier_fits=physical,real_API_requests=len(new),new_unknowns=sum(r['status']!='known' for r in new.values()),
        input_tokens=sum(r['input_tokens'] for r in usage),output_tokens=sum(r['output_tokens'] for r in usage),
        known_incremental_course_cost_rmb=sum(r['course_units'] or 0 for r in new.values())/1e9,usage=usage,
        audit_classifier_fits=0,audit_API_requests=0)
    save(args.output/'results.json',result);save(args.output/'completion.json',result)
    print(json.dumps({k:result[k] for k in ['status','actual_classifier_fits','real_API_requests','known_incremental_course_cost_rmb']}));return not ok


if __name__=='__main__':raise SystemExit(main())
