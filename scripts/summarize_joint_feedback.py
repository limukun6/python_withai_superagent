"""Gather small, shareable results from the matched runs and their independent receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from scripts.unified_released_study import read, require, save, sha

MEASURED='a26b0c02e55787d24aecc06fb2336394fda4a171'
TIMING='65ce99fd12cd048c9b4c83233b42e7c9f7202d6e'


def collect(root,ledger,destination):
    study=root/'development'
    panels=read(study/'scored.json')
    receipts={}
    fits=calls=unknown=0
    for case,panel in panels['cases'].items():
        rows={}
        for arm,record in panel['arms'].items():
            folder=study/case/arm
            agent=folder/'run/agent'
            accounting=read(agent/'run_outcome.json')
            require(accounting['new_api_requests']==record['requests'] and accounting['attempted_classifier_fits']==record['fits'],
                    'Independent receipts disagree')
            require(sha(folder/'delivery/predictions.csv')==sha(folder/'reload/predictions.csv')==record['output_sha'],
                    'Changed final output or reload')
            row={k:record[k] for k in ('primary_score','correct','fits','requests','opportunity','core','tested',
                        'check','verdict','changed','comparisons','remaining','result','model_sha','output_sha','fresh_reload','seconds')}
            row.update(course_cost_rmb=accounting['complete_run_course_cost_rmb'],unknown_requests=accounting['unresolved_api_requests'],
                       decision_source=record['source_at_decision'],receipt_sha=sha(agent/'run_outcome.json'))
            fits+=row['fits'];calls+=row['requests'];unknown+=row['unknown_requests']
            rows[arm]=row
        receipts[case]=dict(metric=panel['metric'],rows=panel['rows'],arms=rows,paired=panel['paired_vs_control'])
    paid=panels['campus_gate']['case']
    raw_scope='joint-nov27-'+paid
    with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
        entries=db.execute('select id,status,reserved_units,course_units,payload from requests where scope=?',(raw_scope,)).fetchall()
    require(len(entries)==calls==unknown==1,'Expected one actual unresolved campus attempt in this batch')
    ticket,status,reserved,amount,raw=entries[0]
    require(status=='unknown' and amount is None,'The recorded state of the unresolved campus attempt has changed')
    audit=[json.loads(line) for line in (study/paid/'feedback-flash/run/agent/llm_usage.jsonl').read_text().splitlines()]
    require(any(row.get('budget_request_id')==ticket and row.get('usage_known') is False for row in audit),
            'Request and ledger did not match')
    sources=panels['registration']
    released=read(root/'released-entry/verified.json')
    require(released['agent']==sources['frozen']['runtime'],'Course and development source versions differ')
    timings=read(root/'timing-replay/verified.json')
    later=read(root/'timing-replay/late_verified.json')
    results=dict(experiment_commit=MEASURED,scheduling_commit=TIMING,registration=sources,
        course_validation=released['cases'],development=receipts,campus_gate=panels['campus_gate'],
        request=dict(scope=raw_scope,request_id=ticket,status=status,raw_record_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            reserved_rmb=reserved/1e9,course_cost_rmb=None,actual_debit_rmb=None,input_tokens=None,output_tokens=None,
            requested_model=json.loads(raw)['requested_model'],issue='planning deadline expired while receiving stream'),
        custom_interruption=read(root/'custom-boundary/verified.json'),slice_replay=read(study/'slice_replay.json'),
        intervention=read(study/'intervention.json'),controlled_review_timing=timings,controlled_late_success=later)
    destination.mkdir(parents=True,exist_ok=True)
    save(destination/'results.json',results)
    costs=dict(cold_trained_runs=2+sum(len(v['arms']) for v in receipts.values()),course_fits=sum(v['fits'] for v in released['cases'].values()),
               development_fits=fits,additional_engineering_or_replay_fits=0,physical_fits_total=fits+sum(v['fits'] for v in released['cases'].values()),
               model_requests=calls,unresolved_requests=unknown,total_course_cost_rmb=None,total_token_usage=None)
    save(destination/'completion.json',dict(experiment_commit=MEASURED,scheduling_commit=TIMING,
        results_sha256=sha(destination/'results.json'),resources=costs,all_final_outputs_replayed=True,
        formal_course_test_read=False,H1_H2_opened=False,changed_outer_predictions_in_matched_local_comparison=0,
        limitations=['Three independent classification sources, one registered outer sample per source, in evaluator-prepared generic packages.',
                     'No valid campus interpretation on this live run; no measured live reviewer-quality improvement.',
                     'Timing change measured with captured training artifacts and controlled reply timing.',
                     'Contrast intervention and specialist forwarding assessed on existing training evidence only.']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','outbox','destination'):p.add_argument('--'+name,required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    collect(args.root,args.outbox,args.destination)
