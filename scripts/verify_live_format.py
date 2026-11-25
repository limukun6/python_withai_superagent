"""One synthetic campus format request with a separately identifiable local reconciliation record."""
import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import ssl
import subprocess
import sys

from scripts.accept_record_formats import ROOT, fixed, read, save
from scripts.campus_direct import campus_resolution
from superagent.ab_flash import ENDPOINT, MODEL
from superagent.autonomous_run import train
from superagent.experiment_llm import ExperimentLLM
from superagent.task_budget import TaskBudgetAccount, TaskBudgetLedger


def campus_address():
    hostname = 'aigc-api.hkust-gz.edu.cn'
    addresses = ['10.121.10.250']
    try:
        addresses.extend(info[4][0] for info in socket.getaddrinfo(hostname,443,type=socket.SOCK_STREAM))
    except socket.gaierror:
        pass
    context = ssl.create_default_context()
    for address in dict.fromkeys(addresses):
        try:
            with socket.create_connection((address,443),timeout=4) as sock:
                with context.wrap_socket(sock,server_hostname=hostname):
                    return address
        except (OSError,ssl.SSLError):
            continue
    raise RuntimeError('Campus TLS preflight did not reach a certificate-verified endpoint. No API request was sent.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--key-file',required=True,type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    scope='synthetic-format-intake-1125'
    save(root/'protocol.json',dict(scope=scope,allowed_model=MODEL,provider=ENDPOINT,
        max_api_requests=1,allowance_rmb=.10,physical_fit_limit_including_refit=2,synthetic_schema_only=True,
        team_ledger_written=False,merge_status='pending_team_handoff',course_project_total='unverified',
        charge_basis='actual authenticated model usage at frozen course rates; no platform debit assumed'))
    ledger=TaskBudgetLedger.initialize(root/'pending_merge.sqlite',new_project=True)
    with sqlite3.connect(ledger.path) as db:
        db.executemany('INSERT INTO metadata VALUES (?,?)',[('ledger_kind','provisional_local_outbox'),
                       ('team_merge_state','pending_team_handoff')])
    folder=root/'inputs'
    folder.mkdir()
    training,query=fixed(folder)
    native=root/'run'/'agent'
    client=ExperimentLLM(ENDPOINT,str(args.key_file),MODEL,.10,max_requests=1,allowed_served_models={MODEL},
                         account=TaskBudgetAccount(ledger,scope,'format-intake-live-1125'),streaming=True)
    status='not_dispatched'
    try:
        address=campus_address()
        with campus_resolution(address):
            code=train(training,native,fit_limit=2,wall=180.,timeout=60.,flash=client)
        status=read(native/'run_outcome.json')['status']
        if code==0:
            dest=root/'delivery'/'predictions.csv'
            replay=root/'replay'/'predictions.csv'
            env=dict(os.environ,SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1')
            for path in (dest,replay):
                subprocess.run([sys.executable,'-m','unified_agent','predict',str(query),'--run-dir',str(native.parent),
                                '--output',str(path)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,check=True,timeout=90)
            if dest.read_bytes()!=replay.read_bytes() or read(dest.parent/'prediction_manifest.json')['prediction_count']!=4:
                raise ValueError('Live format replay failed row or byte verification')
    finally:
        with sqlite3.connect(ledger.path) as db:
            db.row_factory=sqlite3.Row
            rows=[dict(r) for r in db.execute('SELECT * FROM requests ORDER BY rowid')]
            transitions=[dict(r) for r in db.execute('SELECT * FROM transitions ORDER BY seq')]
        usage_path=native/'llm_usage.jsonl'
        usage=[json.loads(s) for s in usage_path.read_text().splitlines()] if usage_path.exists() else []
        calls=[{k:e.get(k) for k in ('budget_request_id','response_id','requested_model','served_model','input_tokens',
               'output_tokens','course_cost_rmb','actual_cost_rmb','event','role')} for e in usage if e.get('budget_request_id')]
        save(root/'pending_merge.json',dict(scope=scope,team_merge_state='pending_team_handoff',
             deduplicate_by='native request UUID; also verify provider response ID and original transitions',
             requests=rows,transitions=transitions,usage=calls,team_ledger_written=False))
        outcome=read(native/'run_outcome.json') if (native/'run_outcome.json').is_file() else {}
        save(root/'receipt.json',dict(status=status,live_api_requests=client.request_count,usage=calls,
             physical_fits=outcome.get('attempted_classifier_fits',0),fresh_process_identical=(root/'replay'/'predictions.csv').is_file()
             and (root/'delivery'/'predictions.csv').read_bytes()==(root/'replay'/'predictions.csv').read_bytes(),
             provisional_ledger=True,team_merge_state='pending_team_handoff',team_total_verified=False))
    return 0 if status=='completed' else 2


if __name__=='__main__':
    raise SystemExit(main())
