"""Verify private B SQLite return and transport receipts without exposing request payloads or sending HTTP."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path,PurePosixPath
import sqlite3
import time
import zipfile

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
EXPECTED='0aaedaaaaa7814ab9ad1f5f7e4bc7d3dd26faba260f936c5ec7abc829c5d4729'
REQUEST='93c46fb2-af8f-4c1e-b2f3-cd019a6e2d68'


def readonly(path):
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    return db


def table(db,name):return [dict(r) for r in db.execute('SELECT * FROM '+name)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);private=a.output/'private';private.mkdir();start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A129',archive_sha256=sha(a.archive),expected_ledger_sha256=EXPECTED,
        runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A129_B_ledger_intake_plan_20261123_zh.md'),
        maximum_classifier_fits=0,real_API_requests=0,request_payloads_logged=False))
    try:
        check=subprocess_check_ignore(private)
        with zipfile.ZipFile(a.archive) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive member')
            for name in names:
                parts=PurePosixPath(name).parts
                if PurePosixPath(name).is_absolute() or '..' in parts or '\\' in name:raise ValueError('Unsafe archive member')
            sums={}
            for line in archive.read('SHA256SUMS.txt').decode('utf-8-sig').splitlines():
                digest,name=line.split(None,1);name=name.strip();raw=archive.read(name)
                if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Archive checksum differs: '+name)
                sums[name]=digest
            if set(names)!={*sums,'SHA256SUMS.txt'}:raise ValueError('Unregistered archive content')
            for name in names:
                target=private/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
        ledger=private/'ledger.sqlite'
        if sha(ledger)!=EXPECTED:raise ValueError('Returned ledger byte hash mismatch')
        with readonly(ledger) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity check failed')
            requests=table(db,'requests');transitions=table(db,'transitions')
        statuses=Counter(r['status'] for r in requests)
        known=sum(r['course_units'] or 0 for r in requests)/1e9
        if (len(requests),statuses['unknown'],statuses['reserved'],known)!=(357,12,0,6.7114475):
            raise ValueError('Returned actual ledger summary differs')
        old_path=BASE/'runs/a23_real_local_history/ledger.sqlite'
        with readonly(old_path) as old:
            old_requests=table(old,'requests');old_transitions=table(old,'transitions')
        new_index={r['id']:r for r in requests}
        if len(old_requests)!=335 or any(new_index.get(r['id'])!=r for r in old_requests):raise ValueError('Original requests changed')
        def canonical(row):return json.dumps(row,sort_keys=True,separators=(',',':'))
        new_events={canonical(r) for r in transitions}
        if any(canonical(r) not in new_events for r in old_transitions):raise ValueError('Original transitions changed')
        failed=new_index[REQUEST]
        evidence=json.loads((private/'bean_E1/request_and_transport.json').read_text(encoding='utf-8'))
        if (failed['status']!='unknown' or failed['course_units'] is not None or evidence['budget_request_id']!=REQUEST or
            evidence['request']['budget_request_id']!=REQUEST or evidence['persistent_status']!='unknown' or
            evidence['transport']['http_status']!=200 or evidence['transport']['data_events']!=286 or
            evidence['final_provider_usage_received'] or any(evidence[k] is not None for k in ('input_tokens','output_tokens','course_cost_rmb'))):
            raise ValueError('Unresolved request/transport evidence differs')
        if sha(ledger)!=EXPECTED:raise ValueError('Read-only audit modified ledger')
        summary=dict(requests=357,known_course_cost_rmb=known,unresolved_requests=12,pending_requests=0,
            complete_course_cost_rmb=None,actual_account_debit_rmb=None)
        owner=dict(owner='A',window_status='paused_new_unknown',ledger_sha256=EXPECTED,project_summary=summary,
            ledger_path=str(ledger),B_stopped_writing='User-delivered B handoff statement',new_paid_calls_permitted=False)
        save(a.output/'ledger_owner.json',owner)
        result=dict(archive_files_verified=sums,ledger_sha256=EXPECTED,sqlite_integrity='ok',
            project_summary=summary,old_requests_preserved=len(old_requests),old_transitions_preserved=len(old_transitions),
            actual_new_requests=22,unknown_request_id=REQUEST,http_status=200,data_events=286,
            final_usage_received=False,new_unknown_resolved=False,new_paid_calls_permitted=False,
            private_files_ignored=check,actual_SQLite_received=True,old_335_copy_obsolete=True)
        save(a.output/'results.json',result)
        save(a.output/'completion.json',dict(status='verified_owned_by_A_paid_paused',actual_classifier_fits=0,
            real_API_requests=0,elapsed_seconds=time.monotonic()-start))
        print(json.dumps({k:result[k] for k in ('ledger_sha256','project_summary','old_requests_preserved','old_transitions_preserved','http_status','data_events','new_unknown_resolved','new_paid_calls_permitted')}),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped_no_handoff',error=str(error),actual_classifier_fits=0,real_API_requests=0));raise


def subprocess_check_ignore(path):
    import subprocess
    result=subprocess.run(['git','check-ignore','-q',str(path)],cwd=BASE)
    if result.returncode:raise ValueError('Private intake directory must be ignored by its owning repository')
    return True


if __name__=='__main__':main()
