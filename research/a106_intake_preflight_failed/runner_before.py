"""Record received versus reported B artifacts and prepare an ignored, consistent real-ledger handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha
from superagent.task_budget import TaskBudgetLedger

ROOT=Path(__file__).resolve().parents[1]
NEW='9ed50e605a1bb19cd990b0c52b550b4fd407a2b7'


def request_digest(path):
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT id,scope,status,reserved_units,course_units,payload FROM requests ORDER BY id').fetchall()
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bridge','bundle','ledger','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A106',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A106_B_reply_intake_plan.md'),
        B_reported_commit=NEW,B_reported_base='9c15e0e',actual_classifier_fits=0,real_API_requests=0,
        inputs=dict(bridge_sha256=sha(a.bridge),bundle_sha256=sha(a.bundle))))
    try:
        commit=subprocess.run(['git','cat-file','-t',NEW],cwd=ROOT,capture_output=True,text=True)
        refs=subprocess.check_output(['git','bundle','list-heads',str(a.bundle)],cwd=ROOT,text=True).splitlines()
        private=a.output/'private';private.mkdir();snapshot=private/'ledger.sqlite'
        ignored=subprocess.run(['git','check-ignore',str(snapshot)],cwd=ROOT,capture_output=True)
        if ignored.returncode!=0:raise ValueError('Refuse private ledger destination outside ignored area')
        ledger=a.ledger.resolve();before=TaskBudgetLedger(ledger).project_summary()
        if before['pending_requests']:raise ValueError('Pending requests prevent handoff snapshot')
        original_digest=request_digest(ledger)
        with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as source:
            with sqlite3.connect(snapshot) as destination:source.backup(destination)
        after=TaskBudgetLedger(ledger).project_summary();copied=TaskBudgetLedger(snapshot).project_summary()
        if before!=after or before!=copied or original_digest!=request_digest(ledger) or original_digest!=request_digest(snapshot):
            raise ValueError('Ledger changed or backup differs; do not transfer')
        result=dict(B_reported_commit=NEW,new_commit_available=commit.returncode==0,
            received_bundle_refs=refs,received_bundle_contains_new_tip=any(line.split()[0]==NEW for line in refs),
            source_artifacts_received=False,review_status='Reported outcomes awaiting original new bundle',
            B_reported_cases=dict(original_pass=45,original_total=48,disputed_input_validity_count=4,
                remaining_reported_pass=41,remaining_reported_total=44,posthoc_replacements_pass=4,
                A_adjudicated=False),
            B_reported_annotations=dict(total=24,factual_error=6,no_error_found=8,insufficient_evidence=10,
                annotator='One B-side Codex; no human review reported',A_verified=False),
            ledger_handoff=dict(summary=copied,snapshot_sha256=sha(snapshot),request_rows_sha256=original_digest,
                request_rows_equal=True,source_unchanged_during_backup=True,state='prepared_not_delivered',
                private_file='runs/a106_B_reply_intake/private/ledger.sqlite',
                writer_assignment='A idle; B must acknowledge receipt and sole writer before either resumes paid calls'))
        save(a.output/'results.json',result)
        save(a.output/'completion.json',dict(status='awaiting_new_B_bundle',ledger_snapshot_prepared=True,
            actual_classifier_fits=0,real_API_requests=0,elapsed_seconds=time.monotonic()-start))
        print(dict(new_commit_available=result['new_commit_available'],received_bundle_refs=refs,
            ledger_summary=copied,ledger_snapshot_prepared=True,ownership_transferred=False))
    except Exception as e:
        save(a.output/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise
