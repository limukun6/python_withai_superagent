"""Archive compact A21-A25 evidence and verify the completed no-dispatch review, read only.

No model, raw dataset, API key, SQLite file or historical request messages are copied.
This does not rerun training, API calls, or the ledger mutation in A25.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from scripts.a_batch_common import save
from superagent.task_budget import TaskBudgetLedger
from superagent.research_budget import research_preflight


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def archive(runs, output):
    output.mkdir(parents=True, exist_ok=False)
    files = {
        'a21_local_usage_review': ['protocol.json', 'results.json', 'completion.json'],
        'a22_export_contract_checks': ['protocol.json', 'completion.json', 'tests.log'],
        'a23_real_local_history': ['history_audit.json', 'completion.json'],
        'a23_incremental_checks': ['protocol.json', 'completion.json', 'tests.log'],
        'a23_incremental_checks_fixed': ['protocol.json', 'completion.json', 'tests.log'],
        'a23_live_flash_preflight': ['protocol.json', 'completion.json'],
        'a23_live_flash_research': ['protocol.json', 'completion.json', 'results.json'],
        'a24_campus_readonly': ['protocol.json', 'results.json', 'completion.json'],
        'a25_no_dispatch_review': ['proof.json', 'completion.json'],
        'a25_final_checks': ['protocol.json', 'completion.json', 'tests.log'],
    }
    inventory = []
    for folder, names in files.items():
        target = output / folder
        target.mkdir()
        for name in names:
            source = runs / folder / name
            shutil.copy2(source, target / name)
            inventory.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    run = runs / 'a23_live_flash_research' / 'raw-history'
    target = output / 'a23_live_flash_research' / 'raw-history'
    target.mkdir()
    for name in ('run_manifest.json', 'run_outcome.json', 'classifier_fits.json'):
        shutil.copy2(run / name, target / name)
    events_path = run / 'events.jsonl'
    events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines()]
    request = [event for event in events if event.get('kind') == 'llm_request']
    failure = [event for event in events if event.get('kind') == 'llm_transport_failure']
    assert len(request) == len(failure) == 1
    proof = read(runs / 'a25_no_dispatch_review' / 'proof.json')
    assert request[0]['budget_request_id'] == proof['ticket']
    for name, digest in proof['original_files_sha256'].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
    save(target / 'transport_evidence.json', dict(
        request={key: request[0].get(key) for key in ('kind', 'role', 'budget_request_id', 'time_utc', 'model')},
        failure=failure[0], original_events_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
        request_messages_omitted=True))
    ledger_path = runs / 'a23_real_local_history' / 'ledger.sqlite'
    with sqlite3.connect(ledger_path.as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute('SELECT status,course_units,payload FROM requests WHERE id=?', (proof['ticket'],)).fetchone()
        assert row[0] == 'not_dispatched' and row[1] == 0
        assert Path(json.loads(row[2])['run_id']).resolve() == run.resolve()
        historical = read(runs / 'a23_real_local_history' / 'history_audit.json')['calls']
        expected = {'historical:' + call['request_id'] for call in historical if call['cost_status'] == 'unknown'}
        unknown = {row[0] for row in db.execute("SELECT id FROM requests WHERE status='unknown'")}
        assert len(expected) == 11 and expected == unknown
        transitions = [row[0] for row in db.execute('SELECT event FROM transitions WHERE request_id=? ORDER BY seq', (proof['ticket'],))]
        assert 'unknown' in transitions and 'reviewed_not_dispatched' in transitions
    ledger = TaskBudgetLedger(ledger_path)
    save(output / 'review_binding_and_current_ledger.json', dict(
        status='verified_read_only', ticket=proof['ticket'], ledger_run_binding=True,
        original_transport_hashes_unchanged=True, original_eleven_unknown_ids_preserved=True,
        transitions=transitions, project_usage=ledger.project_summary(),
        research_preflight=research_preflight(ledger, 'sms_spam', .60),
        ledger_modified=False, real_new_requests=0, classifier_fits=0))
    save(output / 'archive_manifest.json', dict(files=inventory, raw_data_models_credentials_sqlite_included=False,
        historical_messages_included=False, real_new_requests=0, classifier_fits=0))
    print(json.dumps(dict(status='archived', source_files=len(inventory), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive(args.runs_root.resolve(), args.output.resolve())
