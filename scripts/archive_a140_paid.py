"""Archive sealed paid results and verify the original ledger rows were preserved."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3

from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def rows(path,table):
    with sqlite3.connect('file:'+Path(path).resolve().as_posix()+'?mode=ro',uri=True) as db:
        return set(db.execute('SELECT * FROM '+table).fetchall())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=read(a.run/'completion.json')
    if c['status']!='answers_sealed' or read(a.run/'protocol.json')['source_files']!=hashes(ROOT):raise ValueError('Source/completion differs')
    owner=read(a.run/'current_ledger_owner.json');ledger=Path(owner['ledger_path'])
    if sha(ledger)!=owner['ledger_sha256']:raise ValueError('Actual ledger changed after sealed batch')
    before=a.run/'private/ledger_before.sqlite';checks={}
    for table in ('requests','transitions'):
        old,current=rows(before,table),rows(ledger,table)
        if not old<=current:raise ValueError('Historical '+table+' rows changed')
        checks[table]=dict(original=len(old),current=len(current),added=len(current-old),original_preserved=True)
    if checks['requests']['added']!=9:raise ValueError('Expected exactly nine new ledger requests')
    names=['protocol.json','results.json','completion.json','answers_sealed.json','current_ledger_owner.json']
    for r in read(a.run/'answers_sealed.json'):
        names.extend(r['task']+'/'+r['view']+'/'+n for n in ('answer.json','contract.json','sealed_answer.json','llm_usage.jsonl'))
    for name in names:
        target=a.output/'paid'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.run/name,target)
    evaluation=a.output/'evaluation';evaluation.mkdir()
    for path in a.evaluation.glob('*.json'):shutil.copy2(path,evaluation/path.name)
    save(a.output/'ledger_preservation.json',checks)
    save(a.output/'archive_receipt.json',dict(status='archived',source_files=hashes(ROOT),
         paid_source_completion_sha256=sha(a.run/'completion.json'),known_incremental_cost_rmb=.068634,
         original_requests_preserved=True,original_transitions_preserved=True,
         actual_classifier_fits=0,real_API_requests=0,paid_calls_referenced=9,ledger_or_key_copied=False))
    print(json.dumps(dict(status='archived',project_summary=owner['project_summary'],historical_rows_preserved=True)))


if __name__=='__main__':main()
