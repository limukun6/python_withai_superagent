"""Archive new-task transfer metadata and check immutable history in the real budget ledger."""
import argparse
from pathlib import Path
import shutil

from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.archive_a140_paid import rows

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('prepared','training','evaluation','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);source=hashes(ROOT)
    batches={'prepared':(a.prepared,['protocol.json','results.json','completion.json','registered_rows.json']),
             'training':(a.training,['protocol.json','results.json','completion.json','physical_fits.jsonl',
                 'L0/protocol.json','L0/completion.json','L0/model_frozen.json','L0/command.json',
                 'EC0/protocol.json','EC0/completion.json','EC0/model_frozen.json','EC0/command.json','EC0/current_ledger_owner.json']),
             'evaluation':(a.evaluation,['protocol.json','results.json','completion.json','L0/verification.json','EC0/verification.json'])}
    for arm in ('L0','EC0'):
        batches['training'][1].extend(arm+'/run/'+n for n in ('run_manifest.json','run_outcome.json','history.json',
              'actions.json','classifier_fits.json','task_spec.json','final_selection.json','llm_usage.jsonl'))
    records={}
    for name,(src,files) in batches.items():
        dst=a.output/name
        for n in files:
            to=dst/n;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/n,to)
        if read(dst/'protocol.json')['source_files']!=source:raise ValueError('Batch source drift: '+name)
        records[name]={n:sha(dst/n) for n in files}
    owner=read(a.training/'EC0/current_ledger_owner.json');ledger=Path(owner['ledger_path']);backup=a.training/'private/ledger_before.sqlite'
    if sha(ledger)!=owner['ledger_sha256']:raise ValueError('Live ledger changed since queue completed')
    ledger_receipts={}
    for table in ('requests','transitions'):
        old,current=rows(backup,table),rows(ledger,table)
        if not old<=current:raise ValueError('Original history changed: '+table)
        ledger_receipts[table]=dict(original=len(old),current=len(current),added=len(current-old),original_rows_preserved=True)
    save(a.output/'ledger_preservation.json',ledger_receipts)
    train=read(a.training/'completion.json');evaluation=read(a.evaluation/'results.json')
    save(a.output/'summary.json',dict(status='passed',records=records,new_classifier_fits=train['actual_classifier_fits'],
         new_API_requests=train['real_API_requests'],A_cumulative_fits_after=train['A_cumulative_fits_after'],
         project_after=owner['project_summary'],paths=evaluation['paths'],paired=evaluation['paired'],
         source_files=source,original_ledger_rows_preserved=True,source_training_or_models_copied=False))
    save(a.output/'completion.json',dict(status='archived',actual_classifier_fits=0,real_API_requests=0,
         summary_sha256=sha(a.output/'summary.json')))
    print('New-task transfer receipts archived; data/models/ledger/key excluded.')


if __name__=='__main__':main()
