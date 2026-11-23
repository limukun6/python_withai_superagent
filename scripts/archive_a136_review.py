"""Copy the metadata-only A136 receipts byte-for-byte and check their seals."""
import argparse
from pathlib import Path
import shutil
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','results.json','completion.json','verification.log'):
        shutil.copy2(a.input/name,a.output/name)
    c=read(a.output/'completion.json')
    assert c['status']=='passed'
    assert c['protocol_sha256']==sha(a.output/'protocol.json') and c['results_sha256']==sha(a.output/'results.json')
    release=a.input/'releases/bean-task2-2026-11-23'
    save(a.output/'patch_check.json',dict(check='git apply --check',applied=False,status='old_context_conflict',
         conflict_files=['superagent/ab_catalog.py','superagent/ab_methods.py','superagent/ab_run.py','superagent/cli.py'],
         patch_sha256=sha(release/'integration/A_c8a21d6_numeric_kernel.patch'),actual_classifier_fits=0,API_requests=0))
    print('Archived A136 metadata and seals; no model/data/ledger copied.')


if __name__=='__main__':main()
