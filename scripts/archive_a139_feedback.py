"""Archive metadata-only feedback projection evidence before further development."""
import argparse
from pathlib import Path
import shutil
from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    names=['protocol.json','results.json','completion.json','tests.log','frozen_inputs.json']
    for task in ('coupon','sms','flowers'):
        names.extend(task+'/'+n for n in ('state.json','capabilities.json','raw.json','full.json','compact.json'))
    for name in names:
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.input/name,target)
    if read(a.output/'protocol.json')['source_files']!=hashes(ROOT):raise ValueError('Projection source changed')
    save(a.output/'archive_receipt.json',dict(status='archived',files={n:sha(a.output/n) for n in names},
         actual_classifier_fits=0,real_API_requests=0,data_models_ledger_copied=False))
    print('Feedback projection metadata archived with exact source identity.')


if __name__=='__main__':main()
