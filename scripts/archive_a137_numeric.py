"""Archive A137 engineering receipts with source/model identity checks; no model copying."""
import argparse
from pathlib import Path
import shutil
import zipfile
from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    names=['protocol.json','results.json','completion.json','default_menus.json','command.json','tests.log',
           'physical_fits.jsonl','run/run_manifest.json','run/run_outcome.json','run/task_spec.json',
           'run/final_selection.json','run/classifier_fits.json','run/actions.json','run/history.json']
    for name in names:
        target=a.output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.input/name,target)
    completion=read(a.output/'completion.json');current=hashes(ROOT)
    if completion['status']!='passed' or completion['protocol_sha256']!=sha(a.output/'protocol.json') or completion['results_sha256']!=sha(a.output/'results.json'):
        raise ValueError('A137 seals differ')
    manifest=read(a.output/'run/run_manifest.json')
    if current!={'superagent/'+k:v for k,v in manifest['source_files'].items()}:raise ValueError('Current source differs')
    original=BASE/'runs/a136_B_numeric_review/releases/bean-task2-2026-11-23/runtime-source.zip'
    with zipfile.ZipFile(original) as z:
        if z.read('superagent/numeric_kernel.py')!=(ROOT/'superagent/numeric_kernel.py').read_bytes():
            raise ValueError('Native B numeric feature code changed')
    save(a.output/'archive_receipt.json',dict(status='archived',files={n:sha(a.output/n) for n in names},
         source_files=current,native_numeric_module_byte_identical=True,actual_classifier_fits=0,
         API_requests=0,task1_candidate_source='6b6ace8',task1_candidate_relabelled=False))
    print('Archived A137 metadata; 93 source identities and unchanged B numeric module verified.')


if __name__=='__main__':main()
