"""Seal goal robustness metadata before the next production-source change."""
import argparse
from pathlib import Path
import shutil
from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);records={}
    names={'prepared':['protocol.json','completion.json','cases.json','inputs.json','preflight/results.json'],
           'execute':['protocol.json','completion.json','results.json','physical_fits.jsonl','preflight/results.json']}
    for phase,files in names.items():
        src=BASE/('runs/a138_goal_prepared' if phase=='prepared' else 'runs/a138_goal_execute')
        if phase=='execute':
            for n in ('accuracy','renamed','f1_pos0','auc_pos0'):
                files.extend(n+'/'+f for f in ['command.json','run/task_spec.json','run/final_selection.json',
                    'run/run_manifest.json','run/history.json','run/actions.json','run/classifier_fits.json'])
        dst=a.output/phase
        for name in files:
            to=dst/name;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/name,to)
        if read(dst/'protocol.json')['source_files']!=hashes(ROOT):raise ValueError('Goal experiment source changed')
        records[phase]={name:sha(dst/name) for name in files}
    if read(a.output/'execute/completion.json')['actual_classifier_fits']!=12:raise ValueError('Unexpected physical fits')
    save(a.output/'archive_receipt.json',dict(status='archived',records=records,actual_classifier_fits=0,
         real_API_requests=0,starting_A_fits=1602,ending_A_fits=1614,task_data_or_models_copied=False))
    print('A138 archived: 18 preflight cases, four autonomous paths, twelve new fits.')


if __name__=='__main__':main()
