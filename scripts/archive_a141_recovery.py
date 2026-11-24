"""Archive recovery metadata only, excluding checkpoint models, arrays and prediction data."""
import argparse
from pathlib import Path
import shutil
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    names=['protocol.json','results.json','completion.json']
    for c in read(a.input/'results.json')['cases']:
        name=c['case']
        names.extend(name+'/recovered/'+n for n in ('recovery_protocol.json','recovery_completion.json'))
        for n in ('inherited_resources.json','final_selection.json','task_spec.json'):
            if (a.input/name/'recovered'/n).is_file():names.append(name+'/recovered/'+n)
    for name in names:
        dest=a.output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.input/name,dest)
    end=read(a.output/'completion.json')
    if end['status']!='passed' or end['results_sha256']!=sha(a.output/'results.json'):
        raise ValueError('Recovery experiment seal differs')
    if read(a.output/'protocol.json')['source_files']!=hashes(ROOT):raise ValueError('Production source changed')
    save(a.output/'archive_receipt.json',dict(status='archived',files={n:sha(a.output/n) for n in names},
         actual_classifier_fits=0,real_API_requests=0,A_cumulative_fits_after=1614,
         task_data_models_weights_ledger_copied=False))
    print('A141 nine recovery checks archived; zero new fits/API calls.')


if __name__=='__main__':main()
