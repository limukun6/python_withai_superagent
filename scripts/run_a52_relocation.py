"""Four preselected new-structure models moved and predicted under fit/network guards."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('course','native','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic();results=[]
    save(root/'protocol.json',dict(batch='A52-relocation',source_files=frozen,runner_sha256=sha(Path(__file__)),
        rule='Fixed arm 11 from each task, before scores',classifier_fits_forbidden=True,network_forbidden=True,external_labels_opened=False))
    for parent in (args.course,args.native):
        if read(parent/'completion.json')['status']!='completed' or read(parent/'protocol.json')['source_files']!=frozen:
            raise ValueError('Completed same-source live studies required')
    for task in ('coupon','preference','sms','photo'):
        parent=args.course if task in ('coupon','preference') else args.native
        original=parent/(task+'_11');selected=read(original/'final_selection.json');destination=root/task;destination.mkdir()
        if sha(original/selected['model_file'])!=selected['model_sha256']:raise ValueError('Model hash differs')
        for name in ('task_spec.json','final_selection.json',selected['model_file']):shutil.copy2(original/name,destination/name)
        prediction=destination/'predictions/rows.csv'
        with (root/(task+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run([sys.executable,'-m','scripts.a38_predict_guard',str(parent/'evaluation'/task/'features.zip'),
                '--run-dir',str(destination),'--output',str(prediction)],stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
        if code or prediction.read_bytes()!=(original/'predictions/rows.csv').read_bytes():raise ValueError('Relocated prediction mismatch')
        with prediction.open(encoding='utf-8',newline='') as file:rows=sum(1 for _ in csv.DictReader(file))
        results.append(dict(task=task,arm='11',rows=rows,model_sha256=selected['model_sha256'],predictions_sha256=sha(prediction),relocated_identical=True))
    save(root/'results.json',results);save(root/'completion.json',dict(status='passed',models=4,actual_classifier_fits=0,real_api_requests=0,
        source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,external_labels_opened=False))
    print(json.dumps(read(root/'completion.json')))
