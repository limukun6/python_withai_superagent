"""Re-score frozen models and exercise the normal predict CLI using validation-only features."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import joblib
import numpy as np
from released_validation import TASKS,ARMS,SRC,read,sha,validation,allow_only
from superagent.models import predict_with_model

def verify(packages):
    scores=read(SRC/'validation/scores.json')
    protocol=read(SRC/'validation/protocol.json')
    for name,value in protocol['hashes']['core'].items():assert sha(SRC/name)==value
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        for task,zipname in TASKS.items():
            with allow_only({'C_val.npy','y_val.npy','val.csv'},folder/'audit.jsonl','verification'):
                inputs,labels=validation(packages/zipname)
            stage=folder/(task+'.zip')
            with zipfile.ZipFile(stage,'w') as archive:
                if task=='task1':
                    array=io.BytesIO();np.save(array,inputs,allow_pickle=True)
                    archive.writestr('C_test.npy',array.getvalue())
                else:
                    table=io.StringIO();writer=csv.writer(table);writer.writerow(['headline_A','headline_B']);writer.writerows(inputs)
                    archive.writestr('test.csv',table.getvalue())
            for arm in ARMS:
                data=SRC/'solutions'/task/arm
                path=data/'selected_model.joblib'
                row=scores[task]['arms'][arm]
                assert sha(path)==row['model_sha256']
                bundle=joblib.load(path);expected=predict_with_model(bundle['kind'],bundle['model'],inputs)
                assert len(labels)==scores[task]['rows'] and int((expected==labels).sum())==row['correct']
                dest=folder/task/arm/'predictions.csv'
                subprocess.run([sys.executable,'-m','superagent','predict',str(stage),'--run-dir',str(data),
                    '--output',str(dest)],cwd=SRC,check=True,capture_output=True)
                assert sha(dest)==sha(data/'validation_predictions.csv')==row['prediction_sha256']
                print(task,arm,str(row['correct'])+'/'+str(len(labels)),'predictions verified')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--packages',type=Path,required=True)
    verify(parser.parse_args().packages.resolve())
