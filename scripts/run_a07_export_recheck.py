"""Recheck existing three-modality models with current exporter, without new scores/fits."""
import argparse
import json
from pathlib import Path
import sys
from a_batch_common import Journal
from run_a07_delivery_batch import command_run

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--delivery',required=True);p.add_argument('--prepared',required=True)
    a=p.parse_args();j=Journal(a.output,ROOT,{'batch':'A-07-final-export-recheck','classifier_fit_limit':0,'heldout_scoring':False})
    results=[]
    for task in ('bean','sms','digits'):
        folder=j.out/task;folder.mkdir();run=Path(a.delivery)/(task+'_fixed');data=Path(a.prepared)/task;prediction=folder/'predictions.csv'
        command=[sys.executable,'-m','superagent','predict',str(data/'features.zip'),'--run-dir',str(run),'--output',str(prediction)]
        launch=command_run(command,ROOT,folder/'predict.log',60)
        verify=[sys.executable,str(ROOT/'scripts/a07_verify_export.py'),'--source-root',str(ROOT),'--run-dir',str(run),
            '--holdout',str(data/'holdout.joblib'),'--predictions',str(prediction),'--output',str(folder/'verification.json'),'--skip-score']
        check=command_run(verify,ROOT,folder/'verify.log',60) if launch['returncode']==0 else launch
        results.append({'task':task,'passed':check['returncode']==0,'prediction':launch,'verification_process':check,
                        'verification':json.loads((folder/'verification.json').read_text(encoding='utf-8')) if check['returncode']==0 else None})
    j.finish(results,cases=3,passed=sum(r['passed'] for r in results),classifier_fits=0,heldout_scoring=False)
    print(json.dumps(results))


if __name__=='__main__':main()
