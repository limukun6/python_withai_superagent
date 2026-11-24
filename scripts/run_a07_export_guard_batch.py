"""Frozen-model export contract fault probes; no fitting or label evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import numpy as np
from a_batch_common import Journal,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();sys.path.insert(0,str(Path(a.source_root).resolve()))
    from superagent.predict import create_predictions
    from superagent.data import load_prediction_features
    original=Path(a.run_dir);spec=json.loads((original/'task_spec.json').read_text(encoding='utf-8'))
    n=len(load_prediction_features(a.features,'image',spec));domain=spec['metric_spec']['labels'];k=len(domain)
    cases=[('valid',True,'accuracy',np.full(n,domain[0]),None,None),
           ('short',False,'accuracy',np.full(n-1,domain[0]),None,None),
           ('unknown_label',False,'accuracy',np.full(n,999),None,None),
           ('matrix_prediction',False,'accuracy',np.full((n,1),domain[0]),None,None),
           ('nonfinite',False,'accuracy',np.full(n,np.nan),None,None),
           ('score_rows',False,'roc_auc',np.full(n,domain[0]),domain,np.full((n-1,k),1/k)),
           ('score_domain',False,'roc_auc',np.full(n,domain[0]),[999,*domain[1:]],np.full((n,k),1/k)),
           ('score_normalization',False,'roc_auc',np.full(n,domain[0]),domain,np.full((n,k),.9)),
           ('score_permuted_valid',True,'roc_auc',np.full(n,domain[0]),list(reversed(domain)),np.full((n,k),1/k))]
    j=Journal(a.output,a.source_root,{'batch':'A-07-export-guard','cases':[{'name':c[0],'valid':c[1]} for c in cases],
        'scope':'controlled output faults, existing frozen model, no training; AUC contract copy is a fixture','classifier_fit_limit':0})
    results=[]
    if not a.validate_only:
        for name,valid,metric,pred,classes,scores in cases:
            folder=j.out/name;folder.mkdir();run=folder/'run';run.mkdir()
            for file in ('selected_model.joblib','final_selection.json','task_spec.json'):shutil.copyfile(original/file,run/file)
            task=json.loads((run/'task_spec.json').read_text(encoding='utf-8'));task['metric']=metric;task['metric_spec']['name']=metric
            save(run/'task_spec.json',task)
            selection=json.loads((run/'final_selection.json').read_text(encoding='utf-8'))
            selection['task_spec_sha256']=hashlib.sha256((run/'task_spec.json').read_bytes()).hexdigest();save(run/'final_selection.json',selection)
            output=folder/'predictions.csv'
            try:
                with patch('superagent.predict.predict_with_model',return_value=pred),patch('superagent.predict.probability_with_model',return_value=(np.asarray(classes),scores)):
                    create_predictions(a.features,run,output)
                record={'case':name,'valid':valid,'accepted':True,'passed':valid}
            except (ValueError,IndexError,TypeError) as e:
                record={'case':name,'valid':valid,'accepted':False,'error':str(e),'passed':not valid and not output.exists()}
            record['csv_created']=output.exists();results.append(record)
    j.finish(results,cases=len(results),passed=sum(r['passed'] for r in results),classifier_fits=0)
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
