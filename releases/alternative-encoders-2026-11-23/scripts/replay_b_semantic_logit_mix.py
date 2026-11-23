"""Decompose frozen linear preference heads and test a registered semantic-only mixture."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_encoder_agreement import compare, matched_runs, SPLITS
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check,read
from superagent import ab_run
from superagent.prediction_evidence import fingerprint

MODELS = ('deep','e5','bge')


def calibration_rows(rows):
    unique={tuple(pair) for pair in rows}
    def key(pair):
        message=json.dumps(pair,ensure_ascii=False,separators=(',',':')).encode()
        return hashlib.sha256(message).digest(),pair
    return sorted(unique,key=key)[:512]


def components(model,rows):
    features = model.named_steps['features']
    head = model.named_steps['classifier']
    check(tuple(head.classes_)==(0,1),'Class orientation')
    x = features.transform(rows)
    pos=len(features.word_.vocabulary_)+len(features.char_.vocabulary_)+features.scaler_.n_features_in_
    check(x.shape[1]==pos+384 and head.coef_.shape==(1,pos+384),'Unexpected feature layout')
    non=np.asarray(x[:,:pos]@head.coef_[0,:pos]).ravel()+head.intercept_[0]
    semantic=np.asarray(x[:,pos:]@head.coef_[0,pos:]).ravel()
    np.testing.assert_allclose(non+semantic,head.decision_function(x),atol=1e-11,rtol=1e-11)
    return non,semantic,head.predict(x)


def describe(parts,scales,rows):
    first=np.array([a<b for a,b in rows])
    means=sum(parts[k][1]*scales['deep']/scales[k] for k in MODELS)/3
    margin=parts['deep'][0]+means
    chosen=((margin>0)|((margin==0)&first)).astype(int)
    descriptives={}
    for k,(lex,sem,hard) in parts.items():
        descriptives[k]=dict(semantic_rms=float(np.sqrt(np.mean(sem**2))),nonsemantic_rms=float(np.sqrt(np.mean(lex**2))),
            semantic_changes_decision=int((hard!=(lex>0)).sum()),rows=len(rows))
    return chosen,descriptives


def run(inputs,control,batch,out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=read(batch/'protocol.json')
    records=matched_runs(control,batch)
    check(ab_run.sha(inputs/'features.zip')==protocol['identities']['packets']['features.zip'],'Reserve features changed')
    with zipfile.ZipFile(inputs/'features.zip') as z,z.open('rows.csv') as s:
        reserve=[(r['option_a'],r['option_b']) for r in csv.DictReader(io.TextIOWrapper(s,encoding='utf8'))]
    inner={}
    reserve_predictions=reserve_stats=None
    sources={}
    for split in SPLITS:
        if not all((split,k) in records for k in MODELS):
            continue
        first_dir=records[split,'deep'][0]
        expected=protocol['splits'][split]['evaluation_payload_sha256']
        check(ab_run.sha(first_dir/(split+'.joblib'))==expected,'Wrong input')
        payload=joblib.load(first_dir/(split+'.joblib'))
        calib=calibration_rows(payload['X_train'])
        check(len(calib)==512,'Calibration slice too small')
        held=payload['X_search']
        scales,parts,extra={},{},{}
        for k in MODELS:
            folder,trial,seal=records[split,k]
            check(seal['payload']==expected and fingerprint(payload['y_search'])==seal['labels'],'Split identity mismatch')
            file=folder/f'trial_{trial}.joblib'
            sources[split+'/'+k]=ab_run.sha(file)
            model=joblib.load(file)['model']
            _,cal_scores,_=components(model,calib)
            scale=float(np.sqrt(np.mean(cal_scores**2)))
            check(np.isfinite(scale) and scale>0,'Unavailable semantic scale')
            scales[k]=scale
            parts[k]=components(model,held)
            check(fingerprint(parts[k][2])==seal['predictions'],'Single checkpoint predictions changed')
            if split=='primary':
                extra[k]=components(model,reserve)
        choice,desc=describe(parts,scales,held)
        inner[split]=dict(training_only_semantic_rms=scales,parts=desc,
            fixed_mix=compare(choice,parts['deep'][2],payload['y_search']))
        if split=='primary':
            mixed,reserve_stats=describe(extra,scales,reserve)
            reserve_predictions={k:x[2].tolist() for k,x in extra.items()}
            reserve_predictions['fixed_mix']=mixed.tolist()
    check(reserve_predictions is not None,'Missing matched primary models')
    frozen_save(out/'contributors.json',sources)
    frozen_save(out/'reserve_predictions.json',reserve_predictions)
    frozen_save(out/'reserve_seal.json',dict(predictions=ab_run.sha(out/'reserve_predictions.json'),
        contributors=ab_run.sha(out/'contributors.json'),inputs=ab_run.sha(inputs/'features.zip'),
        protocol=ab_run.sha(ROOT/'docs/batches/B_semantic_logit_protocol_1123_zh.md')))
    check(all(ab_run.sha(inputs/n)==protocol['identities']['packets'][n] for n in ('scoring_only.json','groups_only.json')),
          'Reserve score or group identity mismatch')
    y=np.array(read(inputs/'scoring_only.json')['labels'])
    groups=np.array(read(inputs/'groups_only.json'))
    baseline=np.asarray(reserve_predictions['deep'])
    mixed=np.asarray(reserve_predictions['fixed_mix'])
    check(len(groups)==len(y)==len(baseline)==1693,'Reserve rows')
    result=dict(inner=inner,reserve_parts=reserve_stats,reserve=compare(mixed,baseline,y),
        reserve_delta=paired((mixed==y).astype(int)-(baseline==y),groups),
        common_partitions=len(inner),seal=ab_run.sha(out/'reserve_seal.json'),classifier_fits=0,model_api=0)
    frozen_save(out/'results.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','control','batch','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('inputs','control','batch','output')))
