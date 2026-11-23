"""Replay every native conflict-policy action with the target checkout and re-export its model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib


def get(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a):
    sys.path.insert(0, str(a.checkout))
    from superagent import ab_single_conflict
    from superagent.data import load_development_task
    from superagent.splits import sample_groups
    if Path(ab_single_conflict.__file__).resolve().parents[1] != a.checkout:
        raise ValueError('Unexpected scheduler checkout')
    folder=a.result/'autonomous'
    groups=sample_groups(load_development_task(a.inputs/'anonymous.zip'),'shared_text')
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        meta=get(folder/(split+'_split.json'))
        payload=folder/(split+'.joblib')
        if sha(payload)!=meta['evaluation_payload_sha256']:
            raise ValueError('Training split changed')
        data=joblib.load(payload)
        ids=meta['search_indices']
        trusted[split]=dict(row_ids=ids,labels=data['y_search'].tolist(),groups=groups[ids].tolist(),
                            evaluation_payload_sha256=sha(payload),tie_label_one=[l<r for l,r in data['X_search']])
    caps=get(folder/'run_manifest.json')['capabilities']
    actions=get(folder/'actions.json')
    for i, expected in enumerate(actions,1):
        state=get(folder/f'state_{i}.json')
        if ab_single_conflict.action(state,caps,trusted,cover_untried_encoder=True)!=expected:
            raise ValueError('Target scheduler differs on action '+str(i))
    decision=ab_single_conflict.review(state,trusted)
    if decision!=get(folder/'confirmation.json')['b_scheduler_review']:
        raise ValueError('Target scheduler selected a different model')
    a.out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
             AB_B1_OFFLINE='1', PYTHONPATH=os.pathsep.join(map(str,(a.checkout/'scripts/ab_acceptance_guard',a.checkout))))
    for key, file in [('SUPERAGENT_DEEP_MINILM_WEIGHTS','all-MiniLM-L12-v2'),('SUPERAGENT_E5_SMALL_WEIGHTS','e5-small-v2'),
                      ('SUPERAGENT_MINILM_WEIGHTS','all-MiniLM-L6-v2'),('SUPERAGENT_BGE_SMALL_WEIGHTS','bge-small-en-v1.5')]:
        env[key]=str(a.weights/file)
    predict=a.out/'rows.csv'
    if not predict.exists():
        with (a.out/'predict.log').open('x') as f:
            subprocess.run([sys.executable,'-m','superagent','predict',str(a.inputs/'features.zip'),'--run-dir',str(folder),
                '--output',str(predict)],cwd=a.checkout,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
    if sha(predict)!=get(a.result/'exports.json')['new'] or get(a.out/'prediction_manifest.json')['test_labels_read']:
        raise ValueError('Target checkout did not reproduce the isolated row file')
    receipt=dict(checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.checkout,text=True).strip(),
        actions=len(actions),decision=decision['single_conflict']['status'],selected=decision['members'],
        predictions=sha(predict),rows=len(predict.read_bytes().splitlines())-1,model=get(folder/'final_selection.json')['model_sha256'],
        fits=0,model_api_calls=0)
    out=a.out/'receipt.json'
    if not out.exists():
        with out.open('x') as f:
            json.dump(receipt,f,indent=2)
    if receipt!=get(out):
        raise ValueError('Integration evidence changed')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkout','inputs','result','weights','out'):
        p.add_argument('--'+n,type=lambda x:Path(x).resolve(),required=True)
    run(p.parse_args())
