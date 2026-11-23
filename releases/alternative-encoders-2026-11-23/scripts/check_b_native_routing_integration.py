"""Use a separately checked-out runtime to replay the routing decision and load the deployed model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def read(f):
    return json.loads(f.read_bytes())


def run(a):
    sys.path.insert(0,str(a.checkout))
    from superagent import ab_encoder_routing
    from superagent.ab_adaptive_opening import action
    from superagent.data import load_development_task
    from superagent.prediction_evidence import fingerprint
    from superagent.splits import sample_groups
    if Path(ab_encoder_routing.__file__).resolve().parents[1]!=a.checkout:
        raise ValueError('Imported the wrong checkout')
    folder=a.study/'autonomous'
    groups=sample_groups(load_development_task(a.inputs/'anonymous.zip'),'shared_text')
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        meta=read(folder/(split+'_split.json'))
        f=folder/(split+'.joblib')
        local=joblib.load(f)
        ids=meta['search_indices']
        if sha(f)!=meta['evaluation_payload_sha256'] or fingerprint(ids)!=meta['row_ids_sha256']:
            raise ValueError('Original training partition changed')
        trusted[split]=dict(row_ids=ids,labels=local['y_search'].tolist(),evaluation_payload_sha256=sha(f),
                            tie_label_one=[l<r for l,r in local['X_search']],groups=groups[ids].tolist())
    caps=read(folder/'run_manifest.json')['capabilities']
    actions=read(folder/'actions.json')
    for i,expected in enumerate(actions,1):
        state=read(folder/f'state_{i}.json')
        if action(state,caps,trusted,cover_untried_encoder=True)!=expected:
            raise ValueError('Target checkout selected a different training action')
    review=read(folder/'confirmation.json')['b_scheduler_review']
    choice=ab_encoder_routing.assess(review,state,trusted,caps)
    frozen=read(a.study/'model_frozen.json')
    if choice!=frozen['routing'] or not choice['adopted'] or sha(folder/'selected_model.joblib')!=frozen['model']:
        raise ValueError('Model or routing choice differs')
    # The prediction child must resolve both the package and its model classes in the target checkout.
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(a.checkout/'scripts/ab_acceptance_guard',a.checkout))),
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SUPERAGENT_OFFLINE='1',AB_B1_OFFLINE='1')
    from superagent.ab_sentence import ALTERNATIVES, DEEP_ENV
    env[DEEP_ENV]=str(a.weights/'all-MiniLM-L12-v2')
    env[ALTERNATIVES['e5']['env']]=str(a.weights/'e5-small-v2')
    a.out.mkdir(parents=True,exist_ok=True)
    csv=a.out/'rows.csv'
    if not csv.exists():
        with (a.out/'prediction.log').open('x') as log:
            subprocess.run([sys.executable,'-m','superagent','predict',str(a.inputs/'features.zip'),
                            '--run-dir',str(folder),'--output',str(csv)],cwd=a.checkout,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    if sha(csv)!=read(a.study/'exports.json')['native'] or read(a.out/'prediction_manifest.json')['test_labels_read']:
        raise ValueError('Target prediction rows differ')
    status=subprocess.check_output(['git','status','--porcelain','--','superagent','tests/test_ab_encoder_routing.py'],
                                   cwd=a.checkout,text=True)
    if status.strip():
        raise ValueError('Target production files changed')
    record=dict(checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.checkout,text=True).strip(),
        actions=len(actions),first=choice['first'],threshold=choice['threshold'],predictions=sha(csv),
        rows=len(csv.read_bytes().splitlines())-1,model=frozen['model'],physical_fits=0,model_api_calls=0)
    receipt=a.out/'receipt.json'
    if not receipt.exists():
        with receipt.open('x') as f:
            json.dump(record,f,indent=2)
    if read(receipt)!=record:
        raise ValueError('Changed integration evidence')
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkout','inputs','study','weights','out'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
