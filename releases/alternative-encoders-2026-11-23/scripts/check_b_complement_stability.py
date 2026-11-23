"""Resample shared-title groups to measure opening-decision sensitivity without fitting any estimator."""
import argparse
from collections import Counter
from itertools import combinations
import os
from pathlib import Path
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run
from superagent.ab_complement import MIN_GAIN,canonical_probabilities
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


def choose(totals,pairs):
    n=totals[:,-1]
    ranks=totals[:,:4]
    leaders=ranks.argmax(axis=1)
    decisions=[]
    for row,lead in enumerate(leaders):
        candidates=[(j,4+pairs.index(tuple(sorted((lead,j))))) for j in range(4) if j!=lead]
        other,index=max(candidates,key=lambda item:totals[row,item[1]])
        decisions.append(tuple(sorted((int(lead),other))) if (totals[row,index]-ranks[row,lead])/n[row]>=MIN_GAIN else ())
    return leaders,decisions


def audit(folder,group_ids):
    h=read(folder/'history.json')[:4]
    order=[r['candidate'] for r in h]
    split=read(folder/'primary_split.json')
    payload=joblib.load(folder/'primary.joblib')
    digest=ab_run.sha(folder/'primary.joblib')
    y=payload['y_search']
    ids=np.asarray(split['search_indices'])
    g=group_ids[ids]
    unique,local=np.unique(g,return_inverse=True)
    winners=[]
    prob=[]
    first=np.asarray([a<b for a,b in payload['X_search']])
    for r in h:
        art=joblib.load(folder/f"trial_{r['trial']}.joblib")
        rec=art['prediction_record']
        check(r['split_id']=='primary' and r['evaluation_payload_sha256']==digest==rec['evaluation_payload_sha256']
              and fingerprint(ids)==r['row_ids_sha256'] and fingerprint(y)==rec['label_sha256'],'Wrong checkpoint')
        hard=np.asarray(art['search_predictions'])
        check(fingerprint(hard)==rec['prediction_sha256'],'Hard predictions differ')
        p=canonical_probabilities(art['model'],payload['X_search'])[1][:,1]
        winners.append(hard==y)
        prob.append(p)
    pairs=list(combinations(range(4),2))
    for i,j in pairs:
        avg=(prob[i]+prob[j])/2
        winners.append(((avg>.5)|((avg==.5)&first))==y)
    winners.append(np.ones(len(y),dtype=int))
    by_group=np.column_stack([np.bincount(local,weights=w) for w in winners]).astype(int)
    raw=by_group.sum(axis=0,keepdims=True)
    anchor,pick=choose(raw,pairs)
    leads=Counter()
    picks=Counter()
    rng=np.random.default_rng(77891)
    for _ in range(100):
        sample=rng.integers(0,len(unique),size=(50,len(unique)))
        ls,ps=choose(by_group[sample].sum(axis=1),pairs)
        leads.update(int(l) for l in ls)
        picks.update(ps)
    key=lambda pair:' + '.join(order[i] for i in pair) if pair else 'single'
    return dict(seed=split['seed'],rows=len(y),groups=len(unique),draws=5000,
        actual_leader=order[anchor[0]],actual_proposed=key(pick[0]),
        actual_score_counts={order[i]:int(raw[0,i]) for i in range(4)},
        leader_counts={order[i]:v for i,v in leads.items()},portfolio_counts={key(k):v for k,v in picks.items()},
        rate_same_leader=leads[int(anchor[0])]/5000,rate_same_proposal=picks[pick[0]]/5000,
        rate_no_pair=picks[()]/5000)


def run(inputs,earlier,native,out):
    t=load_development_task(inputs/'anonymous.zip')
    gs=sample_groups(t,'shared_text')
    runs={'old_e5':earlier/'deep-e5','old_bge':earlier/'deep-bge','new_e5':native/'autonomous'}
    results={name:audit(folder,gs) for name,folder in runs.items()}
    save(out,dict(results=results,new_fits=0,model_api=0,external_evaluation=False,
        plan=ab_run.sha(ROOT/'docs/batches/B_complement_selection_stability_1123_zh.md')))
    print({k:(v['actual_leader'],v['actual_proposed'],v['rate_same_leader'],v['rate_same_proposal'],
             v['rate_no_pair']) for k,v in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','earlier','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('inputs','earlier','native','output')))
