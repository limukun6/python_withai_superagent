"""Screen all six pairs in each observed four-method opening with no new training."""
import argparse
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
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check,read
from superagent import ab_run
from superagent.ab_complement import MIN_GAIN,canonical_probabilities
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.score_evidence import checked_score_record


def array(path,payload,digest,name):
    artifact=joblib.load(path)
    rec=artifact['prediction_record']
    check(rec['evaluation_payload_sha256']==digest and rec['label_sha256']==fingerprint(payload['y_search'])
          and rec['prediction_sha256']==fingerprint(artifact['search_predictions']),'Mismatched checkpoint')
    if rec.get('score_record'):
        m,order=checked_score_record(rec['score_record'],rec,MetricSpec.from_payload(payload),name,payload['y_search'])
        p=m[:,order.index(1)]
    else:
        p=canonical_probabilities(artifact['model'],payload['X_search'])[1][:,1]
    return artifact['search_predictions'],p


def hits(left,right,payload):
    p=(left[1]+right[1])/2
    first=np.array([a<b for a,b in payload['X_search']])
    return int((((p>.5)|((p==.5)&first))==payload['y_search']).sum())


def experiment(label,folder,siblings,extras):
    history=read(folder/'history.json')
    opening=history[:4]
    names=[r['candidate'] for r in opening]
    primary=folder/'primary.joblib'
    payload=joblib.load(primary)
    fingerprint0=ab_run.sha(primary)
    cols={r['candidate']:array(folder/f"trial_{r['trial']}.joblib",payload,fingerprint0,r['candidate']) for r in opening}
    y=payload['y_search']
    singles={name:int((pred==y).sum()) for name,(pred,_) in cols.items()}
    anchor=max(singles,key=singles.get)
    votes={pair:hits(cols[pair[0]],cols[pair[1]],payload) for pair in combinations(names,2)}
    best=max(votes,key=lambda p:(votes[p],min(singles[n] for n in p),max(singles[n] for n in p)))
    previous=max((p for p in votes if anchor in p),key=votes.get)
    record=dict(arm=label,opening=singles,all_primary_correct=[dict(pair=p,correct=v) for p,v in votes.items()],
        anchor=anchor,original_first_tie=previous,selected=best,selected_includes_anchor=anchor in best,
        restricted_primary_correct=votes[previous],unrestricted_primary_correct=votes[best],primary_rows=len(y))
    if (votes[best]-singles[anchor])/len(y)<MIN_GAIN:
        return dict(**record,decision='keep_single_before_confirmation',passed=False)
    support=list(dict.fromkeys([anchor,*best]))
    confirmation=folder/'confirmation_1.joblib'
    digest=ab_run.sha(confirmation)
    sources={}
    for directory in [folder,*siblings]:
        if ab_run.sha(directory/'confirmation_1.joblib')!=digest:
            continue
        for trial in read(directory/'history.json'):
            name=trial['candidate']
            if trial['split_id']=='confirmation_1' and name in support and name not in sources:
                sources[name]=(directory/f"trial_{trial['trial']}.joblib",'original' if folder==directory else 'matched sibling')
    for name,src in extras:
        if name in support and name not in sources:
            sources[name]=(src,'supplemental counterfactual')
    gaps=[n for n in support if n not in sources]
    if gaps:
        return dict(**record,decision='missing_matched_confirmation',missing=gaps,passed=None)
    payload2=joblib.load(confirmation)
    y2=payload2['y_search']
    confirmed={n:array(src,payload2,digest,n) for n,(src,_) in sources.items()}
    single2={n:int((v[0]==y2).sum()) for n,v in confirmed.items()}
    vote2=hits(confirmed[best[0]],confirmed[best[1]],payload2)
    net=votes[best]+vote2-max(singles[n]+single2[n] for n in support)
    passed=votes[best]>max(singles[n] for n in support) and vote2>max(single2.values()) and net/(len(y)+len(y2))>=MIN_GAIN
    return dict(**record,decision='promote_pair' if passed else 'retain_incumbent',passed=passed,
        confirmation_singles=single2,confirmation_vote=vote2,confirmation_rows=len(y2),pooled_gain=net,
        confirmation_sources={n:v[1] for n,v in sources.items()})


def run(old,full,native,ties,materialized,out):
    out.mkdir(parents=True,exist_ok=True)
    bgeinfo=read(full/'fits.json')['deep-bge']
    extra=[(bgeinfo['candidate'],full/'deep-bge/worker.joblib')]
    tiedname=read(ties/'protocol.json')['missing']
    checks={}
    for label,folder,siblings,e in [('old_e5',old/'deep-e5',[old/'deep-bge'],extra),
        ('old_bge',old/'deep-bge',[old/'deep-e5'],extra),
        ('new_e5',native/'autonomous',[],[(tiedname,ties/'extra_confirmation.joblib')])]:
        checks[label]=experiment(label,folder,siblings,e)
    frozen_save(out/'primary_and_confirmation.json',checks)
    # Resolve existing, same-input full deployments only after freezing the inner decisions.
    deep=read(old/'deep-e5_receipt.json')['selected']
    e5=read(full/'fits.json')['deep-e5']['candidate']
    table={frozenset((deep,e5)):read(full/'results.json')['reserve']['soft_vote']['correct'],
           frozenset((deep,bgeinfo['candidate'])):read(materialized/'results.json')['reserve']['bge_vote']['correct'],
           frozenset((deep,)):read(full/'results.json')['reserve']['deep']['correct']}
    latest=read(ties/'results.json')
    table[frozenset([latest['anchor']])]=latest['results']['anchor']['versus_anchor']['correct']
    for name,key in read(ties/'models_frozen.json')['pairs'].items():
        table[frozenset((latest['anchor'],name))]=latest['results'][key]['versus_anchor']['correct']
    selected={}
    for k,r in checks.items():
        members=r['selected'] if r['passed'] else [r['anchor']] if r['passed'] is False else []
        selected[k]=dict(inner_decision=r['decision'],members=members,matched_reserve_correct=table.get(frozenset(members)))
    frozen_save(out/'results.json',dict(results=selected,decision_seal=ab_run.sha(out/'primary_and_confirmation.json'),
        new_physical_fits=0,model_api=0,formal_test=False))
    print(selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','old','full','native','ties','materialized','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('old','full','native','ties','materialized','output')))
