"""Replay a complement-aware scheduler from measured, aligned inner predictions."""
import argparse
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
from superagent.models import probability_with_model
from superagent.prediction_evidence import fingerprint

MIN_GAIN=.002


def credit(scores,y):
    return int(np.count_nonzero(scores==y))


def mean_labels(left,right,first):
    avg=(left+right)/2
    return ((avg>.5)|((avg==.5)&first)).astype(int)


def propose(primary,y,first,remaining):
    names=list(primary)
    anchor=max(names,key=lambda k:credit(primary[k][0],y))
    best=credit(primary[anchor][0],y)
    scores={name:credit(mean_labels(primary[anchor][1],primary[name][1],first),y)
            for name in names if name!=anchor}
    partner=max(scores,key=scores.get)
    if remaining<4 or (scores[partner]-best)/len(y)<MIN_GAIN:
        partner=None
    return anchor,partner,scores


def columns(artifact,payload,expected):
    a=joblib.load(artifact)
    y=payload['y_search']
    rec=a['prediction_record']
    check(rec['evaluation_payload_sha256']==expected and rec['label_sha256']==fingerprint(y),'Incompatible payload')
    labels,prob=probability_with_model('pair_text',a['model'],payload['X_search'])
    check(set(labels)=={0,1} and prob.shape==(len(y),2),'Invalid probabilities')
    hard=np.asarray(a['search_predictions'])
    check(fingerprint(hard)==rec['prediction_sha256'] and np.array_equal(labels[np.argmax(prob,axis=1)],hard),
          'Hard predictions or class order changed')
    return hard,prob[:,list(labels).index(1)]


def ranking(rows, splits):
    single=[sum(x) for x in zip(*(r['singles'] for r in rows))]
    pooled=sum(r['mean'] for r in rows)-max(single)
    passed=all(r['mean']>max(r['singles']) for r in rows) and pooled/splits>=MIN_GAIN
    return pooled,passed


def replay(arm,batch,followup):
    folder=batch/arm
    hist=read(folder/'history.json')
    opening=hist[:4]
    check(all(row['split_id']=='primary' and row['status']=='completed' for row in opening),'No comparable opening')
    payload=joblib.load(folder/'primary.joblib')
    first=np.array([l<r for l,r in payload['X_search']])
    y=payload['y_search']
    primary={row['candidate']:columns(folder/f"trial_{row['trial']}.joblib",payload,row['evaluation_payload_sha256']) for row in opening}
    anchor,partner,gains=propose(primary,y,first,11-len(opening))
    record=dict(opening=list(primary),anchor=anchor,primary_correct={k:credit(v[0],y) for k,v in primary.items()},
        paired_primary_correct=gains,requested_confirmation=partner)
    if not partner:
        return dict(**record,status='single',selected=[anchor],fit_equivalents=len(opening)+1)
    first_row=dict(rows=len(y),mean=gains[partner],singles=[credit(primary[k][0],y) for k in (anchor,partner)])
    split='confirmation_1'
    pfile=folder/(split+'.joblib')
    digest=ab_run.sha(pfile)
    replication=joblib.load(pfile)
    pair={}
    provenance={}
    for name in (anchor,partner):
        row=next((r for r in hist if r['candidate']==name and r['split_id']==split and r['status']=='completed'),None)
        if row:
            file=folder/f"trial_{row['trial']}.joblib"
            provenance[name]='original autonomous run'
        else:
            sibling=next(d for d in (batch/'deep-e5',batch/'deep-bge') if d!=folder)
            other=next((r for r in read(sibling/'history.json') if r['candidate']==name and r['split_id']==split and r['status']=='completed'),None)
            supplemental=read(followup/'fits.json')['deep-bge']
            if other and ab_run.sha(sibling/(split+'.joblib'))==digest:
                file=sibling/f"trial_{other['trial']}.joblib"
                provenance[name]='identical partition in sibling run'
            elif supplemental['candidate']==name and ab_run.sha(batch/'deep-bge'/(split+'.joblib'))==digest:
                file=followup/'deep-bge/worker.joblib'
                check(ab_run.sha(file)==supplemental['model'],'Supplement changed')
                provenance[name]='additional counterfactual replication'
            else:
                return dict(**record,status='missing measured confirmation',missing=name,fit_equivalents=None)
        pair[name]=columns(file,replication,digest)
    y2=replication['y_search']
    first2=np.array([l<r for l,r in replication['X_search']])
    mean=mean_labels(pair[anchor][1],pair[partner][1],first2)
    second=dict(rows=len(y2),mean=credit(mean,y2),singles=[credit(pair[k][0],y2) for k in (anchor,partner)])
    extra,promote=ranking([first_row,second],len(y)+len(y2))
    final=[anchor,partner] if promote else [anchor]
    return dict(**record,matched=[first_row,second],confirmation_sources=provenance,status='replayed',
        selected=final,pooled_gain=extra,fit_equivalents=len(opening)+2+len(final),new_fits=0)


def main(batch,followup,out):
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(code=ab_run.sha(Path(__file__)),plan=ab_run.sha(ROOT/'docs/batches/B_complement_scheduler_replay_1123_zh.md'),
        fit_budget_per_path=11,new_fit_limit=0,model_api=0,uses_outer_score_for_selection=False)
    frozen_save(out/'protocol.json',protocol)
    picks={arm:replay(arm,batch,followup) for arm in ('deep-e5','deep-bge')}
    frozen_save(out/'decisions_before_score_lookup.json',picks)
    # Match exact portfolio identities only after saving decision records.
    follow=read(followup/'results.json')
    e5=read(followup/'fits.json')['deep-e5']['candidate']
    incumbent=follow['autonomous_original_terminal_choice']
    results={}
    for arm,row in picks.items():
        members=set(row.get('selected',[]))
        if members=={incumbent}:
            performance=follow['reserve']['deep']
        elif members=={incumbent,e5}:
            performance=follow['reserve']['soft_vote']
        else:
            performance=None
        results[arm]=dict(replay=row['status'],members=row.get('selected'),fit_equivalents=row['fit_equivalents'],
            matched_frozen_reserve=None if performance is None else dict(rows=performance['rows'],correct=performance['correct']),
            new_physical_fits=0,model_api_requests=0)
    frozen_save(out/'results.json',dict(results=results,decision_seal=ab_run.sha(out/'decisions_before_score_lookup.json')))
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','batch','followup','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        main(a.batch.resolve(),a.followup.resolve(),a.output.resolve())
