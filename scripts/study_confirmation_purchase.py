"""Compare three rules for acquiring evidence on matched training-only decision opportunities."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.analogy_review import SHARES,promoted
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.experiment_policy import identity
from superagent.finalist_race import paired_error
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN=ROOT/'research/paired_encoder_search_1127/confirmation_purchase_protocol.json'
RULES=('strict','row','group')


def ballots(panel,anchor,adviser,share):
    base,pred=panel['votes'][anchor],panel['votes'][adviser]
    truth=panel['labels']
    before=base.argmax(1)==truth
    after=((1-share)*base+share*pred).argmax(1)==truth
    return after.astype(int)-before.astype(int)


def errors(delta,groups):
    gain=int((delta>0).sum());loss=int((delta<0).sum());n=len(delta)
    row=paired_error(dict(rows=n,gained=gain,lost=loss))
    units,idx,counts=np.unique(groups,return_inverse=True,return_counts=True)
    centered=np.bincount(idx,weights=delta,minlength=len(units))-counts*delta.mean()
    group=float(np.sqrt(len(units)/(len(units)-1)*np.square(centered).sum())/n) if len(units)>1 else 0.
    return gain,loss,row,group


def episode(panels,anchor,other,keys):
    primary,*rest=panels
    changes=[ballots(primary,anchor,other,share) for share in SHARES]
    pos=1+int(np.argmax([v.sum() for v in changes[1:]]))
    share=SHARES[pos]
    delta=changes[pos]
    gain,loss,row,group=errors(delta,primary['groups']);net=gain-loss
    admissions=dict(strict=net>0,row=net>0 or (gain>0 and -net<=row*len(delta)),
                    group=net>0 or (gain>0 and -net<=group*len(delta)))
    conf=[dict(rows=len(p['labels']),gain=int(ballots(p,anchor,other,share).sum())) for p in rest[:2]]
    eff=int(sum(ballots(p,anchor,other,share).sum() for p in rest[2:]))
    held_rows=sum(len(p['labels']) for p in rest[2:])
    decisions={}
    for name,admit in admissions.items():
        accepted=bool(admit and promoted(net,conf,primary_rows=len(delta)))
        decisions[name]=dict(panels=2 if admit else 0,accepted=accepted,realized_net=eff if accepted else 0,
           false_promotion=bool(accepted and eff<0),missed_benefit=bool(not accepted and eff>0))
    return dict(primary=primary['fold'],anchor=keys[anchor],challenger=keys[other],primary_rows=len(delta),held_rows=held_rows,
        proposed_share=share,gains=gain,losses=loss,errors=dict(row=row,group=group),
        confirmation=conf,counterfactual_held_net=eff,decisions=decisions)


def run(root):
    out=root/'confirmation-purchase'
    out.mkdir(exist_ok=True)
    storage=root/'regrouped-runs'
    original=read(storage/'lock.json')
    source=root/'raw/race/run/agent/refit_data.joblib'
    require(sha(source)==original['training_sha256'],'Incorrect training source')
    data=joblib.load(source)
    labels=np.asarray(data['y_train'])
    groups=sample_groups(TrainingTask(data['kind'],data['X_train'],labels),'shared_text')
    folders=[]
    for seed in original['seeds']:
        parts=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(groups,labels,groups)
        for i,(_,held) in enumerate(parts,1):
            d=storage/f'seed_{seed}'/f'fold_{i}'
            stamp=read(d/'partition.json')
            payload=d/'training.joblib'
            require(stamp['held_sha256']==fingerprint(held) and stamp['payload_sha256']==sha(payload),'Wrong training fold')
            paths={m:d/m/'trial.joblib' for m in original['members']}
            for m,path in paths.items():
                rec=read(path.parent/'receipt.json')
                require(rec['payload_sha256']==stamp['payload_sha256'] and rec['member']==m and rec['trial_sha256']==sha(path),'Wrong base fit')
            folders.append((seed,i,held,d,payload,paths,stamp))
    lock(out/'inputs.json',dict(plan=sha(PLAN),supervised=sha(source),folds=[dict(seed=s,fold=i,held=v['held_sha256'],
        payload=v['payload_sha256'],sources={m:sha(p) for m,p in paths.items()}) for s,i,_,_,_,paths,v in folders]))
    totals={};outcomes=[]
    for seed in original['seeds']:
        panels=[];keys={}
        for s,i,ids,d,payload,paths,stamp in folders:
            if s!=seed:
                continue
            batch=joblib.load(payload)
            truth=np.asarray(batch['y_search'])
            require(np.array_equal(truth,labels[ids]),'Truth order differs')
            metric=MetricSpec.from_payload(batch);bank={}
            for n,path in paths.items():
                obj=joblib.load(path);facts=obj['metrics']
                key=identity(facts['resolved_configuration']['params'])
                require(key==keys.setdefault(n,key),'Method changed across training folds')
                obs=planner_observation(obj,facts,dict(sha256=stamp['payload_sha256'],labels=truth),metric)
                bank[n]=np.asarray(obs['probabilities'])
            panels.append(dict(fold=i,labels=truth,votes=bank,groups=groups[ids]))
        rows=[]
        for j in range(5):
            rotated=panels[j:]+panels[:j];primary=rotated[0]
            anchor=min(keys,key=lambda m:(-int((primary['votes'][m].argmax(1)==primary['labels']).sum()),keys[m]))
            for n in sorted(keys,key=keys.__getitem__):
                if n!=anchor:
                    rows.append(episode(rotated,anchor,n,keys))
        totals[str(seed)]={a:{k:int(sum(r['decisions'][a][k] for r in rows)) for k in ('panels','accepted','realized_net','false_promotion','missed_benefit')} for a in RULES}
        outcomes.append(dict(seed=seed,method_keys=keys,opportunities=rows))
    passed=all(v['group']['realized_net']>=v['row']['realized_net'] and v['group']['panels']<=v['row']['panels'] for v in totals.values())
    improved=any(v['group']['realized_net']>v['row']['realized_net'] or v['group']['panels']<v['row']['panels'] for v in totals.values())
    result=dict(inputs=sha(out/'inputs.json'),totals=totals,results=outcomes,advance=bool(passed and improved),
         opportunities=sum(len(v['opportunities']) for v in outcomes),repeated_rotations=True,new_supervised_fits=0,model_api_requests=0,
         course_validation_opened=False,formal_test_opened=False,H2_opened=False)
    lock(out/'results.json',result)
    print(totals,result['advance'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args().root)
