"""Evaluate the newly elected raw model's frozen recipe using ten already fitted component sets."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.study_weighted_evidence import correctness, recipe
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.autonomous_run import planner_observation
from superagent.metrics import MetricSpec


PLAN=ROOT/'research/paired_encoder_search_1127/ordinary_sixteen_transfer_protocol.json'


def run(root):
    out=root/'sixteen-transfer'
    out.mkdir(exist_ok=True)
    saved=root/'regrouped-runs'
    source=read(saved/'lock.json')
    choices={'ordinary':('discovered-ordinary/discovered_default','ordinary'),
             'sixteen':('analogy-raw/original',None),'expanded':('auto-raw/auto_portfolio','expanded_budget')}
    selections={}
    receipts={}
    for label,(directory,older) in choices.items():
        path=root/directory
        seal=read(path/'sealed.json')
        final=read(path/'run/agent/final_selection.json')
        require(seal['independent_reload'] and seal['model_sha256']==final['model_sha256'] and final['refit'],'Unverified model')
        members=final['ensemble_components'] or [final['single_candidate']]
        require(members==seal['members'] and (len(members)==1 or final['family']=='equal_soft_vote'),'Actual vote or members changed')
        selections[label]=recipe(members)
        if older:
            require(selections[label]==source['recipes'][older],'Previous recipe changed')
        receipts[label]=sha(path/'sealed.json')
    names=sorted({m for r in selections.values() for m in r['members']})
    panels=[]
    for seed in source['seeds']:
        for i in range(1,6):
            box=saved/f'seed_{seed}'/f'fold_{i}'
            details=read(box/'partition.json')
            payload=box/'training.joblib'
            require(details['payload_sha256']==sha(payload),'Training fold changed')
            mats={}
            for name in names:
                path=box/name/'trial.joblib'
                receipt=read(box/name/'receipt.json')
                require(receipt['member']==name and receipt['trial_sha256']==sha(path) and receipt['payload_sha256']==sha(payload),
                        'Mismatched retained fold model')
                mats[name]=path
            panels.append((seed,i,payload,mats,details))
    lock(out/'inputs.json',dict(protocol=sha(PLAN),seals=receipts,choices=selections,
         panels=[dict(seed=s,fold=i,held=d['held_sha256'],payload=d['payload_sha256'],trials={m:sha(v) for m,v in mats.items()})
                 for s,i,_,mats,d in panels]))
    results=[]
    for seed,i,payload,paths,details in panels:
        data=joblib.load(payload)
        truth=np.asarray(data['y_search'])
        metric=MetricSpec.from_payload(data)
        require(metric.name=='accuracy','Declared metric differs')
        bank={}
        for name,path in paths.items():
            trial=joblib.load(path)
            require(trial['metrics']['resolved_configuration']['id']==name,'Different component executed')
            obs=planner_observation(trial,trial['metrics'],dict(sha256=details['payload_sha256'],labels=truth),metric)
            bank[name]=np.asarray(obs['probabilities'])
        rights={k:correctness(bank,truth,np.asarray(metric.labels),r) for k,r in selections.items()}
        counts={k:int(v.sum()) for k,v in rights.items()}
        pairs={k:dict(gained=int((rights['sixteen']&~v).sum()),lost=int((~rights['sixteen']&v).sum())) for k,v in rights.items() if k!='sixteen'}
        results.append(dict(seed=seed,fold=i,rows=len(truth),correct=counts,paired=pairs))
    totals={}
    for seed in source['seeds']:
        rec=[v for v in results if v['seed']==seed]
        totals[str(seed)]=dict(rows=sum(v['rows'] for v in rec),correct={k:sum(v['correct'][k] for v in rec) for k in selections},
            paired={k:{s:sum(v['paired'][k][s] for v in rec) for s in ('gained','lost')} for k in ('ordinary','expanded')})
    wins=sum(v['correct']['sixteen']>v['correct']['ordinary'] for v in results)
    passed=wins>=7 and all(t['correct']['sixteen']-t['correct']['ordinary']>=.003*t['rows'] for t in totals.values())
    result=dict(inputs=sha(out/'inputs.json'),totals=totals,folds=results,strict_fold_wins= wins,
                meets_frozen_stability_criterion=passed,new_classifier_fits=0,new_model_requests=0,course_validation_opened=False,
                official_test_opened=False,H2_opened=False)
    lock(out/'results.json',result)
    print(totals,wins,passed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
