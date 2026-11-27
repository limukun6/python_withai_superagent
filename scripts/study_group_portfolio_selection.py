"""Select among nearly tied frozen recipes using independent connected groups, then evaluate once."""
import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.study_consensus_feedback import aligned_training_predictions
from scripts.unified_released_study import ROOT,require,sha


PLAN=ROOT/'research/paired_encoder_search_1127/group_selection_protocol.json'


def buckets(groups):
    return np.asarray([int.from_bytes(hashlib.sha256(('group portfolio reliability v1/'+g).encode()).digest()[:8],
                                     'big')%3 for g in groups])


def group_stats(values,groups):
    units,index,size=np.unique(groups,return_inverse=True,return_counts=True)
    totals=np.bincount(index,weights=values)
    row=float(np.mean(values))
    equal=float(np.mean(totals/size))
    # The residuals are aggregated within the independent sampling unit.
    cluster_se=(np.sqrt(len(units)/(len(units)-1)*np.square(totals-row*size).sum())/len(values)
                if len(units)>1 else 0.)
    return row,equal,float(cluster_se)


def selection(prob,truth,groups,keys):
    rights={r:(p>.5)==truth for r,p in prob.items()}
    measures={}
    for r,p in prob.items():
        row,group,_=group_stats(rights[r],groups)
        brier=float(np.mean(np.square(p-truth)))
        measures[r]=dict(row=row,group=group,brier=brier)
    def standard(r):
        m=measures[r]
        return (-m['row'],len(r),m['brier'],tuple(keys[n] for n in r))
    anchor=min((r for r in prob if len(r)==1),key=standard)
    leader=min(prob,key=standard)
    eligible=[]
    for r in prob:
        margin,_,stderr=group_stats(rights[r].astype(float)-rights[leader].astype(float),groups)
        if margin>=-stderr-1e-12:
            eligible.append(r)
    def order(r):
        m=measures[r]
        return (-m['group'],len(r),-m['row'],m['brier'],tuple(keys[n] for n in r))
    return anchor,leader,min(eligible,key=order),len(eligible)


def test_fold(out,name,bank,truth,groups,keys):
    names=sorted(bank,key=keys.__getitem__)
    prob={r:np.mean([bank[n] for n in r],axis=0) for length in (1,2,3) for r in combinations(names,length)}
    roles=buckets(groups)
    require(set(roles)=={0,1,2},'Expected three separate evidence roles')
    correct={k:np.zeros(len(truth),dtype=bool) for k in ('anchor','row','group')}
    decisions=[]
    for i in range(3):
        learn,confirm,held=[roles==(i+j)%3 for j in range(3)]
        anchor,leader,robust,count=selection({r:p[learn] for r,p in prob.items()},truth[learn],groups[learn],keys)
        candidates={'row':leader,'group':robust}
        lock(out/f'{name}_{i}_learn.json',dict(anchor=anchor,row_leader=leader,group_nominee=robust,
             eligible_within_one_paired_se=count,learning_rows=int(learn.sum())))
        row_right=((prob[anchor][confirm]>.5)==truth[confirm]).astype(float)
        actions={}
        correct['anchor'][held]=(prob[anchor][held]>.5)==truth[held]
        for arm,nominee in candidates.items():
            alt=((prob[nominee][confirm]>.5)==truth[confirm]).astype(float)
            delta,equal,_=group_stats(alt-row_right,groups[confirm])
            approve=delta>0 and (arm=='row' or equal>=-1e-12)
            final=nominee if approve else anchor
            actions[arm]=dict(proposal=nominee,confirmation_row_change=delta,confirmation_group_change=equal,
                              confirmed=approve,selected=final)
        lock(out/f'{name}_{i}_choice.json',actions)
        for arm,action in actions.items():
            correct[arm][held]=(prob[action['selected']][held]>.5)==truth[held]
        decisions.append(dict(rotation=i,anchor=anchor,options=len(prob),near_ties=count,
             proposals_different=leader!=robust,decisions_different=actions['row']['selected']!=actions['group']['selected'],
             held_rows=int(held.sum()),actions=actions))
    row,group=correct['row'],correct['group']
    return dict(fold=name,rows=len(truth),groups=len(set(groups)),options=len(prob),
                correct={k:int(v.sum()) for k,v in correct.items()},gained=int(np.sum(group&~row)),
                lost=int(np.sum(row&~group)),group_accuracy={k:group_stats(v,groups)[1] for k,v in correct.items()},
                rotations=decisions)


def run(root):
    out=root/'group-portfolio-selection'
    out.mkdir(exist_ok=True)
    rows=[test_fold(out,tag,bank,truth,groups,keys)
          for tag,bank,truth,groups,_,keys,_ in aligned_training_predictions(root,out,PLAN)]
    n=sum(r['rows'] for r in rows)
    correct={k:sum(r['correct'][k] for r in rows) for k in ('anchor','row','group')}
    gains=sum(r['gained'] for r in rows)
    losses=sum(r['lost'] for r in rows)
    net=[r['gained']-r['lost'] for r in rows]
    passed=((gains-losses)/n>=.003 and sum(d>0 for d in net)>=3
            and all(d/r['rows']>=-.003 for r,d in zip(rows,net)))
    result=dict(protocol=sha(PLAN),inputs=sha(out/'inputs.json'),rows=n,correct=correct,gained=gains,lost=losses,
        by_fold=net,advance=passed,folds=rows,different_final_decisions=sum(x['decisions_different'] for r in rows for x in r['rotations']),
        paired_supervised_rule_selections=15,new_classifier_fits=0,model_api_requests=0,
        official_test_opened=False,designated_validation_opened=False,H2_opened=False)
    lock(out/'results.json',result)
    print(correct,gains,losses,net,result['different_final_decisions'],passed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
