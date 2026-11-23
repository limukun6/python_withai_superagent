"""Rebuild the recorded decision from observed internal predictions, without any fit."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np


COST={'word':1,'base':1,'deep':1,'plain_equal':2,'stream_adapter':4,'adapter':4,'stream_hybrid':5,'full_hybrid':5}


def audit(path):
    evidence=json.loads((path/'inner_evidence.json').read_text())
    payload=joblib.load(path/'training_payload.joblib')
    ids=np.asarray(evidence['held_row_ids'])
    np.testing.assert_array_equal(ids,payload['held_ids'])
    labels=payload['labels'][ids]
    groups=payload['groups'][ids]
    probe=np.array(evidence['probe'],dtype=bool)
    if set(groups[probe])&set(groups[~probe]):
        raise ValueError('Search probe and confirmation overlap')
    order=list(evidence['observed_predictions'])
    hits=np.array([np.asarray(evidence['observed_predictions'][k])==labels for k in order])
    records=json.loads((path/'worker_result.json').read_text())
    def winner(count):
        return max(range(len(order)),key=lambda i:(int(count[i]),-COST[order[i]],-list(COST).index(order[i])))
    complete=hits.sum(axis=1)
    choice=order[winner(complete)]
    if records['selected']!=choice or any(records['scores'][k]['total']!=int(complete[i]) for i,k in enumerate(order)):
        raise ValueError('Winner or decision evidence mismatch')
    unique,mapping=np.unique(groups,return_inverse=True)
    group_hits=np.stack([np.bincount(mapping,weights=h).astype(int) for h in hits])
    omissions={}
    for g in range(len(unique)):
        name=order[winner(complete-group_hits[:,g])]
        omissions[name]=omissions.get(name,0)+1
    probe_best=order[winner(hits[:,probe].sum(axis=1))]
    a,b=hits[order.index(choice)],hits[order.index(probe_best)]
    paired={key:dict(rows=int(mask.sum()),wins=int((a&~b&mask).sum()),losses=int((b&~a&mask).sum()))
            for key,mask in [('probe',probe),('confirmation',~probe)]}
    catalog_replay={}
    catalogs={'word_and_shallow':['word','base'],
              'add_deep_and_plain_mix':['word','base','deep','plain_equal'],
              'all_measured':order}
    for label,names in catalogs.items():
        available=[k for k in names if k in order]
        best=max(available,key=lambda k:(int(complete[order.index(k)]),-COST[k],-list(COST).index(k)))
        catalog_replay[label]=dict(selected=best,correct_on_inner=int(complete[order.index(best)]),
                                   measured_eligible=available)
    return dict(primary_rows=len(ids),probe_rows=int(probe.sum()),confirmation_rows=int((~probe).sum()),
          probe_best=probe_best,confirmation_best=order[winner(hits[:,~probe].sum(axis=1))],
          original_chosen=choice,paired_chosen_vs_probe_best=paired,distinct_primary_groups=len(unique),
          leave_one_group_out_winners=omissions,catalog_replay=catalog_replay,
          supervised_fits=0,model_requests=0,outer_scores_used=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    with a.output.open('x') as f:
        json.dump(audit(a.run),f,indent=2)
