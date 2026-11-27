"""Estimate the scale of training-panel nomination noise without another fit."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_evidence_allocation import SHARES, load_bank
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import blocks, tokens
from superagent.splits import sample_groups

PLAN=ROOT/"research/task1_feedback_extension_1127/selection_noise.json"


def simulate(changes,groups,observed):
    _,idx=np.unique(groups,return_inverse=True)
    matrix=np.zeros((int(idx.max())+1,changes.shape[1]),dtype=np.float32)
    np.add.at(matrix,idx,changes)
    rng=np.random.default_rng(89137)
    maxima=[]
    with threadpool_limits(limits=2):
        for _ in range(40):
            signs=(rng.integers(0,2,size=(100,len(matrix)),dtype=np.int8)*2-1).astype(np.float32)
            maxima.extend(np.maximum(0,(signs@matrix).max(1)))
    return dict(groups=len(matrix),replications=len(maxima),observed_best_net=observed,
                max_gain_median=float(np.median(maxima)),max_gain_p95=float(np.quantile(maxima,.95)),
                fraction_at_least_observed=float(np.mean(np.asarray(maxima)>=observed)))


def run(out):
    data=out/"development"
    info=read(data/"inputs.json")
    bank,original,base,classes=load_bank(data)
    require(original==read(out/"evidence_allocation/locked.json")["seals"],"Changed model registry")
    queries=[bound(data,r+"_features.joblib",info) for r in ROLES]
    truths=[bound(data,r+"_answers.joblib",info)==classes[1] for r in ROLES]
    fitting=bound(data,"fitting.joblib",info)
    profile,_=blocks(fitting["x"])
    y=np.concatenate(truths)
    ref=(base>.5)==y
    codes,deltas=[],[]
    for model,prob in bank.items():
        for share in SHARES:
            codes.append((model,share))
            proposed=((1-share)*base+share*prob>.5)==y
            deltas.append(proposed.astype(np.int8)-ref.astype(np.int8))
    changes=np.column_stack(deltas)
    all_panels=np.split(changes,np.cumsum([len(q) for q in queries])[:-1])
    fixed=np.array([w==.25 for _,w in codes])
    ranks=np.stack([v[:,fixed].sum(0) for v in all_panels])
    cor={f"{ROLES[i]}-{ROLES[j]}":float(spearmanr(ranks[i],ranks[j]).statistic)
         for i,j in ((0,1),(0,2),(1,2))}
    positive=ranks[0]>0
    panel=all_panels[0]
    observed=int(panel.sum(0).max())
    groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=queries[0]))
    profile_groups=np.asarray(tokens(queries[0][:,profile]))
    result=dict(protocol=sha(PLAN),registry=sha(out/"evidence_allocation/locked.json"),candidates=len(bank),
        share_configurations=len(codes),fixed_share=.25,ranking_spearman=cor,
        first_positive=int(positive.sum()),positive_again_in_both=int((positive&(ranks[1]>0)&(ranks[2]>0)).sum()),
        positive_then_pooled_negative=int((positive&(ranks[1]+ranks[2]<0)).sum()),
        first_panel_rows=len(panel),exact_group_flip=simulate(panel,groups,observed),
        recurring_profile_flip=simulate(panel,profile_groups,observed),actual_fits=0,api_requests=0)
    save(out/"selection_noise_result.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    run(parser.parse_args().out)
