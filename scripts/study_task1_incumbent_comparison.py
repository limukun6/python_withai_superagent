"""Compare a library of frozen challengers directly with the teammate's actual model family."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_evidence_allocation import PREFIXES, SHARES, load_bank, stats
from scripts.study_task1_outer import compare
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.splits import sample_groups

PLAN=ROOT/"research/task1_feedback_extension_1127/incumbent_comparison.json"


def run(out):
    dev=out/"development"
    require((dev/"typedicl_election/results.json").exists(),"Finish typed-encoder experiment first")
    bank,seals,pair,classes=load_bank(dev,(*PREFIXES,"typedicl_"))
    original=bank.pop("trial_anchor")
    bank["previous_research_pair"]=pair
    matrix={("retain_original",0.):original}
    for name,p in bank.items():
        for share in SHARES:
            matrix[name,share]=(1-share)*original+share*p
    info=read(dev/"inputs.json")
    features=[bound(dev,r+"_features.joblib",info) for r in ROLES]
    y=np.concatenate([bound(dev,r+"_answers.joblib",info) for r in ROLES])==classes[1]
    groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=np.concatenate(features)))
    indices=np.split(np.arange(len(y)),np.cumsum([len(x) for x in features])[:-1])
    def choose(parts):
        selected=np.concatenate(parts)
        eligible=[]
        for key,p in matrix.items():
            if key[1]==0:
                continue
            deltas=[int((((p[rr]>.5)==y[rr]).sum())-(((original[rr]>.5)==y[rr]).sum())) for rr in parts]
            if min(deltas)<0:
                continue
            record=stats(y[selected],original[selected],p[selected],groups[selected])
            margin=record["gain"]-record["group_standard_error"]
            if record["gain"]>=.003 and margin>0:
                eligible.append(((-margin,-record["net"],key[1],hashlib.sha256(str(key).encode()).hexdigest()),key))
        return min(eligible)[1] if eligible else ("retain_original",0.)
    choices=[]
    for i,held in enumerate(indices):
        others=[r for j,r in enumerate(indices) if i!=j]
        require(not set(groups[held])&set(groups[np.concatenate(others)]),"Shared row group between panels")
        choices.append(choose(others))
    folder=out/"incumbent_comparison"
    folder.mkdir(exist_ok=False)
    save(folder/"locked.json",dict(protocol=sha(PLAN),registry=seals,choices=[list(c) for c in choices]))
    rows=[]
    elected=np.zeros(len(y),dtype=bool)
    single=(original>.5)==y
    previous=(pair>.5)==y
    for r,ix,key in zip(ROLES,indices,choices):
        elected[ix]=(matrix[key][ix]>.5)==y[ix]
        rows.append(dict(panel=r,rows=len(ix),method=list(key),correct=dict(original=int(single[ix].sum()),
                     previous_pair=int(previous[ix].sum()),elected=int(elected[ix].sum()))))
    result=dict(protocol=sha(PLAN),lock=sha(folder/"locked.json"),panels=rows,rows=len(y),
                correct=dict(original=int(single.sum()),previous_pair=int(previous.sum()),elected=int(elected.sum())),
                differences=dict(to_original=compare(elected,single,groups),to_pair=compare(elected,previous,groups)),
                all_development_nomination=list(choose(indices)),model_fits=0,api_requests=0)
    save(folder/"result.json",result)
    print(result["correct"],result["all_development_nomination"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    run(parser.parse_args().out)
