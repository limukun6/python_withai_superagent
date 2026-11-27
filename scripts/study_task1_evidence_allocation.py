"""Compare how feedback is spent after broad exploration, with each row held out of its decision."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_outer import compare
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.splits import sample_groups

PLAN = ROOT / "research/task1_feedback_extension_1127/evidence_allocation.json"
PREFIXES=("trial_", "icl_", "tabm_", "boost_", "rank_", "factor_", "rules_", "measure_", "mitra_", "binding_",
          "distill_", "semantic_")
SHARES=(.125,.25,.5,1.)
POLICIES=("retain","nomination","pooled","protected")


def load_bank(data):
    classes = read(data/"trial_anchor/sealed.json")["classes"]
    bank, seals, unique = {}, {}, set()
    for entry in sorted(data.iterdir()):
        record=entry/"sealed.json"
        if not entry.name.startswith(PREFIXES) or not record.is_file():
            continue
        d=read(record)
        if not isinstance(d.get("predictions"),dict) or not set(ROLES)<=d["predictions"].keys():
            continue
        order = d.get("classes")
        if order is None and d.get("protocol") == sha(ROOT / "research/task1_feedback_extension_1127/contextual_inference.json"):
            # Both archived TabICL views used the same sklearn LabelEncoder and checked full class support.
            source = read(entry / "launch.json")
            require(source["inputs"] == sha(data / "inputs.json"), "Changed contextual study inputs")
            order = read(data / "trial_pretrained_icl/sealed.json")["classes"]
        require(order==classes,"Probability ordering differs")
        output=[]
        for role in ROLES:
            path=entry/(role+".npy")
            require(sha(path)==d["predictions"][role],"Source prediction changed")
            output.append(np.load(path,allow_pickle=False)[:,1])
        p=np.concatenate(output)
        key=hashlib.sha256(p.astype("<f8").tobytes()).hexdigest()
        if key not in unique:
            bank[entry.name]=p
            seals[entry.name]=sha(record)
            unique.add(key)
    base=(bank["trial_attribute_alignment"]+bank["trial_pretrained_icl"])/2
    return bank,seals,base,classes


def stats(y, base, alt, groups):
    diff=((alt>.5)==y).astype(int)-((base>.5)==y).astype(int)
    mean=float(diff.mean())
    _,g=np.unique(groups,return_inverse=True)
    sums=np.bincount(g,weights=diff-mean)
    stderr=float(np.sqrt(len(sums)/max(len(sums)-1,1)*np.sum(sums**2))/len(y))
    return dict(gain=mean,net=int(diff.sum()),brier_gain=float(np.mean((base-y)**2-(alt-y)**2)),
                group_standard_error=stderr,rows=len(y))


def run(out):
    data=out/"development"
    info=read(data/"inputs.json")
    bank,seals,base,classes=load_bank(data)
    x=np.concatenate([bound(data,r+"_features.joblib",info) for r in ROLES])
    y=np.concatenate([bound(data,r+"_answers.joblib",info) for r in ROLES])==classes[1]
    require(len(base)==len(x)==len(y),"Mismatched feedback rows")
    groups=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    splitter=StratifiedGroupKFold(6,shuffle=True,random_state=583257)
    folds=[held for _,held in splitter.split(x,y,groups)]
    pairs={("retain",0.):base}
    for m,p in bank.items():
        for weight in SHARES:
            pairs[m,weight]=(1-weight)*base+weight*p
    choices=[]
    allocations=[]
    identity=lambda key:hashlib.sha256(str(key).encode()).hexdigest()
    for i in range(6):
        nomination=np.concatenate([folds[(i+j)%6] for j in (1,2)])
        confirmations=[folds[(i+j)%6] for j in (3,4,5)]
        held=folds[i]
        require(not set(groups[held]) & set(groups[np.concatenate([nomination,*confirmations])]),"Overlapping episode")
        def evidence(key, rows):
            return stats(y[rows],base[rows],pairs[key][rows],groups[rows])
        def sort_key(key,record):
            return (-record["net"],key[1],-record["brier_gain"],identity(key))
        first={key:evidence(key,nomination) for key in pairs}
        best=[min([(m,w) for w in SHARES],key=lambda k:sort_key(k,first[k])) for m in bank]
        short=sorted(best,key=lambda k:sort_key(k,first[k]))[:4]
        reads=len(bank)+3*len(short)
        require(reads<=48,"Exceeded feedback budget")
        ref=("retain",0.)
        selected={"retain":ref,"nomination":min([ref,*short],key=lambda k:sort_key(k,first[k]))}
        confirm=np.concatenate(confirmations)
        all_rows=np.concatenate([nomination,confirm])
        pooled={k:evidence(k,all_rows) for k in [ref,*short]}
        selected["pooled"]=min([ref,*short],key=lambda k:sort_key(k,pooled[k]))
        accepted=[ref]
        evidence_log=[]
        for k in short:
            panels=[evidence(k,r) for r in confirmations]
            combined=evidence(k,confirm)
            passes=(sum(r["net"]>0 for r in panels)>=2 and min(r["gain"] for r in panels)>=-.005
                    and combined["gain"]>=.003 and combined["gain"]>combined["group_standard_error"])
            if passes:
                accepted.append(k)
            evidence_log.append(dict(candidate=list(k),checks=panels,pooled_confirmation=combined,accepted=passes))
        selected["protected"]=min(accepted,key=lambda k:sort_key(k,pooled[k]))
        choices.append(selected)
        allocations.append(dict(outer_fold=i,held_rows=len(held),reads=reads,nomination_reads=len(bank),
             choices={k:list(v) for k,v in selected.items()},checks=evidence_log))
    folder=out/"evidence_allocation"
    folder.mkdir(exist_ok=False)
    save(folder/"locked.json",dict(protocol=sha(PLAN),seals=seals,rotations=allocations))
    scores={name:np.zeros(len(y),dtype=bool) for name in POLICIES}
    episodes=[]
    for i,(held,selected) in enumerate(zip(folds,choices)):
        row={}
        for policy in POLICIES:
            correct=(pairs[selected[policy]][held]>.5)==y[held]
            scores[policy][held]=correct
            row[policy]=int(correct.sum())
        episodes.append(dict(outer_fold=i,rows=len(held),correct=row))
    result=dict(lock=sha(folder/"locked.json"),protocol=sha(PLAN),unique_candidates=len(bank),rows=len(y),episodes=episodes,
                correct={p:int(v.sum()) for p,v in scores.items()},
                versus_reference={p:compare(v,scores["retain"],groups) for p,v in scores.items() if p!="retain"},
                protected_against_pooled=compare(scores["protected"],scores["pooled"],groups),
                selections=[a["choices"] for a in allocations],reads_per_pooled_or_protected_episode=allocations[0]["reads"],
                reads_per_nomination_episode=len(bank),new_classifier_fits=0,api_requests=0)
    save(folder/"result.json",result)
    print(result["correct"],result["protected_against_pooled"],result["unique_candidates"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    run(parser.parse_args().out)
