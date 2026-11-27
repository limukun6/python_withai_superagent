"""Train soft-target classification trees from strictly out-of-group teacher predictions."""
import argparse
import gc
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.field_interactions import choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, blocks
from superagent.models import _tabular_catboost
from superagent.splits import sample_groups
from superagent.table_foundation import FrozenTableClassifier

PLAN = ROOT / "research/task1_feedback_extension_1127/crossfit_distillation.json"
METHODS = ("constant", "agreement")


def run(out):
    import torch
    torch.set_num_threads(3)
    info = read(out / "inputs.json")
    fit = bound(out, "fitting.joblib", info)
    x, raw = fit["x"], fit["y"]
    classes, y = np.unique(raw, return_inverse=True)
    require(len(classes)==2,"Soft binary targets require two classes")
    panels = [bound(out,r+"_features.joblib",info) for r in ROLES]
    groups = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=x))
    probabilities = np.full((len(y),2),np.nan)
    viewed = np.zeros(len(y),dtype=int)
    folder = out / "distill_teachers"
    folder.mkdir(exist_ok=False)
    audits = []
    start = time.monotonic()
    for f,(train,held) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=72731).split(x,y,groups)):
        require(not set(groups[train]) & set(groups[held]),"Held row group entered teacher training")
        candidates = [ContextTokenTree(dict(pair_limit=3)),FrozenTableClassifier("categorical_tabular")]
        for j, model in enumerate(candidates):
            model.fit(x[train], y[train])
            require(np.array_equal(model.classes_,[0,1]),"Teacher changed label order")
            probabilities[held,j] = model.predict_proba(x[held])[:,1]
        viewed[held] += 1
        audits.append(dict(fold=f,teacher_fits=2,training=len(train),held=len(held),
             fitting_sha256=hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
             held_sha256=hashlib.sha256(held.astype("<i8").tobytes()).hexdigest()))
        print("teachers",f,len(train),len(held),round(time.monotonic()-start,2),flush=True)
        del candidates, model
        gc.collect()
    require(np.all(viewed==1) and np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all(),
            "Incomplete out-of-group feedback")
    np.save(folder/"out_of_fold.npy",probabilities,allow_pickle=False)
    report = dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),fits=8,overlap=0,folds=audits,
            predictions=sha(folder/"out_of_fold.npy"),rows=len(y),same_label_fraction=float(np.mean((probabilities[:,0]>.5)==(probabilities[:,1]>.5))),
            seconds=round(time.monotonic()-start,2),api_requests=0)
    save(folder/"sealed.json",report)
    mean = probabilities.mean(1)
    spread = np.abs(probabilities[:,0]-probabilities[:,1])
    base = ContextTokenTree(dict(pair_limit=3))
    base.profile,base.context = blocks(x)
    base.pairs = choose_pairs(rank_pairs(x,y),3)
    cat = base.transform(x)
    transformed = [base.transform(p) for p in panels]
    for name in METHODS:
        weight = np.full(len(y),.30) if name=="constant" else .45*(1.-spread)*(.5+np.abs(mean-.5))
        target = (1.-weight)*y + weight*mean
        require(np.array_equal(target>.5,y.astype(bool)),"Hard target lost majority")
        model = _tabular_catboost(y,6,3.)
        model.set_params(loss_function="CrossEntropy",thread_count=5)
        model.fit(cat,target,cat_features=list(range(cat.shape[1])))
        require(np.array_equal(model.classes_,[0,1]),"Soft classifier changed probability order")
        dest = out / ("distill_"+name)
        dest.mkdir(exist_ok=False)
        signatures = {}
        for role,features in zip(ROLES,transformed):
            p = model.predict_proba(features)
            require(np.isfinite(p).all() and np.allclose(p.sum(1),1.),"Invalid student probabilities")
            path = dest/(role+".npy")
            np.save(path,p,allow_pickle=False)
            signatures[role] = sha(path)
        save(dest/"sealed.json",dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),classes=classes.tolist(),
             predictions=signatures,shared_teacher_seal=sha(folder/"sealed.json"),student_fits=1,
             mean_teacher_share=float(weight.mean()),majority_hard_labels=len(y),api_requests=0))
        print("student",name,round(time.monotonic()-start,2),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run","elect"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    if args.action=="run":
        run(args.out)
    else:
        elect(args.out,METHODS,"distill",PLAN)
