"""Learn one sparse probability vote from matched training predictions and test it on two other folds."""
import argparse
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.experiment_policy import identity
from superagent.metrics import MetricSpec


FOLDS = ("primary", "confirmation_1", "confirmation_2")


def available(root):
    paths = {f:{} for f in FOLDS}
    for tag in ("original","expanded","race","default"):
        agent = root/"raw"/tag/"run/agent"
        for row in read(agent/"history.json"):
            if row["status"]=="completed":
                paths[row["split_id"]].setdefault(row["candidate"],agent/f"trial_{row['trial']}.joblib")
    other = root/"raw/original_compute_matched/run/agent"
    for row in read(other/"logical_path.json"):
        if row["source"]=="new":
            paths[row["fold"]].setdefault(row["members"][0],other/f"trial_{row['step']}.joblib")
    for tag in ("frozen-screen","scale-screen"):
        for fold in FOLDS:
            for receipt in (root/tag/fold).glob("*/receipt.json"):
                d = read(receipt)
                path = receipt.parent/"trial.joblib"
                require(d["artifact"]==sha(path),"Research training receipt differs")
                paths[fold].setdefault(d["candidate"],path)
    return paths


def matrices(agent, fold, names, paths):
    payload = agent/(fold+".joblib")
    split = read(agent/(fold+"_split.json"))
    require(sha(payload)==split["payload_sha256"],"Training partition changed")
    d = joblib.load(payload)
    metric = MetricSpec.from_payload(d)
    require(metric.name=="accuracy","This registered comparison evaluates accuracy")
    labels = np.asarray(d["y_search"])
    matrices = {}
    config = {}
    for n in names:
        item = joblib.load(paths[n])
        facts = item["metrics"]
        require(facts["resolved_configuration"]["id"]==n,"Model identity changed")
        observation = planner_observation(item,facts,dict(sha256=split["payload_sha256"],labels=labels),metric)
        if observation["probabilities"] is not None:
            matrices[n] = np.asarray(observation["probabilities"],dtype=float)
            config[n] = identity({key:value for key,value in facts["resolved_configuration"].items()
                                  if key!="registry_sha256"})
    return matrices,labels,np.asarray(metric.labels),config


def optimize(bank, labels, classes, names):
    tensor = np.stack([bank[n] for n in names])
    targets = (labels[:,None]==classes[None,:]).astype(float)
    rows = len(labels)
    uniform = np.full(len(names),1/len(names))
    residual = (tensor-targets).reshape(len(names),-1)
    gram = residual@residual.T/rows
    shrinkage = 1/np.sqrt(rows)
    def loss(w):
        return w@gram@w + shrinkage*np.dot(w-uniform,w-uniform)
    def gradient(w):
        return 2*(gram@w+shrinkage*(w-uniform))
    fit = minimize(loss,uniform,jac=gradient,method="SLSQP",bounds=[(0.,1.)]*len(names),
                   constraints=[dict(type="eq",fun=lambda w:w.sum()-1,jac=lambda w:np.ones_like(w))],
                   options=dict(ftol=1e-12,maxiter=300))
    require(fit.success and np.isfinite(fit.x).all(),"The convex weight fit failed")
    w = np.clip(fit.x,0.,1.)
    return w/w.sum()


def recipe(names,weights=None):
    return dict(members=list(names),weights=list(map(float,weights)) if weights is not None else [1/len(names)]*len(names))


def correctness(bank, labels, classes, r):
    matrix = sum(w*bank[n] for n,w in zip(r["members"],r["weights"]))
    return classes[np.argmax(matrix,axis=1)]==labels


def choose(bank,y,classes,names,plan):
    options = [recipe(sub) for k in range(1,min(3,len(names))+1) for sub in combinations(names,k)]
    simple = min(options,key=lambda r:(-int(correctness(bank,y,classes,r).sum()),len(r["members"]),identity(r["members"])))
    weights = optimize(bank,y,classes,names)
    ranked = sorted(zip(names,weights),key=lambda pair:(-pair[1],identity(pair[0])))
    selected = [n for n,w in ranked[:3] if w>1e-7]
    fits = 1
    if len(selected)==1:
        weights = np.ones(1)
    elif set(selected)!=set(names):
        weights = optimize(bank,y,classes,selected)
        fits += 1
    else:
        weights = [dict(zip(names,weights))[n] for n in selected]
    weighted = recipe(selected,weights)
    uniform = recipe(selected)
    lock = dict(protocol_sha256=sha(plan),eligible=names,equal_options_considered=len(options),simple=simple,
                weighted=weighted,own_uniform=uniform,weight_fits=fits,primary_rows=len(y))
    return lock


def run(root):
    out = root/"weighted-screen"
    out.mkdir(exist_ok=True)
    plan = ROOT/"research/paired_encoder_search_1127/weighted_protocol.json"
    agent = root/"raw/original/run/agent"
    paths = available(root)
    common = sorted(set.intersection(*(set(p) for p in paths.values())))
    bank,y,classes,configs = matrices(agent,FOLDS[0],common,paths[FOLDS[0]])
    names = sorted(bank)
    require(len(names)>=2,"Need at least two aligned probability sources")
    lock_path = out/"selection_lock.json"
    if lock_path.exists():
        lock = read(lock_path)
        require(lock["protocol_sha256"]==sha(plan) and lock["eligible"]==names and lock["primary_rows"]==len(y),
                "Cannot resume a changed primary selection")
    else:
        lock = choose(bank,y,classes,names,plan)
        save(lock_path,lock)
    simple,weighted,uniform = (lock[k] for k in ("simple","weighted","own_uniform"))
    selected = weighted["members"]
    fits = lock["weight_fits"]
    used = sorted(set(simple["members"])|set(selected))
    comparisons = []
    for fold in FOLDS:
        if fold!="primary":
            bank,y,other_classes,other_specs = matrices(agent,fold,used,paths[fold])
            require(np.array_equal(other_classes,classes) and all(configs[n]==other_specs[n] for n in used),
                    "Configuration or class columns changed on confirmation")
        before,own,after = [correctness(bank,y,classes,r) for r in (simple,uniform,weighted)]
        comparisons.append(dict(fold=fold,rows=len(y),reference=int(sum(before)),own_uniform=int(sum(own)),
                                weighted=int(sum(after)),gained=int(sum(after & ~before)),lost=int(sum(~after & before))))
    checks=comparisons[1:]
    net=sum(d["weighted"]-d["reference"] for d in checks)
    own_gain=sum(d["weighted"]-d["own_uniform"] for d in checks)
    passed=all(d["weighted"]>d["reference"] for d in checks) and net/sum(d["rows"] for d in checks)>=.003 and own_gain>0
    result=dict(selection=lock,comparisons=comparisons,confirmation_net=net,confirmation_gain_vs_own_uniform=own_gain,
                advance_to_fresh_groups=passed,new_base_classifier_fits=0,weight_fits=fits,model_api_requests=0,
                designated_validation_opened=False,official_test_opened=False)
    save(out/"results.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    r=run(parser.parse_args().root)
    print(r["selection"]["weighted"],r["comparisons"],r["advance_to_fresh_groups"],flush=True)
