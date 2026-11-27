"""Measure sparse reweighting only among the actual finalists of each completed cold start."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_weighted_evidence import correctness,learned_recipe,matrices,recipe
from scripts.unified_released_study import ROOT,read,require,save,sha


CASES={"matched":"raw/race","frontier":"frontier-raw/frontier","scale":"frontier-raw/frontier_scale",
       "portfolio":"auto-raw/auto_portfolio","image":"anonymous-image"}


def analyze(root,case):
    agent=root/CASES[case]/"run/agent"
    out=root/"within-run-weights"/case
    out.mkdir(parents=True,exist_ok=True)
    finished=read(agent/"run_outcome.json")
    limit=read(agent/"run_manifest.json")["fit_limit"]
    require(finished["status"]=="completed","Need a completed autonomous run")
    initial=dict(case=case,available_fits=limit,original_fits=finished["attempted_classifier_fits"],new_classifier_fits=0,
                 model_requests=0,external_labels_opened=False,protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/within_run_weights_protocol.json"))
    plan_path=agent/"finalist_lock.json"
    if not plan_path.is_file() or len(read(plan_path)["fresh"])<2:
        result=dict(initial,eligible=False,reason="No plan with two independent confirmation folds.",new_meta_fits=0)
        save(out/"results.json",result)
        return result
    plan=read(plan_path)
    original=recipe(finished["members"])
    names=sorted(plan["members"])
    history=read(agent/"history.json")
    def read_fold(fold,ids):
        paths={r["candidate"]:agent/f"trial_{r['trial']}.joblib" for r in history
               if r["split_id"]==fold and r["status"]=="completed" and r["candidate"] in ids}
        require(set(paths)==set(ids),"Finalist has no genuine training artifact")
        return matrices(agent,fold,ids,paths)
    bank,y,labels,configs=read_fold("primary",names)
    names=sorted(bank)
    lock_file=out/"weight_lock.json"
    if lock_file.exists():
        lock=read(lock_file)
        require(lock["plan_sha256"]==sha(plan_path) and lock["candidates"]==names,"Primary selection differs")
    else:
        r,meta_fits=learned_recipe(bank,y,labels,names)
        lock=dict(plan_sha256=sha(plan_path),candidates=names,weighted=r,meta_fits=meta_fits)
        save(lock_file,lock)
    alternative=lock["weighted"]
    used=sorted(set(original["members"])|set(alternative["members"]))
    controls=recipe(alternative["members"])
    comparison=[]
    for fold in plan["fresh"]:
        bank,y,classes,other=read_fold(fold,used)
        require(np.array_equal(labels,classes) and all(configs[n]==other[n] for n in used),"Independent configuration changed")
        before,uniform,weighted=(correctness(bank,y,classes,r) for r in (original,controls,alternative))
        comparison.append(dict(fold=fold,rows=len(y),original=int(sum(before)),own_equal=int(sum(uniform)),
                               weighted=int(sum(weighted)),gained=int(sum(weighted & ~before)),lost=int(sum(before & ~weighted))))
    costs={}
    for n in used:
        row=next(r for r in history if r["candidate"]==n and r["split_id"]=="primary" and r["status"]=="completed")
        costs[n]=joblib.load(agent/f"trial_{row['trial']}.joblib")["metrics"]["classifier_fits"]
    require(all(c==1 for c in costs.values()),"Compare known one-fit finalist costs")
    delta=len(alternative["members"])-len(original["members"])+lock["meta_fits"]
    budget_ok=initial["original_fits"]+delta<=limit
    net=sum(d["weighted"]-d["original"] for d in comparison)
    own_gain=sum(d["weighted"]-d["own_equal"] for d in comparison)
    passed=(budget_ok and all(d["weighted"]>d["original"] for d in comparison)
            and net/sum(d["rows"] for d in comparison)>=.003 and own_gain>0)
    result=dict(initial,eligible=True,actual_recipe=original,weighted_recipe=alternative,comparisons=comparison,
                new_meta_fits=lock["meta_fits"],projected_total_fits=initial["original_fits"]+delta,
                would_fit_budget=budget_ok,confirmation_net=net,gain_over_own_equal=own_gain,advance=passed)
    save(out/"results.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    p.add_argument("--case",choices=tuple(CASES),required=True)
    args=p.parse_args()
    print(analyze(args.root,args.case),flush=True)
