"""Blind, staged comparison against the latest delivered model's existing training folds."""
import argparse
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.cache_paired_encoders import configure_weights
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.paired_encoder import NAMES
from superagent.prediction_evidence import fingerprint


PLAN = ROOT / "research/paired_encoder_search_1127/protocol.json"


def order_key(option):
    return -option["score"], len(option["members"]), tuple(option["members"])


def evaluate(members, columns, labels, incumbent):
    probabilities = np.mean([columns[name] for name in members], axis=0)
    predicted = probabilities > .5
    right = predicted == labels
    original = (columns[incumbent] > .5) == labels
    return dict(members=list(members), correct=int(sum(right)), rows=len(labels), score=float(np.mean(right)),
                gained=int(sum(right & ~original)), lost=int(sum(~right & original)),
                delta=float((right.sum()-original.sum())/len(labels)), ties=int(sum(probabilities==.5)))


def run_study(args):
    agent = args.source / "titles/run/agent"
    incumbent = read(agent/"final_selection.json")["single_candidate"]
    winning = next(row for row in read(agent/"history.json") if row["candidate"]==incumbent)
    require(winning["split_id"]=="primary","Expected a known primary baseline")
    inherited = {"primary":agent/f"trial_{winning['trial']}.joblib", "confirmation_1":args.confirmed}
    args.output.mkdir(parents=True, exist_ok=True)
    configure_weights(args.weights)
    settings = env(args.weights, args.output/"observed_fits.jsonl")
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed", "SUPERAGENT_PAIRED_FEATURE_CACHE":str(args.cache)})
    used = {}
    for name in ("primary","confirmation_1","confirmation_2"):
        split=read(agent/f"{name}_split.json")
        used[name]=set(split["held"])
    assert all(not (a & b) for a,b in combinations(used.values(),2))
    save(args.output/"study_identity.json",dict(plan_sha256=sha(PLAN), baseline=incumbent,
         original_primary_sha256=sha(inherited["primary"]), original_confirmation_sha256=sha(inherited["confirmation_1"]),
         visible_methods=sorted(NAMES)))

    def fold_data(fold, names):
        payload=agent/f"{fold}.joblib"
        partition=read(agent/f"{fold}_split.json")
        require(sha(payload)==partition["payload_sha256"],"Partition payload changed")
        labels=np.asarray(joblib.load(payload)["y_search"])
        require(fingerprint(labels)==partition["label_sha256"],"Label order changed")
        columns={}
        for name in sorted(set(names)):
            path = inherited.get(fold) if name==incumbent else None
            if path is None:
                folder=args.output/fold/name
                folder.mkdir(parents=True, exist_ok=True)
                path=folder/"trial.joblib"
                ledger=folder/"fits.json"
                receipt=folder/"receipt.json"
                if not receipt.exists():
                    require(not ledger.exists() and not path.exists(), "Unsettled attempt requires inspection: "+str(folder))
                    admission=folder/"admission.json"
                    save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)))
                    with (folder/"worker.log").open("w") as stream:
                        subprocess.run([sys.executable,"-m","superagent.experiment_worker",str(payload),name,str(path),
                                        str(agent/"runtime_registry_0.json"),"--fit-admission",str(admission)],
                                       env=settings, cwd=ROOT, check=True, timeout=600, stdout=stream,stderr=subprocess.STDOUT)
                    fit=read(ledger)
                    require(fit["attempted_classifier_fits"]==fit["completed_classifier_fits"]==1,"Unaccounted fits")
                    save(receipt,dict(candidate=name,input=sha(payload),artifact=sha(path),actual_fits=1))
                require(read(receipt)["artifact"]==sha(path),"Stored trial artifact changed")
            item=joblib.load(path)
            record=item["prediction_record"]
            require(record["evaluation_payload_sha256"]==partition["payload_sha256"] and
                    record["label_sha256"]==partition["label_sha256"] and
                    item["metrics"]["resolved_configuration"]["id"]==name,"Mismatched predictions")
            probs=np.asarray(item["search_scores"])[:,list(item["score_classes"]).index(1)]
            hard=np.asarray(item["search_predictions"])
            require(len(probs)==len(labels) and np.allclose(probs,np.clip(probs,0,1)) and
                    np.array_equal(probs>.5,hard),"Tied or inconsistent binary prediction; inspect before scoring")
            columns[name]=probs
        return columns,labels

    columns,labels=fold_data("primary",[incumbent,*NAMES])
    discovery=[evaluate([incumbent],columns,labels,incumbent)]
    for name in sorted(NAMES):
        for members in ([name],[incumbent,name]):
            discovery.append(evaluate(members,columns,labels,incumbent))
    choices=sorted(discovery[1:],key=order_key)[:2]
    save(args.output/"discovery_lock.json",dict(all_options=discovery,locked=[row["members"] for row in choices]))
    print("discovery",[(r["members"],r["correct"],r["gained"]-r["lost"]) for r in discovery],flush=True)
    columns,labels=fold_data("confirmation_1",[incumbent,*(n for c in choices for n in c["members"])])
    checks=[]
    for original in choices:
        other=evaluate(original["members"],columns,labels,incumbent)
        pooled=(original["gained"]-original["lost"]+other["gained"]-other["lost"])/(original["rows"]+other["rows"])
        worst=min(original["delta"],other["delta"])
        checks.append(dict(primary=original, confirmation=other, pooled_gain=pooled, worst_fold=worst,
                           passed=worst>0 and pooled>=.003))
    eligible=sorted([c for c in checks if c["passed"]],key=lambda c:(-c["worst_fold"],-c["pooled_gain"],
                    len(c["primary"]["members"]),tuple(c["primary"]["members"])))
    selected=eligible[0]["primary"]["members"] if eligible else [incumbent]
    save(args.output/"confirmation_lock.json",dict(comparisons=checks, locked=selected))
    third=None
    if eligible:
        columns,labels=fold_data("confirmation_2",[incumbent,*selected])
        third=evaluate(selected,columns,labels,incumbent)
    promote=bool(third and third["delta"]>=0)
    physical=[read(p) for p in args.output.glob("*/*/fits.json")]
    output=dict(primary=discovery, confirmations=checks, third=third, candidate_locked=selected,
                advance_to_raw_solve=promote, baseline=incumbent, actual_fits=sum(x["attempted_classifier_fits"] for x in physical),
                new_model_api_requests=0, designated_validation_opened=False)
    save(args.output/"results.json",output)
    print("completed",[(c["primary"]["members"],c["confirmation"]["correct"],c["passed"]) for c in checks],
          "third",third,"fits",output["actual_fits"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","confirmed","output","weights","cache"):
        p.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    run_study(p.parse_args())
