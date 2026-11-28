"""Test an independently confirmed fallback when the largest nominated blend failed."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_accuracy_veto import accuracy_pass
from scripts.study_task1_conditional_opportunity import metrics, ordered
from scripts.study_task1_extensions import key
from scripts.study_task1_outer import arrays, compare
from scripts.study_task1_population_expert import PHASES, SHARES, STAGE as POPULATION, combine, rebuild_control
from scripts.study_task1_population_tree import PopulationTree, novel_queries
from scripts.study_task1_unseen_profiles import FOLDS, STUDY
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import tokens
from superagent.models import _clean_categories
from superagent.profile_history import discover_columns

PLAN = ROOT / "research/task1_feedback_extension_1127/novel_profile_runnerup.json"
STAGE = "population_runnerup_1128"


def brier_gain(panels):
    return sum(r["brier_change"]*r["rows"] for r in panels)/sum(r["rows"] for r in panels)


def nominate(out):
    root = out / STAGE
    root.mkdir(exist_ok=False)
    records = {}
    for i in FOLDS:
        bank = out / POPULATION / str(i)
        strict = read(bank / "decision.json")
        panels = []
        for name in PHASES:
            path = bank / f"{name}-feedback.json"
            require(sha(path)==strict["feedback"][name], "Changed training feedback")
            panels.append(read(path)["feedback"])
        options = {}
        for w in SHARES[1:]:
            evidence = [p[str(w)] for p in panels]
            if accuracy_pass(evidence) and brier_gain(evidence)<=0:
                floor = min(r["net"]/r["rows"] for r in evidence)
                pooled = sum(r["net"] for r in evidence)/sum(r["rows"] for r in evidence)
                options[w] = (-floor, -pooled, brier_gain(evidence), w)
        nominee = min(options, key=options.get) if options and not strict["chosen"] else 0.
        records[str(i)] = dict(weight=nominee, incumbent_share=strict["chosen"], prior_decision=sha(bank/"decision.json"),
                               shortlist=list(options), previous_panels=[p[str(nominee)] for p in panels])
    save(root / "nominations.json",dict(protocol=sha(PLAN), folds=records))
    print("third-panel nominations", [(i,d["weight"]) for i,d in records.items()], flush=True)


def confirm(out):
    root = out / STAGE
    nominated = read(root / "nominations.json")
    require(nominated["protocol"] == sha(PLAN), "Nominations belong to another rule")
    records = {}
    for i in FOLDS:
        folder = out / STUDY / str(i)
        bank = out / POPULATION / str(i)
        source = folder / "replicated"
        area = root / str(i)
        area.mkdir(exist_ok=False)
        state = read(bank / "built.json")
        old = nominated["folds"][str(i)]
        require(sha(bank/"decision.json") == old["prior_decision"], "Old decision changed")
        weight, chosen = old["weight"], old["incumbent_share"]
        extra, routes, third = 0, 0, None
        if weight:
            training = folder / "anonymous/training.zip"
            require(sha(training) == read(folder / "inputs.json")["entry_inputs"], "Changed inner training rows")
            x, y = arrays(training, ["C_train.npy", "y_train.npy"])
            classes = np.unique(y)
            table = _clean_categories(x)
            groups = np.asarray(tokens(table[:,discover_columns(table)]))
            splits = list(StratifiedGroupKFold(5,shuffle=True,random_state=39013).split(x,y,groups))
            fit, held = splits[1]
            require(set(groups[held]).isdisjoint(groups[fit]), "Third profile panel overlaps its training data")
            for name, idx in zip(PHASES,(0,3),strict=True):
                binding = read(bank / f"{name}-seed-review.json")
                require(binding["held"] == key(splits[idx][1]) and set(groups[held]).isdisjoint(groups[splits[idx][1]]),
                        "Third panel reused scoring groups")
            rc = read(source/"sealed.json")
            base, _, extra = rebuild_control(source/"run/agent",rc["members"],fit,held,x,y,classes,area,"third")
            mask = novel_queries(x[fit], x[held])
            proposal = base
            if mask.any():
                save(area/"third-expert.json",dict(fit=key(fit),held=key(held)))
                model = PopulationTree().fit(x[fit],y[fit])
                require(np.array_equal(mask,model.novel(x[held])), "Wrong third-panel novelty mask")
                proposal = ordered(model,x[held],classes)
                extra += 1
                routes = model.audit["supervised_route_estimations"]
            third = metrics(combine(base,proposal,mask,weight),y[held]==classes[1],base)
            evidence = old["previous_panels"]+[third]
            if accuracy_pass(evidence) and brier_gain(evidence)<=0:
                chosen = weight
        require(state["upper_bound_path_fits"]+extra<=64, "Third panel exceeded the fit allowance")
        scores = bank / "sealed_features.npz"
        require(sha(scores)==state["probabilities"], "Frozen deployment probabilities changed")
        with np.load(scores,allow_pickle=False) as f:
            p = combine(f["base"],f["expert"],f["novel"],chosen)
            labels = f["classes"][(p>.5).astype(int)]
        dst = area / "chosen.npy"
        np.save(dst,labels,allow_pickle=False)
        record = area / "sealed.json"
        save(record,dict(nomination=sha(root/"nominations.json"),previous_share=old["incumbent_share"],
                         nominee=weight,chosen=chosen,feedback=third,new_fits=extra,route_constructions=routes,prediction=sha(dst)))
        records[str(i)] = sha(record)
    save(root/"score_lock.json",dict(protocol=sha(PLAN),folds=records))


def grade(out):
    root = out/STAGE
    lock = read(root/"score_lock.json")
    records, groups, hits = [], [], {k:[] for k in ("ordinary", "strict", "runnerup")}
    for i in FOLDS:
        bank, study, area = out/POPULATION/str(i), out/STUDY/str(i), root/str(i)
        record = read(area/"sealed.json")
        prior = read(bank/"sealed.json")["predictions"]
        require(sha(area/"sealed.json")==lock["folds"][str(i)], "Third-panel output is not locked")
        answers = study/"outer_answers.joblib"
        require(sha(answers)==read(study/"inputs.json")["external_answers"], "Wrong held answer packet")
        truth = joblib.load(answers)
        y = truth["y"].astype(str)
        groups.extend(truth["groups"])
        counts = {}
        for n,path,digest in (("ordinary",bank/"control.npy",prior["control"]),
                              ("strict",bank/"chosen.npy",prior["chosen"]),("runnerup",area/"chosen.npy",record["prediction"])):
            require(sha(path)==digest, "Prediction changed after locking")
            labels = np.load(path,allow_pickle=False)
            require(labels.shape==y.shape,"Unexpected output rows")
            bit = labels==y
            counts[n] = int(bit.sum())
            hits[n].extend(bit)
        records.append(dict(fold=i,rows=len(y),correct=counts,**{k:record[k] for k in
                            ("previous_share","nominee","chosen","feedback","new_fits","route_constructions")}))
    totals = {k:int(np.sum(h)) for k,h in hits.items()}
    paired = {k:compare(np.asarray(hits["runnerup"]),np.asarray(hits[k]),groups) for k in ("ordinary","strict")}
    advance = all(r["correct"]["runnerup"]>r["correct"]["ordinary"] for r in records) and paired["ordinary"]["gain"]>=.003
    result = dict(lock=sha(root/"score_lock.json"),rows=len(groups),folds=records,correct=totals,compared=paired,
                  advance=bool(advance),new_fits=sum(r["new_fits"] for r in records),
                  route_constructions=sum(r["route_constructions"] for r in records),model_api_requests=0,
                  designated_validation_opened=False,official_test_opened=False,h2_opened=False)
    save(root/"result.json",result)
    print("third-panel outcome",totals,advance,result["new_fits"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("nominate","confirm","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    {"nominate":nominate,"confirm":confirm,"grade":grade}[args.action](args.out)
