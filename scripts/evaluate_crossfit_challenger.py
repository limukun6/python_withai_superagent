"""Freeze matched controls and native predictions before scoring any outer preference labels."""
import argparse
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_crossfit_challenger import PLAN, native_predict, study_splits
from scripts.study_encoder_fusion import save_csv
from scripts.unified_released_study import read, require, save, sha
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import score_members, ties
from tools.crossfit_challenger import combine, select, training_roles
from tools.portfolio_training import configure, fit_one
from tools.preference_handoff import binary
from tools.preference_portfolio import choose, panel, probabilities


def decision_from_feedback(names, task, roles, scores):
    groups = sample_groups(task,"shared_text")
    data, prior = {},{}
    for role in ("a","b"):
        ids = roles[role]
        require(np.array_equal(ids,scores[role+"_ids"]),"OOF identity differs")
        rows = [task.X_train[i] for i in ids]
        p = scores[role]
        data[role] = dict(scores=p,ties=ties(rows),labels=task.y_train[ids],groups=groups[ids])
        prior[role] = panel(p,rows)
    legacy = choose(names,prior["a"],data["a"]["labels"],prior["b"],data["b"]["labels"])
    return select(data),legacy


def project(matrix, rows, decision, legacy):
    t = ties(rows)
    base = binary(matrix[:,:2].mean(1),t)
    columns = {"baseline":base,"original_agent":base,"equal_three":base,"learned_weight":base,"agent":base}
    if legacy is not None:
        columns["equal_three"] = binary(matrix.mean(1),t)
        columns["learned_weight"] = binary(combine(matrix,t,decision["weights"]),t)
        columns["original_agent"] = binary(probabilities(legacy["rules"][legacy["chosen"]],panel(matrix,rows)),t)
        if decision["chosen"]=="agreement_weight":
            columns["agent"] = columns["learned_weight"]
    return columns


def freeze_fold(folder,protocol,weights,cache):
    run = folder/"run"
    require(sha(folder/"train.zip")==protocol["train_packet"] and sha(folder/"features.zip")==protocol["features_packet"],
            "Changed anonymous input")
    started = time.monotonic()
    end = read(run/"challenger_completion.json")
    decision = read(run/"decision.json")
    require(sha(run/"decision.json")==end["decision_sha256"] and sha(run/"agent/selected_model.joblib")==end["model_sha256"],
            "Autonomous decision must be frozen before comparison")
    configure(weights,cache)
    task = load_training_task(folder/"train.zip")
    rows = load_prediction_features(folder/"features.zip","pair_text",read(run/"agent/task_spec.json"))
    names = decision["members"]
    legacy = None
    added = 0
    models,model_records = [],[]
    with reuse_frozen_vectors():
        if len(names)==3:
            roles = training_roles(sample_groups(task,"shared_text"),task.y_train)
            require(sha(run/"oof.npz")==decision["oof"],"Changed OOF feedback")
            with np.load(run/"oof.npz",allow_pickle=False) as scores:
                check,legacy = decision_from_feedback(names,task,roles,scores)
            require(all(decision[k]==v for k,v in check.items()),"Fresh two-fold decision differs")
        save(folder/"control.json",{"decision_sha256":end["decision_sha256"],"legacy":legacy})
        for i,name in enumerate(names):
            path = run/"refit"/name
            if not path.exists():
                require(i==2 and decision["chosen"]=="baseline","Missing autonomous refit")
                path = folder/"comparison_refit"/name
                ids = np.arange(len(task.y_train))
                fit_one(path,name,task.X_train,task.y_train,{"package":protocol["train_packet"],"ids":fingerprint(ids.tolist())})
                added += 1
            rec = read(path/"trained.json")
            require(rec["model"]==sha(path/"model.joblib") and rec["ledger"]==sha(path/"fits.json"),"Refit identity differs")
            model_records.append({"name":name,"model":rec["model"],"ledger":rec["ledger"]})
            models.append(joblib.load(path/"model.joblib"))
        matrix = score_members(models,rows)
    probability_path = folder/"outer_scores.npz"
    if probability_path.exists():
        with np.load(probability_path,allow_pickle=False) as saved:
            require(np.array_equal(saved["probabilities"],matrix),"Resumed refit probabilities differ")
    else:
        np.savez_compressed(probability_path,probabilities=matrix)
    columns = project(matrix,rows,decision,legacy)
    prediction_path = folder/"rows.csv"
    if prediction_path.exists():
        expected = {"row_id":np.arange(len(rows)),**columns}
        saved = load_export(prediction_path)
        require(saved.keys()==expected.keys() and all(np.array_equal(saved[k],v) for k,v in expected.items()),
                "Resumed decision rows differ")
    else:
        save_csv(prediction_path,range(len(rows)),columns)
    native = native_predict(folder,weights)
    require(np.array_equal(native,columns["agent"]),"Cold raw-title export differs from the frozen rule")
    require(end["classifier_fits"]+added<=9,"Paired budget exceeded")
    seals = {file:sha(folder/file) for file in ("rows.csv","control.json","outer_scores.npz","direct/predictions.csv",
             "run/challenger_completion.json","run/decision.json","run/nomination.json","run/nomination.npz")}
    if len(names)==3:
        seals["run/oof.npz"] = sha(run/"oof.npz")
    result = dict(records=seals,models=model_records,autonomous_fits=end["classifier_fits"],
                  comparison_only_fits=added,old_weight_fits=2 if legacy else 0,
                  old_discrete_shortlists=2 if legacy else 0,training_seconds=end["elapsed_seconds"],
                  raw_export_seconds=read(folder/"direct/elapsed.json")["seconds"],
                  comparison_and_raw_export_seconds=round(time.monotonic()-started,4))
    save(folder/"sealed.json",result)
    print(folder.name,"autonomous fits",end["classifier_fits"],"comparison fits",added,"native predictions",len(native),flush=True)


def freeze(out,weights,cache):
    protocol = read(out/"protocol.json")
    require(protocol["plan"]==sha(PLAN),"Unregistered protocol")
    hashes=[]
    for i in range(5):
        folder = out/f"fold_{i}"
        if not (folder/"sealed.json").exists():
            freeze_fold(folder,protocol["folds"][str(i)],weights,cache)
        sealed=read(folder/"sealed.json")
        require(all(sha(folder/k)==v for k,v in sealed["records"].items()),"Stored comparison changed")
        hashes.append(sha(folder/"sealed.json"))
    save(out/"sealed.json",{"protocol":sha(out/"protocol.json"),"folds":hashes})


def score(out,archive):
    frozen=read(out/"sealed.json")
    require(frozen["protocol"]==sha(out/"protocol.json"),"Changed plan after prediction freeze")
    task,groups,splits = study_splits(archive)
    folds, cols, ids_all = [],{},[]
    seen=set()
    protocol=read(out/"protocol.json")
    for i,(_,held) in enumerate(splits):
        folder=out/f"fold_{i}"
        sealed=read(folder/"sealed.json")
        require(sha(folder/"sealed.json")==frozen["folds"][i] and
                all(sha(folder/k)==v for k,v in sealed["records"].items()),"Unsealed fold")
        require(fingerprint(held.tolist())==protocol["folds"][str(i)]["outer"]["ids"] and not set(groups[held]) & seen,
                "Repeated or mismatched groups in outer scoring")
        seen.update(groups[held])
        pred=load_export(folder/"rows.csv")
        require(np.array_equal(pred.pop("row_id"),np.arange(len(held))),"Outer row order differs")
        ids_all.extend(held)
        for k,v in pred.items():
            cols.setdefault(k,[]).append(v)
        dec=read(folder/"run/decision.json")
        prior=read(folder/"control.json")["legacy"]
        folds.append(dict(fold=i,rows=len(held),candidate=read(folder/"run/nomination.json")["candidate"],
                          agent=dec["chosen"],old=prior["chosen"] if prior else "baseline",
                          autonomous_fits=sealed["autonomous_fits"],comparison_only_fits=sealed["comparison_only_fits"],
                          train_seconds=sealed["training_seconds"],cold_predict_seconds=sealed["raw_export_seconds"],
                          arms={k:comparison(pred["baseline"],v,task.y_train[held],groups[held]) for k,v in pred.items()}))
    ids=np.asarray(ids_all)
    require(len(ids)==len(set(ids))==6608,"Mismatched coverage")
    data={k:np.concatenate(v) for k,v in cols.items()}
    arms={k:comparison(data["baseline"],v,task.y_train[ids],groups[ids]) for k,v in data.items()}
    wins=sum(row["arms"]["agent"]["net"]>0 for row in folds)
    current=arms["agent"]
    result=dict(fold_seal=sha(out/"sealed.json"),rows=len(ids),groups=len(seen),folds=folds,arms=arms,
                positive_folds=wins,passes=wins>=4 and current["net"]>=5 and current["paired_95_fraction"][0]>0,
                autonomous_fits=sum(f["autonomous_fits"] for f in folds),
                comparison_only_fits=sum(f["comparison_only_fits"] for f in folds),model_api_calls=0,
                model_api_rmb=0,designated_validation_used=False,formal_test_used=False)
    save(out/"result.json",result)
    print("scores",{k:(v["correct"],v["net"]) for k,v in arms.items()},wins,result["passes"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","score"))
    p.add_argument("--out",type=lambda x:Path(x).resolve(),required=True)
    for arg in ("weights","cache","archive"):
        p.add_argument("--"+arg,type=lambda x:Path(x).resolve())
    a=p.parse_args()
    if a.stage=="freeze":
        freeze(a.out,a.weights,a.cache)
    else:
        score(a.out,a.archive)
