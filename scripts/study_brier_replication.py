"""Cold native trial of a locked nomination rule on regrouped development data."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.evaluate_pilot_nomination import project, store_arrays
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_crossfit_challenger import native_predict, study_splits
from scripts.study_encoder_fusion import save_csv
from scripts.study_pilot_nomination import current_components, link_file, run_arm
from scripts.unified_released_study import ROOT, read, require, save, sha, text_packet
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from tools.challenger_training import rows_at, score_members
from tools.crossfit_challenger import ranked_groups
from tools.pilot_fit_cache import recipe
from tools.portfolio_training import configure, write_selection
from tools.preference_handoff import changes

PLAN = ROOT/"docs/batches/pilot_brier_replication_plan_20261124_zh.md"
RULE = "pilot-brier"
SALT = "pilot-replicate-outer-v2/"


def signature():
    return {n:sha(ROOT/n) for n in ("unified_agent.py","tools/pilot_challenger.py","tools/crossfit_challenger.py",
            "tools/challenger_training.py","tools/pilot_fit_cache.py","scripts/unified_encoder_portfolio.py",
            "tools/feedback_policy_pool.py","scripts/study_brier_replication.py")}


def partitions(archive):
    task,groups,original = study_splits(archive)
    main = np.sort(np.concatenate([held for _,held in original]))
    codes = {g:i%5 for i,g in enumerate(ranked_groups(groups[main],SALT))}
    positions = np.asarray([codes[g] for g in groups[main]])
    splits = [(main[positions!=i],main[positions==i]) for i in range(5)]
    for i,(train,held) in enumerate(splits):
        require(not set(groups[train]) & set(groups[held]) and len(held) and
                not np.array_equal(held,original[i][1]),"Invalid new grouped split")
    return task,groups,splits,original


def smoke(out, parent, weights, cache):
    source = read(parent/"smoke.json")
    for file,key in (("train.zip","train"),("features.zip","features")):
        require(sha(parent/file) == source[key], "Earlier anonymous input changed")
        link_file(parent/file,out/file)
    end = run_arm(out,RULE,current_components(parent/RULE),weights,cache)
    pred = native_predict(out/RULE,weights)
    first, current = parent/RULE/"run", out/RULE/"run"
    require(end["new_classifier_fits"]==0 and len(pred)==24 and
            sha(out/RULE/"direct/predictions.csv")==source["arms"][RULE]["export"] and
            read(first/"nomination.json")==read(current/"nomination.json") and
            read(first/"decision.json")["approved"]==read(current/"decision.json")["approved"],
            "Locked native rule changed its anonymous decision")
    save(out/"smoke.json", dict(parent=sha(parent/"smoke.json"), rows=len(pred),new_classifier_fits=0,
         model_api_calls=0,source=signature(),decision=sha(current/"decision.json"),export=sha(out/RULE/"direct/predictions.csv")))


def prepare(out, archive):
    out.mkdir(parents=True,exist_ok=False)
    task,groups,splits,old = partitions(archive)
    registry = {}
    for i,(train,held) in enumerate(splits):
        folder=out/f"fold_{i}"
        folder.mkdir()
        text_packet(folder/"train.zip",[(*task.X_train[j],int(task.y_train[j])) for j in train],True)
        text_packet(folder/"features.zip",rows_at(task,held),False)
        registry[str(i)] = {k:dict(ids=fingerprint(ids.tolist()),rows=len(ids),groups=len(set(groups[ids])))
                            for k,ids in (("train",train),("outer",held))}
        registry[str(i)].update(training=sha(folder/"train.zip"),features=sha(folder/"features.zip"),
                                rows_from_original_folds=[len(np.intersect1d(held,x)) for _,x in old])
    save(out/"protocol.json",dict(plan=sha(PLAN),source=signature(),fit_recipe=recipe(),folds=registry,
                                   budget=45, model_api_calls=0, nomination=RULE, group_salt=SALT))


def train(out, smoke_path, weights, cache):
    protocol=read(out/"protocol.json")
    require(protocol["plan"]==sha(PLAN) and protocol["source"]==read(smoke_path/"smoke.json")["source"]==signature()
            and protocol["fit_recipe"]==recipe(),"Rule or software changed before replication")
    for i in range(5):
        folder=out/f"fold_{i}"
        require(sha(folder/"train.zip")==protocol["folds"][str(i)]["training"],"Changed cold input")
        end=run_arm(folder,RULE,[],weights,cache)
        require(end["reused_classifier_fits"]==0 and 8<=end["new_classifier_fits"]<=9,"Cold fit budget differs")
        print(folder.name,end["new_classifier_fits"],"new fits",flush=True)


def freeze_one(folder, weights, cache):
    arm = folder/RULE
    run = arm/"run"
    end,decision,nom = (read(run/f) for f in ("pilot_completion.json","decision.json","nomination.json"))
    selection = read(run/"agent/final_selection.json")
    task = load_training_task(folder/"train.zip")
    input_spec = read(run/"agent/task_spec.json")
    raw = load_prediction_features(folder/"features.zip","pair_text",input_spec)
    require(end["decision_sha256"]==sha(run/"decision.json") and
            selection["model_sha256"]==end["model_sha256"]==sha(run/"agent/selected_model.joblib"),"Unsealed delivery")
    members = selection["ensemble_components"]
    require(members==nom["members"][:2]+([decision["approved"]] if decision["approved"] else []),"Refit order differs")
    models=[]
    records={}
    for name in members:
        path=run/"refit"/name
        r=read(path/"trained.json")
        require(r["model"]==sha(path/"model.joblib") and r["ledger"]==sha(path/"fits.json")
                and r["identity"]==dict(package=sha(folder/"train.zip"),ids=fingerprint(list(range(len(task.y_train))))),
                "Wrong full-data refit")
        records[name]=r["model"]
        models.append(joblib.load(path/"model.joblib"))
    baseline=folder/"baseline"
    link_file(folder/"features.zip",baseline/"features.zip")
    baseline_run=baseline/"run"
    baseline_run.mkdir(exist_ok=True)
    if not (baseline_run/"agent/final_selection.json").exists():
        vote=PreferenceProbabilityVote("pair_text", models[:2], np.array([0,1]))
        write_selection(baseline_run/"agent",vote,members[:2],task.spec,"baseline","matched-pilot-reference")
    require(read(baseline_run/"agent/final_selection.json")["model_sha256"]==sha(baseline_run/"agent/selected_model.joblib"),
            "Reference model changed")
    b,d = native_predict(baseline,weights),native_predict(arm,weights)
    configure(weights,cache)
    with reuse_frozen_vectors():
        probabilities=score_members(models,raw)
    projected=project(members,probabilities,raw)
    require(np.array_equal(b,projected["baseline"]) and np.array_equal(d,projected[decision["approved"] or "baseline"]),
            "New-process raw delivery differs")
    store_arrays(folder/"scores.npz",{"scores":probabilities})
    columns=dict(baseline=b,agent=d)
    rowfile=folder/"rows.csv"
    if not rowfile.exists():
        save_csv(rowfile,range(len(raw)),columns)
    observed=load_export(rowfile)
    require(np.array_equal(observed.pop("row_id"),np.arange(len(raw))) and observed.keys()==columns.keys()
            and all(np.array_equal(v,observed[k]) for k,v in columns.items()),"Different frozen rows")
    names = ["scores.npz","rows.csv","baseline/direct/predictions.csv","baseline/run/agent/final_selection.json"]
    names.extend(f"{RULE}/{f}" for f in ("run/pilot_protocol.json","run/pilot_completion.json","run/feedback.npz","run/decision.json",
                 "run/nomination.json","run/agent/final_selection.json","direct/predictions.csv"))
    save(folder/"sealed.json",dict(records={f:sha(folder/f) for f in names},members=members,refit_models=records,
         fits=end["new_classifier_fits"],train_seconds=end["elapsed_seconds"],
         native_seconds={k:read(v/"direct/elapsed.json")["seconds"] for k,v in (("baseline",baseline),("agent",arm))}))
    print(folder.name,"native rows",len(raw),flush=True)


def freeze(out, weights, cache):
    protocol=read(out/"protocol.json")
    require(protocol["source"]==signature() and protocol["plan"]==sha(PLAN),"Registered source changed")
    hashes=[]
    for i in range(5):
        f=out/f"fold_{i}"
        require(sha(f/"features.zip")==protocol["folds"][str(i)]["features"],"Unexpected evaluation features")
        if not (f/"sealed.json").exists():
            freeze_one(f,weights,cache)
        require(all(sha(f/k)==v for k,v in read(f/"sealed.json")["records"].items()),"Existing export changed")
        hashes.append(sha(f/"sealed.json"))
    save(out/"frozen.json",dict(protocol=sha(out/"protocol.json"),folds=hashes))


def score(out, archive):
    task,groups,splits,_=partitions(archive)
    frozen=read(out/"frozen.json")
    require(frozen["protocol"]==sha(out/"protocol.json") and read(out/"protocol.json")["plan"]==sha(PLAN),"Wrong protocol")
    folds,base,agent,indices=[],[],[],[]
    for i,(_,held) in enumerate(splits):
        folder=out/f"fold_{i}"
        seal=read(folder/"sealed.json")
        require(frozen["folds"][i]==sha(folder/"sealed.json") and all(sha(folder/k)==v for k,v in seal["records"].items())
                and fingerprint(held.tolist())==read(out/"protocol.json")["folds"][str(i)]["outer"]["ids"],"Fold changed")
        p=load_export(folder/"rows.csv")
        require(np.array_equal(p["row_id"],np.arange(len(held))),"Wrong export order")
        m=changes(p["baseline"],p["agent"],task.y_train[held])
        d=read(folder/RULE/"run/decision.json")
        folds.append(dict(fold=i,rows=len(held),nominee=d["nominee"],approved=d["approved"],fits=seal["fits"],
                          confirmation=d["confirm"],performance=m,train_seconds=seal["train_seconds"],native_seconds=seal["native_seconds"]))
        base.extend(p["baseline"])
        agent.extend(p["agent"])
        indices.extend(held)
    ix=np.asarray(indices)
    require(len(ix)==len(set(ix))==6608 and len(set(groups[ix]))==4874,"Incomplete coverage")
    result=comparison(np.asarray(base),np.asarray(agent),task.y_train[ix],groups[ix])
    wins=sum(f["performance"]["net"]>0 for f in folds)
    total=sum(f["fits"] for f in folds)
    require(total<=45,"Cold study budget exceeded")
    passed=wins>=4 and result["net"]>=5 and result["paired_95_fraction"][0]>0
    save(out/"result.json", dict(frozen=sha(out/"frozen.json"),folds=folds,performance=result,positive_folds=wins,
         passes=passed,actual_new_classifier_fits=total,rows=len(ix),groups=len(set(groups[ix])),
         model_api_calls=0,model_api_rmb=0,designated_validation_used=False,formal_test_used=False))
    print("cold regrouping",result,wins,passed,"fits",total)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","smoke","train","freeze","score"))
    for arg in ("out","archive","parent","smoke","weights","cache"):
        p.add_argument("--"+arg, required=arg=="out", type=lambda x:Path(x).resolve())
    a=p.parse_args()
    if a.stage=="smoke":
        smoke(a.out,a.parent,a.weights,a.cache)
    elif a.stage=="prepare":
        prepare(a.out,a.archive)
    elif a.stage=="train":
        train(a.out,a.smoke,a.weights,a.cache)
    elif a.stage=="freeze":
        freeze(a.out,a.weights,a.cache)
    else:
        score(a.out,a.archive)
