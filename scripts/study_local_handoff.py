"""Confirm a fourth routing mechanism without exposing fresh evaluation outcomes to tuning."""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np

from scripts.prepare_conditional_handoff import ROLES
from scripts.run_conditional_handoff import fresh_score, historical_panels, load_export, score_panel
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.local_preference_handoff import build_memory, competence, learn_threshold, route
from tools.preference_handoff import changes, inputs

POLICY_SOURCE=ROOT/"tools/local_preference_handoff.py"
PLAN=ROOT/"docs/batches/unified_local_handoff_plan_20261124_zh.md"
SCRIPT=Path(__file__).resolve()


def xy(out,run,fold):
    if fold is None:
        data,labels,meta=historical_panels(out,run)
        task=load_training_task(run/"training.zip")
        group=sample_groups(task,"shared_text")
        prior=set()
        panels={}
        for role,split in (("learn","primary"),("confirm","confirmation_1")):
            ix=read(run/"agent"/(split+"_split.json"))["search_indices"]
            kept=[i for i in ix if group[i] not in prior]
            prior.update(group[ix])
            require(fingerprint(kept)==meta[role]["ids"],"Learning/confirmation rows changed")
            panels[role]={"pairs":[task.X_train[i] for i in kept],"groups":group[kept]}
        return data,labels,panels,out/"expanded",out/"local_replay"
    folder=out/f"fold_{fold}"
    protocol=read(out/"protocol.json")["splits"][fold]
    require(sha(folder/"feedback.npz")==protocol["feedback"] and sha(folder/"features.joblib")==protocol["features"]
            and sha(folder/"probabilities.npz")==read(folder/"scores_frozen.json")["scores"],"Frozen fold changed")
    panels=joblib.load(folder/"features.joblib")
    with np.load(folder/"feedback.npz",allow_pickle=False) as yy, np.load(folder/"probabilities.npz",allow_pickle=False) as p:
        data={k:inputs(p[k],panels[k]["pairs"]) for k in ("learn","confirm")}
        labels={k:yy[k] for k in data}
    return data,labels,panels,folder/"routing",folder/"local"


def current_rule(earlier,local_eligible,confirm):
    name=earlier["chosen"]
    if not local_eligible:
        return name
    if name=="baseline":
        return "local"
    old=earlier["evidence"][name]["confirmation"]
    return "local" if (confirm["net"],-confirm["switches"])>(old["net"],-old["switches"]) else name


def fit(out,run,fold):
    data,ys,panels,earlier,target=xy(out,run,fold)
    target.mkdir(exist_ok=True)
    prev=read(earlier/"policy.json")["decision"]
    if not (target/"decision.json").exists():
        with (target/"memory.admission.json").open("x") as f:
            f.write('{"maximum_supervised_fits":1}\n')
        memory=build_memory(data["learn"],panels["learn"]["pairs"],panels["learn"]["groups"],ys["learn"])
        joblib.dump(memory,target/"memory.joblib",compress=1)
        learn_rates=competence(memory,data["learn"],panels["learn"]["pairs"],panels["learn"]["groups"])
        with (target/"threshold.admission.json").open("x") as f:
            f.write('{"maximum_supervised_fits":1}\n')
        threshold,options=learn_threshold(data["learn"],ys["learn"],learn_rates)
        learn=changes(data["learn"]["base"],route(data["learn"],learn_rates,threshold),ys["learn"])
        confirm_rates=competence(memory,data["confirm"],panels["confirm"]["pairs"],panels["confirm"]["groups"])
        confirm=changes(data["confirm"]["base"],route(data["confirm"],confirm_rates,threshold),ys["confirm"])
        minimum=max(2,math.ceil(.002*len(ys["confirm"])))
        eligible=learn["net"]>0 and confirm["net"]>=minimum
        save(target/"decision.json",dict(previous=prev["chosen"], chosen=current_rule(prev,eligible,confirm),
             threshold=threshold,options=options,eligible=eligible,learning=learn,confirmation=confirm,minimum=minimum,
             memory_examples=memory["size"],memory=sha(target/"memory.joblib"),prior_policy=sha(earlier/"policy.json"),
             plan=sha(PLAN),model_source=sha(POLICY_SOURCE),study=sha(SCRIPT),supervised_memory_fits=1,
             supervised_threshold_selections=1))
    files=[]
    for i in range(2):
        dest=target/f"predictions_{i}.csv"
        if not dest.exists():
            cmd=[sys.executable,"-m","scripts.study_local_handoff","predict","--out",str(out),"--output",str(dest)]
            cmd+=["--run",str(run)] if fold is None else ["--fold",str(fold)]
            subprocess.run(cmd,cwd=ROOT,check=True,stdin=subprocess.DEVNULL,timeout=60)
        files.append(sha(dest))
    require(files[0]==files[1],"Local handoff was not reproducible")
    save(target/"predictions_frozen.json",{"predictions":files,"decision":sha(target/"decision.json")})


def infer(out,run,fold,output):
    if fold is None:
        pairs=load_prediction_features(run/"audit_features.zip","pair_text",read(run/"agent/task_spec.json"))
        require(sha(out/"historical_probabilities.npy")==read(out/"historical_inputs.json")["probabilities"],"Wrong frozen scores")
        data=inputs(np.load(out/"historical_probabilities.npy",allow_pickle=False),pairs)
        groups=sample_groups(SimpleNamespace(kind="pair_text",X_train=pairs),"shared_text")
        ids=list(range(len(pairs)))
        earlier,target=out/"expanded",out/"local_replay"
        old_csv=earlier/"predictions.csv"
        old_sha=read(earlier/"predictions_frozen.json")["predictions"]
    else:
        folder=out/f"fold_{fold}"
        require(sha(folder/"features.joblib")==read(out/"protocol.json")["splits"][fold]["features"] and
                sha(folder/"probabilities.npz")==read(folder/"scores_frozen.json")["scores"],"Wrong fresh probabilities/features")
        obs=joblib.load(folder/"features.joblib")["outer"]
        pairs,groups,ids=obs["pairs"],obs["groups"],obs["ids"]
        with np.load(folder/"probabilities.npz",allow_pickle=False) as p:
            data=inputs(p["outer"],pairs)
        earlier,target=folder/"routing",folder/"local"
        old_csv=earlier/"predictions_0.csv"
        old_sha=read(earlier/"predictions_frozen.json")["prediction_replicas"][0]
    decision=read(target/"decision.json")
    require(sha(old_csv)==old_sha and sha(earlier/"policy.json")==decision["prior_policy"] and
            sha(target/"memory.joblib")==decision["memory"] and sha(POLICY_SOURCE)==decision["model_source"],"Wrong local policy")
    cols=load_export(old_csv)
    require(cols["row_id"].tolist()==ids and np.array_equal(cols["baseline"],data["base"]),"Prediction alignment differs")
    estimated=competence(joblib.load(target/"memory.joblib"),data,pairs,groups)
    cols["previous_agent"]=cols.pop("agent")
    cols["local"]=route(data,estimated,decision["threshold"])
    cols["agent"]=cols["local"] if decision["chosen"]=="local" else cols["previous_agent"]
    with output.open("x",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(cols)
        writer.writerows(zip(*cols.values(),strict=True))


def check_export(folder):
    seal=read(folder/"predictions_frozen.json")
    require(seal["predictions"]==[sha(folder/f"predictions_{i}.csv") for i in range(2)] and
            seal["decision"]==sha(folder/"decision.json"),"Unsealed local routing output")
    return load_export(folder/"predictions_0.csv")


def all_scores(out):
    folds=[out/f"fold_{i}" for i in range(5)]
    seals=[{"original":sha(p/"routing/predictions_frozen.json"),"local":sha(p/"local/predictions_frozen.json")} for p in folds]
    if not (out/"all_handoffs_frozen.json").exists():
        require(not (out/"five_fold_results.json").exists(),"Five-fold answers were opened before the fourth policy froze")
        save(out/"all_handoffs_frozen.json",seals)
    require(read(out/"all_handoffs_frozen.json")==seals,"Method seal changed")
    fresh_score(out)
    panels,cols_global,ys,groups=[],{ },[],[]
    protocol=read(out/"protocol.json")
    for n,p in enumerate(folds):
        cols=check_export(p/"local")
        truth=out/"scoring_only"/f"fold_{n}.json"
        require(sha(truth)==protocol["splits"][n]["targets"],"Targets differ")
        raw=read(truth)
        require(cols["row_id"].tolist()==raw["ids"] and not set(groups).intersection(raw["groups"]),"Repeated outer group")
        perf=score_panel(cols,raw["labels"],raw["groups"])
        panels.append({"fold":n,"chosen":read(p/"local/decision.json")["chosen"],**perf})
        groups.extend(raw["groups"]); ys.extend(raw["labels"])
        for k,v in cols.items():
            cols_global.setdefault(k,[]).append(v)
    all_cols={k:np.concatenate(v) for k,v in cols_global.items()}
    pooled=score_panel(all_cols,ys,groups)
    from scripts.run_conditional_handoff import comparison
    delta=comparison(all_cols["previous_agent"],all_cols["agent"],np.asarray(ys),groups)
    actual=pooled["methods"]["agent"]
    positive=sum(p["methods"]["agent"]["net"]>0 for p in panels)
    result={"folds":panels,"pooled":pooled,"positive_folds":positive,"agent_versus_previous_agent":delta,
            "passes":bool(positive>=4 and actual["net"]>=5 and actual["paired_95_fraction"][0]>0),
            "model_api_calls":0,"course_validation_used":False,"course_test_used":False}
    save(out/"local_five_fold_results.json",result)
    print("Extended policy by fold:",[(p["chosen"],p["methods"]["agent"]["net"]) for p in panels],
          "pooled",actual,"upgrade",result["passes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","predict","replay","score"))
    for k in ("out","run","train-study","output"):
        p.add_argument("--"+k,type=lambda s:Path(s).resolve(),required=k=="out")
    p.add_argument("--fold",type=int)
    a=p.parse_args()
    if a.stage=="score":
        all_scores(a.out)
    elif a.stage=="predict":
        infer(a.out,a.run,a.fold,a.output)
    elif a.stage=="freeze":
        for n in ([a.fold] if a.fold is not None else range(5)):
            fit(a.out,None,n)
    else:
        fit(a.out,a.run,None)
        truth=a.train_study/"development_truth.json"
        require(sha(truth)==read(a.train_study/"protocol.json")["development_truth"],"Historical training truth differs")
        target=read(truth)
        report=score_panel(check_export(a.out/"local_replay"),target["labels"],target["groups"])
        save(a.out/"local_replay/result.json",report)
        print("Historical memory:",read(a.out/"local_replay/decision.json")["chosen"],report["methods"]["local"],flush=True)
