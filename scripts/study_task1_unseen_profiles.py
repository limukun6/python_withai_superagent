"""Run the frozen ordinary entry with all evaluation profiles excluded from its training rows."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_extensions import key,npy
from scripts.study_task1_outer import arrays,compare,prediction_csv
from scripts.unified_released_study import ROOT,make_zip,read,require,save,sha
from superagent.joint_context import tokens
from superagent.models import _clean_categories,predict_with_model
from superagent.profile_history import discover_columns

PLAN=ROOT/"research/task1_feedback_extension_1127/unseen_profile_stress.json"
STUDY="unseen_profile_stress_1128"
FOLDS=(0,3)
RESTART=PLAN.with_name("serial_stress_recovery.json")


def completed_arm(folder):
    first=folder/"replicated"
    return folder/"replicated_serial" if (folder/"interruption.json").is_file() else first


def record_interrupted(out):
    for i in FOLDS:
        folder=out/STUDY/str(i)
        side=folder/"replicated"
        native=side/"run/agent"
        partial=read(side/"prediction_manifest.json")
        require(not (side/"sealed.json").exists() and not (native/"run_outcome.json").exists()
                and partial["delivery_status"]=="partial" and partial["prediction_sha256"]==sha(side/"solve.csv"),
                "Expected an interrupted run with an actual first delivery")
        physical=side/"physical.jsonl"
        save(folder/"interruption.json",dict(protocol=sha(RESTART),launch=sha(side/"launch.json"),
             physical_log=sha(physical),physical_attempts=len(physical.read_text().splitlines()),
             last_accounting=sha(native/"classifier_fits.json"),partial_prediction=sha(side/"solve.csv"),
             partial_rows=partial["prediction_count"]))


def prepare(out):
    source=out/"remaining_folds_source"
    train=source/"anonymous/training.zip"
    require(sha(train)==read(source/"inputs.json")["entry_inputs"],"Wrong course-training packet")
    x,y=arrays(train,["C_train.npy","y_train.npy"])
    table=_clean_categories(x)
    columns=list(discover_columns(table))
    require(bool(columns),"No structurally recurring profile")
    groups=np.asarray(tokens(table[:,columns]))
    exact=np.asarray(tokens(table))
    root=out/STUDY
    root.mkdir(exist_ok=False)
    selected=set()
    for i,(fit,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=481936).split(x,y,groups)):
        if i not in FOLDS:
            continue
        seen=set(groups[held])
        require(seen.isdisjoint(groups[fit]) and seen.isdisjoint(selected) and set(exact[held]).isdisjoint(exact[fit]),
                "A held profile or complete feature row occurred in training")
        selected.update(seen)
        folder=root/str(i)
        inputs=folder/"anonymous"
        inputs.mkdir(parents=True)
        fit_zip=inputs/"training.zip"
        q_zip=inputs/"queries.zip"
        make_zip(fit_zip,[("C_train.npy",npy(x[fit])),("y_train.npy",npy(y[fit])),
                          ("Readme.md",b"Binary categorical classification. The primary metric is accuracy.")])
        make_zip(q_zip,[("C_test.npy",npy(x[held]))])
        answers=folder/"outer_answers.joblib"
        joblib.dump(dict(y=y[held],groups=groups[held]),answers)
        labs,counts=np.unique(y[fit],return_counts=True)
        save(folder/"inputs.json",dict(protocol=sha(PLAN),source=sha(train),entry_inputs=sha(fit_zip),entry_queries=sha(q_zip),
             external_answers=sha(answers),profile_overlap=0,full_feature_overlap=0,majority=str(labs[counts.argmax()]),
             roles=dict(external=dict(rows=len(held),profiles=len(seen),digest=key(held)),
                        fit=dict(rows=len(fit),profiles=len(set(groups[fit])),digest=key(fit)))))
    save(root/"prepared.json",dict(protocol=sha(PLAN),source=sha(train),classes=len(set(y)),rows=len(y),
         profile_columns=len(columns),profiles=len(set(groups)),folds={str(i):sha(root/str(i)/"inputs.json") for i in FOLDS}))


def seal(out):
    root=out/STUDY
    original_source=read(out/"release/inputs.json")["source"]
    manifest=read(root/"prepared.json")
    records={}
    for i in FOLDS:
        folder=root/str(i)
        arm=completed_arm(folder)
        info=read(folder/"inputs.json")
        launch=read(arm/"launch.json")
        receipt=read(arm/"sealed.json")
        require(manifest["folds"][str(i)]==sha(folder/"inputs.json")==launch["inputs"] and launch["source"]==original_source
                and launch["method_hint"] is None and launch["cap"]==64 and launch["feedback_mode"]=="seed-replication"
                and receipt["status"]=="completed" and receipt["independent_reload"] and not receipt["model_api_requests"]
                and receipt["prediction"]==sha(arm/"solve.csv")==sha(arm/"reload.csv"),"Unverified ordinary export")
        src=arm/"run/agent"
        decision=read(src/"seed_replication.json")
        path=src/"selected_model.joblib"
        require(sha(path)==decision["incumbent"] and sha(folder/"anonymous/queries.zip")==info["entry_queries"],
                "Changed pre-pilot control or query input")
        model=joblib.load(path)
        q,=arrays(folder/"anonymous/queries.zip",["C_test.npy"])
        base=np.asarray(predict_with_model(model["kind"],model["model"],q)).astype(str)
        base_file=folder/"before_pilot.npy"
        np.save(base_file,base,allow_pickle=False)
        interrupted=folder/"interruption.json"
        previous=0
        if interrupted.is_file():
            prior=read(interrupted)
            require(prior["protocol"]==sha(RESTART) and sha(folder/"replicated/physical.jsonl")==prior["physical_log"]
                    and read(arm/"launch.json")==read(folder/"replicated/launch.json"), "Unmatched serial recovery")
            previous=prior["physical_attempts"]
        records[str(i)]=dict(original=sha(base_file),current=receipt["prediction"],fits=receipt["fits"],decision=decision["action"],
                members=receipt["members"],source=launch["source"],model=receipt["model"],nomination=decision.get("pilot"),
                interrupted_fits=previous,runtime_seconds=receipt["seconds"],labels=info["external_answers"])
        del model
    save(root/"score_lock.json",dict(protocol=sha(PLAN),solves=records))


def grade(out):
    root=out/STUDY
    lock=read(root/"score_lock.json")
    results,groups,hits=[],[],{n:[] for n in ("majority","before_pilot","frozen")}
    for i in FOLDS:
        data=lock["solves"][str(i)]
        folder=root/str(i)
        arm=completed_arm(folder)
        info=read(folder/"inputs.json")
        file=folder/"outer_answers.joblib"
        require(sha(file)==data["labels"] and sha(folder/"before_pilot.npy")==data["original"]
                and sha(arm/"solve.csv")==data["current"],"Changed stress-test packets")
        packet=joblib.load(file)
        y=packet["y"].astype(str)
        require(not set(packet["groups"]).intersection(groups),"Overlapping stress folds")
        groups.extend(packet["groups"])
        candidates=dict(majority=np.repeat(info["majority"],len(y)),before_pilot=np.load(folder/"before_pilot.npy",allow_pickle=False),
                        frozen=prediction_csv(arm/"solve.csv",len(y)))
        scores={}
        for name,p in candidates.items():
            k=p==y
            scores[name]=int(k.sum())
            hits[name].extend(k)
        results.append(dict(fold=i,rows=len(y),profiles=len(set(packet["groups"])),train_rows=info["roles"]["fit"]["rows"],
                            fits=data["fits"],interrupted_fits=data["interrupted_fits"],members=data["members"],
                            decision=data["decision"],correct=scores))
    totals={k:int(np.sum(v)) for k,v in hits.items()}
    diff=compare(np.asarray(hits["frozen"]),np.asarray(hits["before_pilot"]),groups)
    absolute=compare(np.asarray(hits["frozen"]),np.zeros(len(groups),dtype=bool),groups)["group_bootstrap_95"]
    final=dict(lock=sha(root/"score_lock.json"),rows=len(groups),held_profiles=len(set(groups)),profile_overlap=0,
               folds=results,correct=totals,absolute_profile_bootstrap_95=absolute,versus_incumbent=diff,
               fits=sum(r["fits"] for r in results),interrupted_fits=sum(r["interrupted_fits"] for r in results),
               model_api_requests=0,designated_validation_opened=False,
               official_test_opened=False,h2_opened=False)
    save(root/"result.json",final)
    print(totals,len(groups),absolute,diff,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","record-interrupted","seal","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    {"prepare":prepare,"record-interrupted":record_interrupted,"seal":seal,"grade":grade}[args.action](args.out)
