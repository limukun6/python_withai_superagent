"""Fit only alternatives proposed by the live controller's synthesis rule."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.cache_paired_encoders import configure_weights
from scripts.study_paired_encoders import evaluate,order_key
from scripts.unified_released_study import ROOT,env,read,require,save,sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.ab_run import registered_capabilities
from superagent.evidence_strength import propose
from superagent.prediction_evidence import fingerprint


def study(args):
    args.output.mkdir(parents=True,exist_ok=False)
    agent=args.control/"run/agent"
    configure_weights(args.weights)
    settings=env(args.weights,args.output/"physical.jsonl")
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(args.cache)})
    os.environ.update({k:v for k,v in settings.items() if k.startswith("SUPERAGENT_")})
    contract=read(agent/"task_contract.json")
    profile=read(agent/"training_profile.json")
    history=read(agent/"history.json")
    caps=registered_capabilities(contract,read(agent/"run_outcome.json")["training_rows"])
    entries,reason=propose(contract["kind"],caps,history,profile["statistics"]["train_rows"])
    require(bool(entries),"No eligible training feedback")
    parent=reason["template"]
    previous=read(args.reference/"results.json")
    require(previous["candidate_locked"]==[parent] and previous["advance_to_raw_solve"],"No matched reference fits")
    registry=args.output/"candidate_registry.json"
    save(registry,dict(kind=contract["kind"],schema_version=4,entries=entries))
    save(args.output/"proposal.json",dict(reference=reason,new_configurations=entries,
         protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/strength_protocol.json"),
         control_history=sha(agent/"history.json"),registry=sha(registry)))
    def fold(split,names):
        packet=agent/(split+".joblib")
        labels=np.asarray(joblib.load(packet)["y_search"])
        spec=read(agent/(split+"_split.json"))
        require(sha(packet)==spec["payload_sha256"] and fingerprint(labels)==spec["label_sha256"],"Training split changed")
        paths={parent:args.reference/split/parent/"trial.joblib"}
        for name in names:
            if name==parent: continue
            folder=args.output/split/name
            folder.mkdir(parents=True)
            admission=folder/"admission.json"
            ledger=folder/"fits.json"
            destination=folder/"trial.joblib"
            save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)))
            with (folder/"worker.log").open("x") as out:
                subprocess.run([sys.executable,"-m","superagent.experiment_worker",str(packet),name,str(destination),
                                str(registry),"--fit-admission",str(admission)],cwd=ROOT,env=settings,stdout=out,
                               stderr=subprocess.STDOUT,check=True,timeout=360)
            require(read(ledger)["completed_classifier_fits"]==1,"Missing supervised fit")
            paths[name]=destination
        cols={}
        for name,path in paths.items():
            data=joblib.load(path)
            conf=data["metrics"]["resolved_configuration"]
            evidence=data["prediction_record"]
            require(evidence["evaluation_payload_sha256"]==spec["payload_sha256"] and
                    evidence["label_sha256"]==spec["label_sha256"] and conf["id"]==name,"Incompatible predictions")
            if name in entries:
                require(conf["origin"]=="registry" and conf["params"]["C"]==entries[name]["C"],"Wrong generated value")
            cols[name]=np.asarray(data["search_scores"])[:,list(data["score_classes"]).index(1)]
        return cols,labels
    cols,y=fold("primary",entries)
    options=[evaluate(g,cols,y,parent) for n in sorted(entries) for g in ([n],[parent,n])]
    best=min(options,key=order_key)
    choice=best["members"] if best["delta"]>0 else []
    save(args.output/"primary_lock.json",dict(options=options,selected=choice))
    confirmed=[]
    if choice:
        for split in ("confirmation_1","confirmation_2"):
            cols,y=fold(split,choice)
            row=evaluate(choice,cols,y,parent)
            confirmed.append(row)
            if row["delta"]<=0:break
    all_checks=[best,*confirmed]
    gain=sum(r["gained"]-r["lost"] for r in all_checks)/sum(r["rows"] for r in all_checks)
    fits=len((args.output/"physical.jsonl").read_text().splitlines())
    require(0<fits==sum(read(p)["attempted_classifier_fits"] for p in args.output.glob("*/*/fits.json"))<=4,
            "Incorrect physical fit accounting")
    report=dict(parent=parent,generated=entries,primary=options,selected=choice,confirmation=confirmed,pooled_gain=gain,
                passed=len(confirmed)==2 and min(r["delta"] for r in all_checks)>0 and gain>=.003,
                actual_fits=fits,model_requests=0,designated_validation_opened=False)
    save(args.output/"results.json",report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("control","output","reference","weights","cache"):
        p.add_argument("--"+key,type=lambda s:Path(s).resolve(),required=True)
    study(p.parse_args())
