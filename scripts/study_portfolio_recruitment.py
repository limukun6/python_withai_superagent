"""Compare two ordinary controllers and the challenger's own protected delivery."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import packet, source_identity
from scripts.study_portfolio_stability import group_interval
from scripts.study_resource_audition import paired
from scripts.study_resource_menu import inventory
from scripts.unified_released_study import ROOT, env, predictions, read, require, save, sha, text_packet
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import bank_path, vectors
from superagent.metrics import MetricSpec
from superagent.paired_encoder import AVAILABLE_ENCODERS, bank_name, text_key
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take


PLAN=ROOT/"research/paired_encoder_search_1127/recruitment_protocol.json"


def prepare(root,control):
    out=root/"portfolio-recruitment"
    out.mkdir(exist_ok=True)
    library=root/"arctic-library"
    specs=inventory(library)
    activate_resource_directory(library)
    sealed=root/"audition-scale/results.json"
    require(read(sealed)["advance"],"Large-data study did not pass")
    seed=int(hashlib.sha256(b"B protected recruitment ordinary development v1").hexdigest()[:8],16)
    d=joblib.load(root/"analogy-portfolio-raw/portfolio_revised/run/agent/refit_data.joblib")
    wine=load_wine()
    info={}
    for name,kind,metric,rows,y in (("comparison","pair_text","accuracy",d["X_train"],np.asarray(d["y_train"])),
            ("table","numeric_tabular","macro_f1",wine.data,wine.target)):
        folder=out/name
        folder.mkdir(exist_ok=True)
        groups=sample_groups(TrainingTask(kind,rows,y),"shared_text")
        splits=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        fit,held=next(splits.split(groups,y,groups))
        require(not set(groups[fit])&set(groups[held]),"Overlap across outer groups")
        p,q,truth=folder/"p.zip",folder/"q.zip",folder/"truth.joblib"
        if not p.exists():
            if kind=="pair_text":
                text_packet(p,[(*rows[i],int(y[i])) for i in fit],True)
                text_packet(q,take(rows,held),False)
            else:
                packet(p,"tabular",metric,rows[fit],y[fit])
                packet(q,"tabular",metric,rows[held])
            joblib.dump(dict(labels=y[held],groups=groups[held]),truth)
        info[name]=dict(metric=metric,classes=np.unique(y).tolist(),fit_rows=len(fit),held_rows=len(held),
            fit=fingerprint(fit),held=fingerprint(held),fit_groups=len(set(groups[fit])),held_groups=len(set(groups[held])),
            p=sha(p),q=sha(q),truth=sha(truth))
        if kind!="pair_text":
            continue
        texts=sorted({t for i in fit for t in rows[i]})
        allowed={text_key(t) for t in texts}
        require(not set(texts)&{t for i in held for t in rows[i]},"Shared fitting text in outer holdout")
        cache=folder/"fit_features"
        cache.mkdir(exist_ok=True)
        names={bank_name(v) for v in AVAILABLE_ENCODERS}|{bank_path(cache,s).name for s in specs.values()}
        for n in sorted(names):
            dest=cache/n
            base=root/"feature_bank"/n
            if not dest.exists() and base.is_file():
                with np.load(base,allow_pickle=False) as source:
                    keys=source["keys"]
                    use=np.isin(keys,list(allowed))
                    require(int(use.sum())==len(allowed),"Previously installed vector bank incomplete")
                    np.savez(dest,keys=keys[use],vectors=source["vectors"][use])
        os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(root/"arctic-menu/fit_features")
        for spec in specs.values():
            dest=bank_path(cache,spec)
            if not dest.exists():
                np.savez(dest,keys=[text_key(t) for t in texts],vectors=vectors(spec,texts))
        for f in cache.glob("*.npz"):
            with np.load(f,allow_pickle=False) as z:
                require(set(z["keys"].tolist())==allowed,"Cache contains outer texts")
        info[name]["fit_vector_texts"]=len(allowed)
    lock(out/"inputs.json",dict(plan=sha(PLAN),trigger=sha(sealed),seed=seed,cases=info,
        control=source_identity(control),challenger=source_identity(ROOT),resources=sorted(specs),fit_cap=32,seconds=1800))
    print({n:(v["fit_rows"],v["held_rows"]) for n,v in info.items()},flush=True)


def solve(root,control,case,arm):
    out=root/"portfolio-recruitment"
    plan=read(out/"inputs.json")
    impl=control if arm=="control" else ROOT
    folder=out/case
    info=plan["cases"][case]
    require(source_identity(impl)==plan[arm] and sha(folder/"p.zip")==info["p"] and sha(folder/"q.zip")==info["q"],
            "Changed source or inputs")
    base=folder/arm
    base.mkdir(exist_ok=False)
    library=root/"arctic-library"
    require(sorted(inventory(library))==plan["resources"],"Changed resource inventory")
    physical=base/"physical.jsonl"
    settings=env(library,physical)
    for k in list(settings):
        if k.startswith("SUPERAGENT_"):
            del settings[k]
    # Load only runtime/guard code from the arm under test.
    settings["PYTHONPATH"]=os.pathsep.join(map(str,(impl/"scripts/ab_acceptance_guard",impl)))
    if case=="comparison":
        settings["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(folder/"fit_features")
    lock(base/"launch.json",dict(inputs=sha(out/"inputs.json"),case=case,source=plan[arm],
                                cap=plan["fit_cap"],seconds=plan["seconds"],model_hint=None,planner_hint=None))
    for mode,cmd in (("solve",["solve",folder/"p.zip","--fit-limit",32,"--time-limit",1800,"--prediction-input",folder/"q.zip"]),
                     ("reload",["predict",folder/"q.zip"])):
        if mode=="reload":
            settings.pop("SUPERAGENT_PAIRED_FEATURE_CACHE",None)
        with (base/(mode+".log")).open("x") as log:
            subprocess.run([sys.executable,"-m","unified_agent",*map(str,cmd),"--run-dir",str(base/"run"),"--weights-dir",
                str(library),"--output",str(base/(mode+".csv"))],cwd=impl,env=settings,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2000)
    agent=base/"run/agent"
    outcome,final=read(agent/"run_outcome.json"),read(agent/"final_selection.json")
    count=len(physical.read_text().splitlines())
    require(outcome["status"]=="completed" and final["refit"] and final["training_rows"]==info["fit_rows"]
            and sha(agent/final["model_file"])==final["model_sha256"] and sha(base/"reload.csv")==sha(base/"solve.csv")
            and outcome["new_api_requests"]==0 and count==outcome["attempted_classifier_fits"]<=32,
            "Unverified source, budget or delivery")
    predictions(base/"reload.csv",info["held_rows"])
    protected=None
    if arm=="challenger":
        review=read(agent/"portfolio_review.json")
        inc=read(agent/"portfolio_incumbent.json") if (agent/"portfolio_incumbent.json").exists() else final
        require(sha(agent/inc["model_file"])==inc["model_sha256"]==review["incumbent_model"],"Incumbent identity changed")
        if inc["model_sha256"]==final["model_sha256"]:
            shutil.copyfile(base/"reload.csv",base/"protected.csv")
        else:
            model=base/"protected_snapshot"
            model.mkdir()
            shutil.copyfile(agent/"task_spec.json",model/"task_spec.json")
            (model/inc["model_file"]).symlink_to(agent/inc["model_file"])
            save(model/"final_selection.json",inc)
            activate_resource_directory(library,settings)
            with (base/"protected.log").open("x") as log:
                subprocess.run([sys.executable,"-m","superagent","predict",str(folder/"q.zip"),"--run-dir",str(model),
                    "--output",str(base/"protected.csv")],cwd=impl,env=settings,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        predictions(base/"protected.csv",info["held_rows"])
        protected=dict(model=inc["model_sha256"],sha=sha(base/"protected.csv"),review=sha(agent/"portfolio_review.json"),
                       status=review["status"])
    lock(base/"sealed.json",dict(launch=sha(base/"launch.json"),model=final["model_sha256"],predictions=sha(base/"solve.csv"),
        members=outcome["members"],independent_reload=True,supervised_fits=count,
        completed=outcome["completed_classifier_fits"],unknown=outcome["unknown_failed_attempts"],api=0,protected=protected))
    print(case,arm,count,outcome["members"],protected,flush=True)


def score(root,case):
    out=root/"portfolio-recruitment"
    folder=out/case
    info=read(out/"inputs.json")["cases"][case]
    arms=["control","challenger"] if case=="comparison" else ["challenger"]
    seals={a:read(folder/a/"sealed.json") for a in arms}
    paths={a:(folder/a/"solve.csv",s["predictions"]) for a,s in seals.items()}
    paths["protected"]=(folder/"challenger/protected.csv",seals["challenger"]["protected"]["sha"])
    outputs={}
    for arm,(p,digest) in paths.items():
        require(sha(p)==digest,"Predictions no longer match their seal")
        outputs[arm]=predictions(p,info["held_rows"])
    truth=folder/"truth.joblib"
    require(sha(truth)==info["truth"],"Different held training labels")
    lock(folder/"score_lock.json",dict(seals={a:sha(folder/a/"sealed.json") for a in arms},truth=info["truth"]))
    data=joblib.load(truth)
    y,groups=data["labels"],data["groups"]
    metric=MetricSpec(info["metric"],tuple(info["classes"]))
    scores={a:metric.evaluate(y,p) for a,p in outputs.items()}
    correct={a:v==y for a,v in outputs.items()}
    deltas={a:dict(**paired(correct["challenger"],v),cluster_accuracy_interval=group_interval(correct["challenger"],v,groups,538902))
            for a,v in correct.items() if a!="challenger"}
    lock(folder/"results.json",dict(lock=sha(folder/"score_lock.json"),task_metric=metric.record(),
        scored=scores,paired_accuracy=deltas,elections=seals,received_score_after_sealing=True,
        course_designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(scores,deltas,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","control","challenger","score"))
    p.add_argument("--case",choices=("comparison","table"),default="comparison")
    p.add_argument("--root",required=True,type=lambda x:Path(x).resolve())
    p.add_argument("--control",required=True,type=lambda x:Path(x).resolve())
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.root,args.control)
    elif args.action=="score":
        score(args.root,args.case)
    else:
        solve(args.root,args.control,args.case,args.action)
