"""Compare complete unhinted runs that differ only by one published local encoder."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env, read, require, sha, text_packet
from superagent.data import TrainingTask
from superagent.local_sentence import bank_path, describe, discover, encode
from superagent.paired_encoder import AVAILABLE_ENCODERS, bank_name, text_key
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN = ROOT/"research/paired_encoder_search_1127/arctic_menu_protocol.json"
ARMS = ("original","expanded")


def inventory(folder):
    mapping = {}
    discover(folder, mapping)
    specs=[describe(Path(path)) for path in mapping.values()]
    return {spec["signature"]:spec for spec in specs}


def prepare(root):
    out = root/"arctic-menu"
    out.mkdir(exist_ok=True)
    src = root/"analogy-portfolio-raw/portfolio_revised/run/agent/refit_data.joblib"
    payload = joblib.load(src)
    y = np.asarray(payload["y_train"])
    x = payload["X_train"]
    g = sample_groups(TrainingTask(payload["kind"], x, y),"shared_text")
    seed = int(hashlib.sha256(b"AIAA3102 resource menu blind development v1").hexdigest()[:8],16)
    splitter = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    fit, held = next(splitter.split(g,y,g))
    require(len(y)==8301 and not set(g[fit])&set(g[held]),"Invalid development split")
    fit_texts = sorted({s for i in fit for s in x[i]})
    held_texts = {s for i in held for s in x[i]}
    require(not set(fit_texts)&held_texts,"Shared title crosses the evaluation boundary")
    packet, features = out/"anonymous.zip", out/"features.zip"
    if not packet.exists():
        text_packet(packet,[(*x[i],int(y[i])) for i in fit],True)
    if not features.exists():
        text_packet(features,[x[i] for i in held],False)
    truth = out/"outer_labels.joblib"
    if not truth.exists():
        joblib.dump(dict(labels=y[held],groups=g[held]),truth)
    old = inventory(root/"discovery-library")
    new = inventory(root/"arctic-library")
    added = set(new)-set(old)
    require(len(added)==1 and set(old)<=set(new),"Expected exactly one new eligible encoder")
    signature = next(iter(added))
    publication = read(PLAN)["resource"]
    path = root/"arctic-library"/publication["repo"].split("/")[-1]
    expected = new[signature]
    upstream = read(root/"arctic-installed.json")[publication["repo"]]
    require(describe(path)==expected and upstream["revision"]==publication["revision"]
            and upstream["files"]==expected["files"],"New resource lacks matching published bytes and revision")
    lock(out/"inputs.json",dict(protocol=sha(PLAN),runtime=source_identity(ROOT),seed=seed,source=sha(src),
         fit_key=fingerprint(fit),held_key=fingerprint(held),fit_rows=len(fit),held_rows=len(held),
         fit_groups=len(set(g[fit])),held_groups=len(set(g[held])),fit_texts=len(fit_texts),shared_groups=0,
         task=sha(packet),features=sha(features),outer_labels=sha(truth),signature=signature,
         original_encoders=sorted(old),expanded_encoders=sorted(new),publisher_receipt=sha(root/"arctic-installed.json")))
    cache=out/"fit_features"
    cache.mkdir(exist_ok=True)
    key_set={text_key(s) for s in fit_texts}
    basic={bank_name(v) for v in AVAILABLE_ENCODERS}|{bank_path(root,s).name for s in old.values()}
    for name in sorted(basic):
        source=root/"feature_bank"/name
        target=cache/name
        if source.is_file() and not target.exists():
            with np.load(source,allow_pickle=False) as bank:
                keys=bank["keys"]
                use=np.isin(keys,list(key_set))
                require(use.sum()==len(key_set),"Existing label-free vectors omit training texts")
                np.savez(target,keys=keys[use],vectors=bank["vectors"][use])
    os.environ[expected["environment_variable"]]=str(path)
    target=bank_path(cache,expected)
    if not target.exists():
        start=time.monotonic()
        vectors=encode(expected,fit_texts)
        np.savez(target,keys=[text_key(s) for s in fit_texts],vectors=vectors)
        lock(out/"new_features.json",dict(encoder=signature,texts=len(fit_texts),seconds=time.monotonic()-start,
             shape=list(vectors.shape),cache=sha(target),new_classifier_fits=0,model_api_requests=0))
    for p in cache.glob("*.npz"):
        with np.load(p,allow_pickle=False) as cached:
            require(set(cached["keys"].tolist())==key_set,"Evaluation texts entered a fitting cache")
    require(sha(target)==read(out/"new_features.json")["cache"],"Changed added representation")
    print(len(fit),len(held),signature,flush=True)


def solve(root, arm):
    out=root/"arctic-menu"
    data=read(out/"inputs.json")
    weights=root/("arctic-library" if arm=="expanded" else "discovery-library")
    ids=sorted(inventory(weights))
    require(ids==data[arm+"_encoders"] and source_identity(ROOT)==data["runtime"],"Changed menu or runtime")
    require(sha(out/"anonymous.zip")==data["task"] and sha(out/"features.zip")==data["features"],"Input changed")
    run=out/arm
    run.mkdir(exist_ok=False)
    ledger=run/"physical.jsonl"
    settings=env(weights,ledger)
    for key in list(settings):
        if key.startswith("SUPERAGENT_"):
            del settings[key]
    settings.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(out/"fit_features"),SUPERAGENT_FROZEN_BATCH_REUSE="1")
    lock(run/"launch.json",dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),arm=arm,runtime=data["runtime"],
         available_signatures=ids,fit_cap=22,wall_seconds=1800,model_hint=None,policy_hint=None))
    for action,arguments in (("solve",["solve",out/"anonymous.zip","--fit-limit",22,"--time-limit",1800,"--prediction-input",out/"features.zip"]),
                             ("reload",["predict",out/"features.zip"])):
        if action=="reload":
            settings.pop("SUPERAGENT_PAIRED_FEATURE_CACHE",None)
            settings.pop("SUPERAGENT_FROZEN_BATCH_REUSE",None)
        with (run/(action+".log")).open("x") as stream:
            subprocess.run([sys.executable,"-m","unified_agent",*map(str,arguments),"--run-dir",str(run/"run"),
                "--weights-dir",str(weights),"--output",str(run/(action+".csv"))],cwd=ROOT,env=settings,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=1990,check=True)
    agent=run/"run/agent"
    result=read(agent/"run_outcome.json")
    final=read(agent/"final_selection.json")
    fit_count=len(ledger.read_text().splitlines())
    require(result["status"]=="completed" and fit_count==result["attempted_classifier_fits"]==result["completed_classifier_fits"]<=22
            and result["new_api_requests"]==0 and final["training_rows"]==data["fit_rows"] and final["refit"]
            and final["model_sha256"]==sha(agent/final["model_file"]) and sha(run/"solve.csv")==sha(run/"reload.csv"),
            "Incomplete autonomous run")
    signature=data["signature"]
    history=read(agent/"history.json")
    first_menu=read(agent/"decisions.json")[0]["alternatives"]
    candidates={name for choice in first_menu for name in choice["members"]}
    voted=read(agent/"analogy_review.json") if (agent/"analogy_review.json").exists() else {}
    trials=[dict(split=r["split_id"],outcome=r["status"],member=r["candidate"]) for r in history
            if r.get("resolved_configuration",{}).get("params",{}).get("revision")==signature]
    tag="local_sentence_"+signature[:20]
    require((tag in candidates)==(arm=="expanded"),"Installed option did not reach the actual menu")
    outcome=dict(launch=sha(run/"launch.json"),fits=fit_count,model_api_requests=0,rows=data["fit_rows"],
         members=result["members"],model=final["model_sha256"],predictions=sha(run/"solve.csv"),independent_reload=True,
         added_available=tag in candidates,added_trials=trials,added_deployed=(tag in result["members"] or
         voted.get("status")=="accepted" and voted.get("derived_configuration",{}).get("revision")==signature),
         first_actions=len(first_menu),adviser=voted.get("status"))
    lock(run/"sealed.json",outcome)
    print(arm,outcome,flush=True)


def score(root):
    out=root/"arctic-menu"
    facts=read(out/"inputs.json")
    seals={a:read(out/a/"sealed.json") for a in ARMS}
    predictions={}
    for a in ARMS:
        p=out/a/"solve.csv"
        require(sha(p)==seals[a]["predictions"],"Prediction seal mismatch")
        with p.open(newline="") as stream:
            data=list(csv.DictReader(stream))
        require([int(x["row_id"]) for x in data]==list(range(facts["held_rows"])),"Invalid row ordering")
        predictions[a]=np.asarray([int(x["prediction"]) for x in data])
    truth=out/"outer_labels.joblib"
    lock(out/"score_lock.json",dict(protocol=sha(PLAN),seals={a:sha(out/a/"sealed.json") for a in ARMS},
                                    outer=facts["outer_labels"]))
    require(sha(truth)==facts["outer_labels"],"Changed evaluation data")
    data=joblib.load(truth)
    rights={a:y==data["labels"] for a,y in predictions.items()}
    before,after=[rights[a] for a in ARMS]
    _,idx=np.unique(data["groups"],return_inverse=True)
    count=np.bincount(idx)
    deltas=np.bincount(idx,weights=after.astype(float)-before)
    rng=np.random.default_rng(facts["seed"])
    bootstrap=[]
    for _ in range(40):
        draw=rng.integers(len(count),size=(100,len(count)))
        bootstrap.extend(deltas[draw].sum(axis=1)/count[draw].sum(axis=1))
    result=dict(lock=sha(out/"score_lock.json"),held_rows=len(before),fitting_rows=facts["fit_rows"],
         correct={a:int(v.sum()) for a,v in rights.items()},fixed=int((after&~before).sum()),lost=int((before&~after).sum()),
         cluster_interval=np.quantile(bootstrap,[.025,.975]).tolist(),arms=seals,actual_fits=sum(v["fits"] for v in seals.values()),
         model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result["correct"],result["fixed"],result["lost"],result["cluster_interval"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","original","expanded","score"))
    parser.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    if args.action=="prepare":
        prepare(args.root)
    elif args.action=="score":
        score(args.root)
    else:
        solve(args.root,args.action)
