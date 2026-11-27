"""Compare previously frozen recipes on the two unused group folds with shared component fits."""
import argparse
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.cache_paired_encoders import configure_weights
from scripts.study_weighted_evidence import correctness, recipe
from scripts.unified_released_study import ROOT,env,read,require,save,sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups,take


def nominations(root):
    weighted=read(root/"weighted-screen/results.json")
    require(weighted["advance_to_fresh_groups"],"The weighted experiment did not pass its registered gate")
    old=read(root/"raw/original_compute_matched/run/agent/selection.json")["choice"]
    small=read(root/"raw/default/run/agent/selection.json")["choice"]
    scaled=read(root/"scale-screen/results.json")
    require(scaled["advance_to_raw_solve"],"The training scale screen did not pass")
    return dict(single=recipe(small["members"]),legacy_vote=recipe(old["members"]),
                scale_vote=recipe(scaled["candidate_locked"]),**{k:weighted["selection"][k] for k in ("simple","weighted","own_uniform")})


def folds(training, agent):
    task=TrainingTask(training["kind"],training["X_train"],training["y_train"])
    groups=sample_groups(task,"shared_text")
    records=[read(agent/(f+"_split.json")) for f in ("primary","confirmation_1","confirmation_2")]
    count=records[0]["resource"]["fold_count"]
    require(count==5,"The archive should contain three measured and two unused folds")
    seed=read(agent/"run_manifest.json")["seed"]
    partitions=list(StratifiedGroupKFold(count,shuffle=True,random_state=seed).split(groups,task.y_train,groups))
    for (fit,held),r in zip(partitions,records):
        require(np.array_equal(fit,r["fit"]) and np.array_equal(held,r["held"]),"Partition reconstruction mismatch")
    require(len(set(np.concatenate([held for _,held in partitions])))==len(groups),"Overlapping held rows")
    require(all(not (set(groups[fit])&set(groups[held])) for fit,held in partitions),"Connected group crosses partition")
    return partitions[3:]


def measure(name,payload,registry,folder,environment,metric,labels):
    folder.mkdir(parents=True,exist_ok=True)
    trial=folder/"trial.joblib"
    receipt=folder/"receipt.json"
    ledger=folder/"fits.json"
    if not receipt.exists():
        require(not ledger.exists() and not trial.exists(),"Existing attempt needs recovery")
        admission=folder/"admission.json"
        save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)))
        with (folder/"worker.log").open("x") as log:
            subprocess.run([sys.executable,"-m","superagent.experiment_worker",str(payload),name,str(trial),str(registry),
                            "--fit-admission",str(admission)],cwd=ROOT,env=environment,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
        counts=read(ledger)
        require(counts["attempted_classifier_fits"]==counts["completed_classifier_fits"]==1,"Component did not complete exactly one fit")
        save(receipt,dict(member=name,payload_sha256=sha(payload),trial_sha256=sha(trial),fits=1))
    require(read(receipt)==dict(member=name,payload_sha256=sha(payload),trial_sha256=sha(trial),fits=1),
            "Cannot reuse a changed trial or payload")
    item=joblib.load(trial)
    require(item["metrics"]["resolved_configuration"]["id"]==name,"Incorrect fitted method")
    obs=planner_observation(item,item["metrics"],dict(sha256=sha(payload),labels=labels),metric)
    require(obs["probabilities"] is not None,"Missing aligned probability matrix")
    return obs["probabilities"]


def run(root,weights):
    out=root/"unused-groups"
    out.mkdir(exist_ok=True)
    agent=root/"raw/race/run/agent"
    training=joblib.load(agent/"refit_data.joblib")
    metric=MetricSpec.from_payload(training)
    recipes=nominations(root)
    members=sorted({n for r in recipes.values() for n in r["members"]})
    parts=folds(training,agent)
    require(len(members)*len(parts)<=12,"Too many component fits")
    physical=out/"physical.jsonl"
    configure_weights(weights)
    settings=env(weights,physical)
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(root/"feature_bank"),
                     "SUPERAGENT_FROZEN_BATCH_REUSE":"1"})
    lock=dict(protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/unused_groups_protocol.json"),recipes=recipes,
              members=members,training_file=sha(agent/"refit_data.joblib"),
              held_sha256=[fingerprint(held) for _,held in parts],planned_fits=len(members)*len(parts))
    save(out/"selection_lock.json",lock)
    measurements=[]
    for i,(fit,held) in enumerate(parts,4):
        folder=out/f"fold_{i}"
        folder.mkdir(exist_ok=True)
        payload=folder/"training.joblib"
        labels=training["y_train"][held]
        if not payload.exists():
            joblib.dump(dict(kind=training["kind"],X_train=take(training["X_train"],fit),y_train=training["y_train"][fit],
                             X_search=take(training["X_train"],held),y_search=labels,metric_spec=training["metric_spec"],
                             pair_semantics=training["pair_semantics"],capture_probabilities=True),payload)
        save(folder/"rows.json",dict(held_sha256=fingerprint(held),fit_sha256=fingerprint(fit),payload_sha256=sha(payload),
                                      held_rows=len(held),fit_rows=len(fit)))
        bank={}
        for name in members:
            bank[name]=measure(name,payload,agent/"runtime_registry_0.json",folder/name,settings,metric,labels)
        correctnesses={tag:correctness(bank,labels,np.asarray(metric.labels),r) for tag,r in recipes.items()}
        reference=correctnesses["single"]
        counts={tag:dict(correct=int(sum(right)),gained=int(sum(right&~reference)),lost=int(sum(~right&reference)))
                for tag,right in correctnesses.items()}
        measured=dict(fold=i,rows=len(held),scores=counts)
        save(folder/"result.json",measured)
        measurements.append(measured)
        print(measured,flush=True)
    actual=len(physical.read_text().splitlines())
    require(actual==lock["planned_fits"],"Unexpected independently observed fit total")
    result=dict(selections=recipes,folds=measurements,actual_classifier_fits=actual,new_weight_fits=0,
                new_model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    save(out/"results.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","weights"):
        parser.add_argument("--"+name,type=lambda s:Path(s).resolve(),required=True)
    args=parser.parse_args()
    run(args.root,args.weights)
