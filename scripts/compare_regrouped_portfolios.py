"""Compare fixed autonomous choices on two fresh allocations of training groups."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.cache_paired_encoders import configure_weights
from scripts.confirm_unused_training_groups import measure
from scripts.study_weighted_evidence import correctness, recipe
from scripts.unified_released_study import ROOT, env, read, require, save, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.data import TrainingTask
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take

PLAN=ROOT/"research/paired_encoder_search_1127/regrouped_protocol.json"


def lock(path, value):
    if path.exists():
        require(read(path) == value, "A frozen study identity changed")
    else:
        save(path, value)


def run(root, weights):
    out=root/"regrouped-runs"
    out.mkdir(exist_ok=True)
    arms=read(PLAN)["runs"]
    recipes={}
    traces={}
    for name, path in arms.items():
        sealed=read(root/path/"sealed.json")
        require(sealed["rows"]==8301 and sealed["independent_reload"] and sealed["api"]==0,"Incomplete source delivery")
        choice=read(root/path/"run/agent/selection.json")["choice"]
        require(sealed["members"]==choice["members"],"Historic members do not match the selected model")
        recipes[name]=recipe(sealed["members"])
        traces[name]=sha(root/path/"sealed.json")
    members=sorted({m for rs in recipes.values() for m in rs["members"]})
    require(len(members)*10<=40,"Unexpected number of component fits")
    agent=root/arms["ordinary"]/"run/agent"
    source=agent/"refit_data.joblib"
    data=joblib.load(source)
    metric=MetricSpec.from_payload(data)
    labels=np.asarray(data["y_train"])
    require(len(labels)==8301 and data["pair_semantics"]=="preference" and metric.name=="accuracy","Training task changed")
    groups=sample_groups(TrainingTask(data["kind"],data["X_train"],labels),"shared_text")
    seeds=[int(hashlib.sha256(f"AIAA3102 paired-development new allocation v1/{i}".encode()).hexdigest()[:8],16) for i in (1,2)]
    existing={fingerprint(np.asarray(read(agent/(name+"_split.json"))["held"])) for name in ("primary","confirmation_1","confirmation_2")}
    existing.update(read(root/"unused-groups"/f"fold_{i}/rows.json")["held_sha256"] for i in (4,5))
    lock(out/"lock.json",dict(protocol_sha256=sha(PLAN),training_sha256=sha(source),members=members,recipes=recipes,
                             original_seals=traces,seeds=seeds,unique_rows=len(labels),components=len(set(groups))))
    configure_weights(weights)
    physical=out/"physical.jsonl"
    settings=env(weights,physical)
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(root/"feature_bank"),
                     "SUPERAGENT_FROZEN_BATCH_REUSE":"1"})
    splits=[]
    for seed in seeds:
        partitions=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(groups,labels,groups))
        require(np.array_equal(np.sort(np.concatenate([held for _,held in partitions])),np.arange(len(labels))),
                "Partition does not cover every training row exactly once")
        for i,(fit,held) in enumerate(partitions,1):
            local=out/f"seed_{seed}"/f"fold_{i}"
            local.mkdir(parents=True,exist_ok=True)
            held_hash=fingerprint(held)
            require(held_hash not in existing and not (set(groups[fit]) & set(groups[held])),"Overlapping or repeated groups")
            existing.add(held_hash)
            payload=local/"training.joblib"
            y=labels[held]
            if not payload.exists():
                joblib.dump(dict(kind=data["kind"], X_train=take(data["X_train"],fit), y_train=labels[fit],
                                 X_search=take(data["X_train"],held), y_search=y, metric_spec=metric.record(),
                                 pair_semantics=data["pair_semantics"], capture_probabilities=True),payload)
            lock(local/"partition.json",dict(train_rows=len(fit), held_rows=len(held), fit_sha256=fingerprint(fit),
                                             held_sha256=held_hash,payload_sha256=sha(payload)))
            bank={}
            for m in members:
                bank[m]=np.asarray(measure(m,payload,agent/"runtime_registry_0.json",local/m,settings,metric,y),dtype=float)
            classes=np.asarray(metric.labels)
            rights={name:correctness(bank,y,classes,r) for name,r in recipes.items()}
            singles={m:correctness(bank,y,classes,recipe([m])) for m in members}
            baseline=rights["ordinary"]
            missing=np.logical_and.reduce([~singles[m] for m in recipes["expanded_budget"]["members"]])
            alternative_names=[m for m in members if m not in recipes["expanded_budget"]["members"]]
            repairs={m:int(sum(missing & singles[m])) for m in alternative_names}
            results={k:dict(correct=int(sum(v)),gained=int(sum(v & ~baseline)),lost=int(sum(~v & baseline))) for k,v in rights.items()}
            row=dict(seed=seed, fold=i, held_rows=len(held), results=results, missed_by_both=int(sum(missing)),
                     third_model_individual_repairs=repairs, any_third_repairs=int(sum(missing & np.logical_or.reduce([singles[m] for m in alternative_names]))))
            lock(local/"result.json",row)
            splits.append(row)
            print(seed,i,len(held),{k:v["correct"] for k,v in results.items()},flush=True)
    summary={}
    for seed in seeds:
        batch=[s for s in splits if s["seed"]==seed]
        summary[str(seed)]={n:{k:sum(row["results"][n][k] for row in batch) for k in ("correct","gained","lost")} for n in arms}
    fits=len(physical.read_text().splitlines())
    require(fits==len(members)*len(splits),"Actual fitted model count differs")
    lock(out/"results.json",dict(protocol_sha256=sha(PLAN), recipes=recipes, splits=splits, by_seed=summary,
                                unique_training_rows=len(labels),actual_classifier_fits=fits, model_api_requests=0,
                                formal_test_opened=False, H2_opened=False, designated_validation_opened=False))
    print("all",summary,"fits",fits,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","weights"):
        p.add_argument("--"+key,required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    run(args.root,args.weights)
