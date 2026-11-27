"""Learn block scales on fitting groups and compare the resulting classifiers on unseen training groups."""
import argparse
from pathlib import Path
import json

import joblib
import numpy as np

from scripts.cache_paired_encoders import configure_weights
from scripts.confirm_unused_training_groups import measure
from scripts.study_weighted_evidence import FOLDS,available,correctness,matrices,recipe
from scripts.unified_released_study import ROOT,env,read,require,save,sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.metrics import MetricSpec
from superagent.paired_encoder import NAMES
from superagent.view_weighting import materialize

PLAN=ROOT/"research/paired_encoder_search_1127/feature_weighting_protocol.json"


def freeze(path,value):
    if path.is_file():
        require(read(path)==value,"Frozen inputs or measurements changed")
    else:
        save(path,value)


def study(root,weights):
    out=root/"feature-weighting"
    out.mkdir(exist_ok=True)
    agent=root/"anonymous-choice/anonymous_default/run/agent"
    measured=read(agent/"history.json")
    candidates=[r for r in measured if r["status"]=="completed" and r["split_id"]=="primary"
                and NAMES.get(r["candidate"],(None,None))[1]=="linear"]
    parent=min(candidates,key=lambda r:(-r["search_score"],r["trial"]))["candidate"]
    configure_weights(weights)
    entries={}
    for mode in ("energy","aligned"):
        spec=dict(kind="pair_text",operator="weight_feature_blocks",mode=mode,template=parent)
        entries[materialize(spec).family]=spec
    old=read(agent/"runtime_registry_0.json")
    registry=out/"registry.json"
    freeze(registry,dict(old,entries={**old["entries"],**entries}))
    control_names=read(root/"scale-screen/results.json")["candidate_locked"]
    control=recipe(control_names)
    subsets=[set(read(agent/(fold+"_split.json"))["held"]) for fold in FOLDS]
    require(all(a.isdisjoint(b) for i,a in enumerate(subsets) for b in subsets[i+1:]),"Evaluation groups overlap")
    freeze(out/"proposal_lock.json",dict(protocol=sha(PLAN),registry=sha(registry),selected_parent=parent,reference=control))
    path_counts=out/"physical.jsonl"
    extra_counts=out/"supervised_weights.jsonl"
    settings=env(weights,path_counts)
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(root/"feature_bank"),
                     "SUPERAGENT_BLOCK_WEIGHT_LOG":str(extra_counts)})
    inherited=available(root)
    def evaluate(fold,options):
        if not options:
            return []
        payload=agent/(fold+".joblib")
        bank,y,classes,_=matrices(agent,fold,control_names,inherited[fold])
        metric=MetricSpec.from_payload(joblib.load(payload))
        before=correctness(bank,y,classes,control)
        details={}
        for name in sorted({n for n,rule in options}):
            store=out/fold/name
            bank[name]=np.asarray(measure(name,payload,registry,store,settings,metric,y),dtype=float)
            features=joblib.load(store/"trial.joblib")["model"].named_steps["features"]
            details[name]=dict(scales=features.scales_.tolist(),widths=features.widths_)
        rows=[]
        for name,choice in options:
            pred=correctness(bank,y,classes,choice)
            rows.append(dict(name=name,mode=entries[name]["mode"],rule=choice,fold=fold,rows=len(y),reference=int(sum(before)),
                             correct=int(sum(pred)),gained=int(sum(pred&~before)),lost=int(sum(~pred&before)),
                             net=int(pred.sum()-before.sum()),feature_blocks=details[name]))
        return rows
    pool=[(name,choice) for name in entries for choice in (recipe([name]),recipe([*control_names,name],[.25,.25,.5]))]
    primary=evaluate("primary",pool)
    locked=[]
    for name in entries:
        choices=[r for r in primary if r["name"]==name and r["net"]>0]
        if choices:
            locked.append(min(choices,key=lambda r:(-r["net"],len(r["rule"]["members"]))))
    freeze(out/"primary_lock.json",dict(primary=primary,chosen=locked))
    first=evaluate("confirmation_1",[(r["name"],r["rule"]) for r in locked])
    passed=[]
    for a,b in zip(locked,first):
        pooled=(a["net"]+b["net"])/(a["rows"]+b["rows"])
        if b["net"]>0 and pooled>=.003:
            passed.append((min(a["net"]/a["rows"],b["net"]/b["rows"]),pooled,b))
    passed.sort(key=lambda r:(-r[0],-r[1],len(r[2]["rule"]["members"]),r[2]["name"]))
    chosen=passed[0][2] if passed else None
    freeze(out/"confirmation_lock.json",dict(first=first,selected=chosen))
    second=evaluate("confirmation_2",[(chosen["name"],chosen["rule"])]) if chosen else []
    fits=len(path_counts.read_text().splitlines())
    require(fits==len(entries)+len(locked)+len(second)<=5,"Unexpected supervised classifier count")
    stages=[json.loads(line) for line in extra_counts.read_text().splitlines()]
    n_learning=sum(v["event"]=="start" for v in stages)
    expected=1+sum(r["mode"]=="aligned" for r in first+second)
    require(n_learning==expected==len(stages)-n_learning<=3,"Unsettled supervised scale stage")
    result=dict(protocol_sha256=sha(PLAN),selected_parent=parent,reference=control,primary=primary,confirmation=first,third=second,
                advance=bool(second and second[0]["net"]>0),new_classifiers=fits,additional_supervised_weight_fits=n_learning,
                model_api_requests=0,released_validation_opened=False,formal_test_opened=False,H2_opened=False)
    freeze(out/"results.json",result)
    print([(r["mode"],len(r["rule"]["members"]),r["net"]) for r in primary],[(r["mode"],r["net"]) for r in first],
          second, "fits",fits,n_learning,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","weights"):
        parser.add_argument("--"+key,required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    study(args.root,args.weights)
