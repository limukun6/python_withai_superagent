"""Compare a feedback-nominated sentence encoder with or without a contextual word-predictability feature."""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np

from scripts.cache_paired_encoders import configure_weights
from scripts.confirm_unused_training_groups import measure
from scripts.study_weighted_evidence import FOLDS,available,correctness,matrices,recipe
from scripts.unified_released_study import ROOT,env,read,require,save,sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.masked_context import CACHE_NAME, ENV, SPEC, materialize
from superagent.metrics import MetricSpec
from superagent.paired_encoder import NAMES

PLAN=ROOT/"research/paired_encoder_search_1127/masked_context_protocol.json"


def lock(path,value):
    if path.exists():
        require(read(path)==value,"Prior selection or observations differ")
    else:
        save(path,value)


def run(root,weights):
    agent=root/"resource-raw/resource_default/run/agent"
    out=root/"masked-context"
    out.mkdir(exist_ok=True)
    cache=root/"feature_bank"
    require((cache/CACHE_NAME).exists(),"Missing verified training-only masked features")
    configure_weights(weights)
    os.environ[ENV]=str(weights/SPEC["model"].split("/")[-1])
    supports=[r for r in read(agent/"history.json") if r["status"]=="completed" and r["split_id"]=="primary"
              and NAMES.get(r["candidate"],(None,None))[1]=="linear"]
    require(bool(supports),"No measured compatible parent")
    parent=min(supports,key=lambda r:(-r["search_score"],r["trial"]))["candidate"]
    spec=dict(kind="pair_text",operator="add_masked_context",template=parent)
    name=materialize(spec).family
    baseline=read(root/"scale-screen/results.json")["candidate_locked"]
    reference=recipe(baseline)
    reg=read(agent/"runtime_registry_0.json")
    registry=out/"registry.json"
    lock(registry,dict(reg,entries={**reg["entries"],name:spec}))
    lock(out/"proposal_lock.json",dict(protocol=sha(PLAN),reference=reference,parent=parent,proposed=name,registry=sha(registry)))
    physical=out/"physical.jsonl"
    settings=env(weights,physical)
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(cache)})
    inherit=available(root)
    splits=[set(read(agent/(fold+"_split.json"))["held"]) for fold in FOLDS]
    require(all(not (s & t) for i,s in enumerate(splits) for t in splits[i+1:]),"Held groups overlap")
    def compare(fold,options):
        bank,y,classes,_=matrices(agent,fold,baseline,inherit[fold])
        path=agent/(fold+".joblib")
        metric=MetricSpec.from_payload(joblib.load(path))
        bank[name]=np.asarray(measure(name,path,registry,out/fold/name,settings,metric,y),dtype=float)
        before=correctness(bank,y,classes,reference)
        rows=[]
        for candidate in options:
            after=correctness(bank,y,classes,candidate)
            rows.append(dict(fold=fold,rule=candidate,rows=len(y),reference=int(sum(before)),correct=int(sum(after)),
                             net=int(after.sum()-before.sum()),gained=int(sum(after&~before)),lost=int(sum(~after&before))))
        return rows
    primary=compare(FOLDS[0],[recipe([name]),recipe([*baseline,name],[.25,.25,.5])])
    eligible=[row for row in primary if row["net"]>0]
    first=min(eligible,key=lambda r:(-r["net"],len(r["rule"]["members"]))) if eligible else None
    lock(out/"primary_lock.json",dict(measured=primary,selected=first))
    second=compare(FOLDS[1],[first["rule"]])[0] if first else None
    eligible=bool(second and second["net"]>0 and (first["net"]+second["net"])/(first["rows"]+second["rows"])>=.003)
    lock(out/"confirmation_lock.json",dict(second=second,eligible=eligible))
    third=compare(FOLDS[2],[first["rule"]])[0] if eligible else None
    fits=len(physical.read_text().splitlines())
    require(fits==1+bool(second)+bool(third)<=3,"Physical supervised fits differ")
    result=dict(protocol_sha256=sha(PLAN),nomination=spec,primary=primary,first_confirmation=second,
                second_confirmation=third,advance_to_cold=bool(third and third["net"]>0),
                actual_fits=fits, model_api_requests=0, released_validation_opened=False, formal_test_opened=False,H2_opened=False)
    lock(out/"result.json",result)
    print([(r["net"],r["correct"]) for r in primary],second,third,"fits",fits,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","weights"):
        p.add_argument("--"+key,required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    run(args.root,args.weights)
