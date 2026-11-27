"""Replay an added encoder-geometry signal on historical, aligned one-step choices."""
import argparse
from contextlib import nullcontext
import hashlib
from itertools import combinations
import os
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, read, require, sha
from superagent import experiment_policy as policy
from superagent.ab_run import registered_capabilities
from superagent.autonomous_run import installed_methods, planner_observation
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.preference_analogy import embedding_function, views
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/representation_geometry_protocol.json"
ARMS={"small":"last-token-subset/original/run/agent","large":"discovered-raw/discovered/run/agent"}


def probe_rows(rows):
    groups=sample_groups(TrainingTask("pair_text",rows,np.zeros(len(rows))),"shared_text")
    rank={}
    for i,(g,pair) in enumerate(zip(groups,rows)):
        key=policy.identity(["B geometry order v1",*sorted(pair)])
        if g not in rank or key<rank[g][0]:
            rank[g]=(key,i)
    selected=[i for _,i in sorted(rank.values())[:96]]
    return [rows[i] for i in selected]


def matrix(rows,feature):
    change,_=views(rows,embedding_function(feature))
    centered=np.asarray(change,dtype=np.float64)-change.mean(axis=0,dtype=np.float64)
    gram=centered @ centered.T
    norm=np.linalg.norm(gram)
    require(norm>1e-10 and np.isfinite(norm),"Degenerate representation")
    return gram/norm


def slice_record(record,mask):
    selected={k:np.asarray(record[k])[mask] for k in ("labels","predictions","probabilities")}
    return dict(**selected,row_key=fingerprint(np.flatnonzero(mask)),
                score=float((selected["labels"]==selected["predictions"]).mean()))


def elect(members,learn,held,keys):
    available=[(m,) for m in members]+list(combinations(members,2))
    def predictions(records,parts):
        if len(parts)==1:
            return records[parts[0]]["predictions"]
        return np.mean([records[m]["probabilities"] for m in parts],axis=0).argmax(axis=1)
    truth=learn[members[0]]["labels"]
    selected=min(available,key=lambda group:(-int((predictions(learn,group)==truth).sum()),len(group),
                                               tuple(sorted(keys[m] for m in group))))
    return list(selected),int((predictions(held,selected)==held[members[0]]["labels"]).sum())


def run(root):
    out=root/"representation-geometry"
    out.mkdir(exist_ok=False)
    activate_resource_directory(root/"discovery-library")
    os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(root/"feature_bank")
    archive=read(root/"training-fidelity/inputs.json")
    paths={k:root/v for k,v in ARMS.items()}
    payloads={k:joblib.load(p/"primary.joblib") for k,p in paths.items()}
    inputs=probe_rows(payloads["small"]["X_train"])
    by_rep={}
    for key,entry in sorted(archive["candidates"].items()):
        small=entry["small"]
        source=paths["small"]/f"trial_{small['trial']}.joblib"
        require(sha(source)==small["sha256"],"Different fitted source")
        trial=joblib.load(source)
        params=trial["metrics"]["resolved_configuration"]["params"]
        encoder=params.get("encoder")
        feature=getattr(trial["model"],"named_steps",{}).get("features")
        if encoder and feature is not None and encoder not in by_rep:
            by_rep[encoder]=(key,entry,feature)
    require(len(by_rep)>=5,"Too few distinct eligible representations")
    grams={encoder:matrix(inputs,value[2]) for encoder,value in by_rep.items()}
    align={(a,b):float(np.clip(np.sum(grams[a]*grams[b]),0.,1.)) for a in grams for b in grams}
    lock(out/"inputs.json",dict(protocol=sha(PLAN),archive=sha(root/"training-fidelity/inputs.json"),
         data={s:sha(p/"primary.joblib") for s,p in paths.items()},probe=policy.identity(inputs),representations={
         e:dict(config=v[0],trials=v[1]) for e,v in by_rep.items()},alignment={a:{b:align[a,b] for b in grams} for a in grams}))
    legacy_similarity=policy.similarity
    def with_geometry(a,b):
        return .5*(legacy_similarity(a,b)+align[a[1],b[1]])
    all_summaries={}
    for size,agent in paths.items():
        readout=payloads[size]
        metric=MetricSpec.from_payload(readout)
        require(metric.name=="accuracy" and list(metric.labels)==[0,1],"Unexpected task metric")
        labels=np.asarray(readout["y_search"])
        groups=sample_groups(TrainingTask("pair_text",readout["X_search"],labels),"shared_text")
        roles=np.asarray([int(hashlib.sha256(("B geometry role v1/"+str(g)).encode()).hexdigest()[:8],16)%2 for g in groups])
        profile,contract=read(agent/"training_profile.json"),read(agent/"task_contract.json")
        digest=sha(agent/"primary.joblib")
        with installed_methods():
            static=registered_capabilities(contract,len(readout["y_train"])+len(labels))
        caps,bank,gaps,keys={},{},{},{}
        for encoder,(key,entry,_) in by_rep.items():
            rec=entry[size]; member=rec["name"]
            file=agent/f"trial_{rec['trial']}.joblib"
            require(sha(file)==rec["sha256"],"Trial does not match the registered history")
            obj=joblib.load(file); facts=obj["metrics"]
            bank[member]=planner_observation(obj,facts,dict(sha256=digest,labels=labels),metric)
            if bank[member]["probabilities"] is None:
                raise ValueError("Candidate has no recorded probabilities")
            caps[member]=static[member];gaps[member]=facts["train_score"];keys[member]=key
            require(caps[member]["params"]["encoder"]==encoder,"Descriptor and actual representation differ")
        details=[]
        for role in (0,1):
            learn={m:slice_record(r,roles!=role) for m,r in bank.items()}
            held={m:slice_record(r,roles==role) for m,r in bank.items()}
            cases=[]
            for pair in combinations(sorted(bank,key=keys.get),2):
                prior=[dict(candidate=m,train_score=gaps[m],search_score=learn[m]["score"],split_id="primary",
                            status="completed") for m in pair]
                seen={"primary":{m:learn[m] for m in pair}}
                nominations={}
                for mode in ("descriptions","geometry"):
                    context=patch.object(policy,"similarity",with_geometry) if mode=="geometry" else nullcontext()
                    with context:
                        menu=policy.actions(caps,profile,prior,seen,metric,["primary"],4)
                    next_action=next(c for c in menu if c["action"]=="explore")
                    member=next_action["members"][0]
                    elected,correct=elect([*pair,member],learn,held,keys)
                    nominations[mode]=dict(next=member,elected=elected,correct=correct)
                a,b=nominations.values()
                cases.append(dict(start=list(pair),choices=nominations,changed=a["next"]!=b["next"],net=b["correct"]-a["correct"]))
            count=int((roles==role).sum())
            summary=dict(role=role,held_rows=count,opportunities=len(cases),changed=sum(c["changed"] for c in cases),
                         mean_held_delta=float(np.mean([c["net"] for c in cases])/count))
            lock(out/f"{size}_role_{role}.json",dict(summary=summary,cases=cases))
            details.append(summary)
        total=sum(r["opportunities"] for r in details)
        fraction=sum(r["changed"] for r in details)/total
        benefit=float(np.average([r["mean_held_delta"] for r in details],weights=[r["held_rows"] for r in details]))
        all_summaries[size]=dict(opportunities=total,roles=details,changed_fraction=fraction,mean_held_delta=benefit)
    advanced=all(v["changed_fraction"]>=.1 and v["mean_held_delta"]>=.002
                 and min(r["mean_held_delta"] for r in v["roles"])>=0 for v in all_summaries.values())
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),eligible_encoders=len(by_rep),by_training_size=all_summaries,
         next_full_run_warranted=advanced,overlapping_counterfactuals=True,new_base_fits=0,model_api_calls=0,
         designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(all_summaries,advanced,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args().root)
