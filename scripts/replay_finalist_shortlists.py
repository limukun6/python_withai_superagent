"""Replay only measured training predictions on predeclared non-target development problems."""
import argparse
from pathlib import Path

import joblib

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.finalist_race import FinalistRace, shortlist
from superagent.metrics import MetricSpec


class MemoryLog(dict):
    def write_json(self,name,value):
        self[name]=value


def replay(agent):
    contract=read(agent/"task_contract.json")
    metric=MetricSpec.from_payload(dict(metric_spec=contract["metric"]))
    data,observed,windows={},{},{}
    for split in ("primary","confirmation_1","confirmation_2"):
        path=agent/(split+".joblib")
        if not path.exists():continue
        spec=read(agent/(split+"_split.json"))
        require(sha(path)==spec["payload_sha256"],"Frozen training split changed")
        data[split]=dict(sha256=spec["payload_sha256"],labels=joblib.load(path)["y_search"])
        windows[split]=(spec["fit"],spec["held"],{})
        observed[split]={}
    for event in read(agent/"history.json"):
        name,fold=event["candidate"],event["split_id"]
        if event["status"]=="completed" and fold in data:
            artifact=joblib.load(agent/("trial_"+str(event["trial"])+".joblib"))
            observed[fold][name]=planner_observation(artifact,event,data[fold],metric)
    primary=observed["primary"]
    result=dict(modality=contract["kind"],metric=metric.name,classes=len(metric.labels),measured_singles=len(primary))
    found=shortlist(primary,metric,"linear_only" not in contract.get("method_restrictions",[]),set(primary))
    if found is None:
        return dict(result,nomination=None,reason="No different competitive alternative met the training gate")
    anchor,challenger,groups,change=found
    members=set(anchor)|set(challenger)
    occupied=set(windows["primary"][1])
    fresh=[]
    for f in observed:
        held=set(windows[f][1])
        if f!="primary" and members.issubset(observed[f]) and not occupied.intersection(held):
            fresh.append(f)
            occupied.update(held)
    result.update(anchor=list(anchor),nomination=list(challenger),primary_change=change,fully_matched_confirmation=fresh)
    if fresh:
        race=FinalistRace(MemoryLog(),windows)
        race.plan=dict(anchor=list(anchor),groups=groups,fresh=fresh)
        choice=race.finish(observed,metric,{k:dict(refit_classifier_fits=1) for k in members})
        result["decision_given_available_evidence"]=choice["members"]
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cases={n:replay(args.development/n/"feedback-local/run/agent") for n in ("v21a","v21b","v83a","v83b")}
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"results.json",dict(cases=cases,fits=0,requests=0,
         protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/replay_protocol.json")))
    print([(k,v["modality"],v["nomination"],v.get("fully_matched_confirmation",[])) for k,v in cases.items()])
