"""Test feedback dependence using a locked real experiment and its unaltered training observations."""
import argparse
from copy import deepcopy
from pathlib import Path

import joblib

from scripts.unified_released_study import read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.finalist_race import FinalistRace
from superagent.metrics import MetricSpec


class Recorder:
    def write_json(self,name,value):
        self.result=value


def run(agent,output):
    locked=read(agent/"finalist_lock.json")
    recorded=read(agent/"finalist_result.json")
    names=locked["members"]
    require(len(names)==2 and all(len(g)==1 for g in locked["groups"]),"Expected two locked single finalists")
    folds={}
    observed={}
    caps={}
    history=read(agent/"history.json")
    for f in ["primary",*locked["fresh"]]:
        split=read(agent/(f+"_split.json"))
        p=agent/(f+".joblib")
        require(sha(p)==split["payload_sha256"],"Training partition differs")
        data=joblib.load(p)
        labels=data["y_search"]
        metric=MetricSpec.from_payload(data)
        folds[f]=(split["fit"],split["held"],{})
        observed[f]={}
        for n in names:
            row=next(r for r in history if r["candidate"]==n and r["split_id"]==f and r["status"]=="completed")
            trial=joblib.load(agent/f"trial_{row['trial']}.joblib")
            fact=trial["metrics"]
            require(fact["resolved_configuration"]["id"]==n and fact["classifier_fits"]==1,"Changed method or cost")
            caps[n]=dict(refit_classifier_fits=1)
            observed[f][n]=planner_observation(trial,fact,dict(labels=labels,sha256=split["payload_sha256"]),metric)
    def decide(plan,bank,options):
        recorder=Recorder()
        race=FinalistRace(recorder,folds)
        race.plan=plan
        action=race.finish(bank,metric,options)
        return action["members"]
    original=decide(locked,observed,caps)
    require(original==recorded["selected"],"Could not replay the real decision")
    alias={n:f"anonymous_{i}" for i,n in enumerate(reversed(names))}
    remap=lambda g:[alias[n] for n in g]
    plan=deepcopy(locked)
    for k in ("members","anchor","challenger"):
        plan[k]=remap(plan[k])
    plan["groups"]=[remap(g) for g in reversed(plan["groups"])]
    anonymous={f:{alias[n]:bank[n] for n in reversed(names)} for f,bank in observed.items()}
    options={alias[n]:caps[n] for n in reversed(names)}
    renamed=decide(plan,anonymous,options)
    require(renamed==remap(original),"Aliasing changed a strict score comparison")
    switched={f:b.copy() for f,b in anonymous.items()}
    a,b=remap(names)
    for f in plan["fresh"]:
        switched[f][a],switched[f][b]=switched[f][b],switched[f][a]
    new=decide(plan,switched,options)
    reverse={v:k for k,v in alias.items()}
    result=dict(real=original,renamed_and_reordered=[reverse[n] for n in renamed],
                exchanged_independent_evidence=[reverse[n] for n in new],decision_changed=new!=renamed,
                observed_confirmations=plan["fresh"],primary_evidence_unchanged=True,lock_sha256=sha(agent/"finalist_lock.json"),
                new_classifier_fits=0,new_model_requests=0,designated_validation_opened=False,
                scope="Causal check of the final selection step; exchanged observations are counterfactual.")
    save(output,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("agent","output"):
        parser.add_argument("--"+name,type=lambda s:Path(s).resolve(),required=True)
    args=parser.parse_args()
    print(run(args.agent,args.output))
