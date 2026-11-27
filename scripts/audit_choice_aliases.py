"""Replay complete observed local decisions using opaque method names and reversed menu order."""
import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.ab_run import registered_capabilities
from superagent.autonomous_run import installed_methods,planner_observation
from superagent.experiment_policy import actions,identity,option_keys
from superagent.external_assets import activate_resource_directory
from superagent.finalist_race import FinalistRace
from superagent.metrics import MetricSpec
from superagent.open_methods import MethodSpace

SALTS=("violet","copper","marble")


class AuditLog:
    def write_json(self,*args):
        pass


def renamed(name,salt):
    return name if salt is None else "option_"+identity([salt,name])[:24]


def decide(salt,race,caps,records,history,profile,metric,folds,remaining,blocked):
    labels={n:renamed(n,salt) for n in caps}
    require(len(set(labels.values()))==len(labels),"Duplicate opaque identifier")
    names=list(caps) if salt is None else list(reversed(caps))
    offered={}
    for name in names:
        item=dict(caps[name],id=labels[name])
        if item.get("route")==name:
            item["route"]=labels[name]
        offered[labels[name]]=item
    observations={}
    for fold,bank in records.items():
        idx=list(bank) if salt is None else list(reversed(bank))
        observations[fold]={labels[n]:bank[n] for n in idx}
    earlier=[dict(r,candidate=labels[r["candidate"]]) for r in history]
    blocked={labels[n] for n in blocked}
    menu=actions(offered,profile,earlier,observations,metric,list(folds),remaining,blocked)
    chosen=menu[0]
    if race is not None:
        opponent=race.consider(menu,offered,observations,metric,remaining,blocked,
                               "linear_only" not in profile.get("method_restrictions",()))
        if opponent is not None:
            chosen=opponent
    inverse={v:k for k,v in labels.items()}
    return chosen["action"],sorted(inverse[n] for n in chosen["members"]),chosen["split"]


def previous_ties(root):
    case=root/"resource-raw/resource_default/run/agent"
    previous=[]
    for step in read(case/"decisions.json"):
        ch=step["choice"]
        if ch["action"]!="explore":
            continue
        tied=[a for a in step["alternatives"] if a["action"]==ch["action"] and a["utility"]==ch["utility"]
              and len(a["members"])==len(ch["members"])]
        if len(tied)<2:
            continue
        outcomes=[]
        for salt in SALTS:
            winner=min(tied,key=lambda a:identity([a["action"],[renamed(n,salt) for n in a["members"]],a["split"]]))
            outcomes.append(winner["members"])
        previous.append(dict(step=step["step"],chosen=ch["members"],tied=[a["members"] for a in tied],renamed_selections=outcomes))
    return previous


def audit(directory):
    agent=directory/"run/agent"
    spec=read(agent/"task_contract.json")
    profile=read(agent/"training_profile.json")
    decisions=read(agent/"decisions.json")
    history=read(agent/"history.json")
    manifest=read(agent/"run_manifest.json")
    model=read(agent/"final_selection.json")
    require(model["refit"] and read(agent/"run_outcome.json")["status"]=="completed","Incomplete source trajectory")
    metric=MetricSpec.from_payload({"metric_spec":spec["metric"]})
    inputs,folds={},{}
    for name in ("primary","confirmation_1","confirmation_2"):
        path=agent/(name+".joblib")
        if path.exists():
            split=read(agent/(name+"_split.json"))
            require(split["payload_sha256"]==sha(path),"Training partition changed")
            inputs[name]=dict(sha256=sha(path),labels=joblib.load(path)["y_search"])
            folds[name]=(split["fit"],split["held"],split["resource"])
    is_race="automatic_finalist_review_available" in (agent/"events.jsonl").read_text()
    races={s:FinalistRace(AuditLog(),folds,frontier=True,multiple=True) if is_race else None for s in (None,*SALTS)}
    records={f:{} for f in inputs}
    blocked=set()
    rows=[]
    intervention=None
    with installed_methods(), TemporaryDirectory() as temp:
        static=registered_capabilities(spec,model["training_rows"],numeric_kernel_width=profile["statistics"].get("feature_count"))
        space=MethodSpace(spec,profile,Path(temp))
        for index,step in enumerate(decisions):
            prior=history[:index]
            caps,generation=space.refresh(prior,static)
            require(generation["sha256"]==step["registry_sha256"],"Reconstructed candidate registry differs")
            recorded=step["choice"]
            actual=(recorded["action"],sorted(recorded["members"]),recorded["split"])
            remaining=step["remaining_fits"]
            original=decide(None,races[None],caps,records,prior,profile,metric,folds,remaining,blocked)
            require(original==actual,f"Observed step {index+1} did not replay: {original} versus {actual}")
            aliases=[decide(s,races[s],caps,records,prior,profile,metric,folds,remaining,blocked) for s in SALTS]
            require(all(replayed==actual for replayed in aliases),f"Display labels change the action at step {index+1}")
            keys=option_keys(caps,records.get("primary",{}))
            rows.append(dict(step=index+1,action=actual[0],members=actual[1],fold=actual[2],
                             menu_size=len(caps),distinct_execution_keys=len(set(keys.values())),opaque_replays=len(aliases)))
            if actual[0]=="stop" and not is_race and len(actual[1])==1 and remaining==1:
                ranked=sorted(records["primary"],key=lambda n:-records["primary"][n]["score"])
                if len(ranked)>=2:
                    x,z=ranked[:2]
                    flipped={f:dict(bank) for f,bank in records.items()}
                    flipped["primary"][x],flipped["primary"][z]=flipped["primary"][z],flipped["primary"][x]
                    earlier=deepcopy(prior)
                    originals={r["candidate"]:r for r in prior if r["status"]=="completed" and r["split_id"]=="primary"}
                    for row in earlier:
                        if row["split_id"]=="primary" and row["candidate"] in (x,z):
                            other=originals[z if row["candidate"]==x else x]
                            row.update(train_score=other["train_score"],search_score=other["search_score"])
                    moved=decide(None,None,caps,flipped,earlier,profile,metric,folds,remaining,blocked)
                    require(moved!=actual and moved[0]=="stop","Different evidence did not change the measured terminal decision")
                    intervention=dict(unchanged_configurations=True,exchanged_primary_records=[x,z],actual=actual[1],
                                      alternate=moved[1],scope="counterfactual decision replay")
            if index<len(history):
                trial=history[index]
                if trial["status"]!="completed":
                    blocked.add(trial["candidate"])
                else:
                    bundle=joblib.load(agent/f"trial_{trial['trial']}.joblib")
                    fold=trial["split_id"]
                    records[fold][trial["candidate"]]=planner_observation(bundle,bundle["metrics"],inputs[fold],metric)
    return dict(runtime_sha256=identity(manifest["source_files"]), decisions=len(rows), matched_opaque_replays=len(rows)*len(SALTS),
                automatic_finalist_race=is_race,terminal_feedback_intervention=intervention, steps=rows,
                actual_classifier_fits=read(agent/"run_outcome.json")["attempted_classifier_fits"],new_fits_for_replay=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","weights"):
        p.add_argument("--"+key,required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    activate_resource_directory(args.weights)
    out=args.root/"anonymous-choice"
    evidence=dict(original_ties=previous_ties(args.root),new_cases={})
    for case,folder in (("task2",out/"anonymous_default"),("image",args.root/"anonymous-choice-image")):
        evidence["new_cases"][case]=audit(folder)
    evidence.update(protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/anonymous_choice_protocol.json"),
                    image_protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/anonymous_choice_image_protocol.json"),
                    external_targets_opened=False,new_model_api_requests=0,salts=list(SALTS))
    save(out/"alias_audit.json",evidence)
    print({name:(v["decisions"],v["matched_opaque_replays"],v["terminal_feedback_intervention"]) for name,v in evidence["new_cases"].items()})
