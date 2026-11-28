"""Exercise the same production nomination on real, renamed, neutral and ideal feedback."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.ab_run import registered_capabilities
from superagent.autonomous_run import installed_methods, planner_observation
from superagent.experiment_policy import identity
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.open_methods import capability
from superagent.portfolio_review import evidence, nominate


PLAN=ROOT/"research/paired_encoder_search_1127/recruitment_replay_protocol.json"


def run(root):
    out=root/"recruitment-replay"
    out.mkdir(exist_ok=False)
    activate_resource_directory(root/"discovery-library")
    sources={"comparison":root/"corrected-delivery/corrected_delivery","movie":root/"movie-adviser/revised"}
    lock(out/"inputs.json",dict(plan=sha(PLAN),seals={k:sha(p/"sealed.json") for k,p in sources.items()}))
    results={}
    for task,folder in sources.items():
        agent=folder/"run/agent"
        final=read(agent/"final_selection.json")
        review=read(agent/"analogy_review.json")
        contract=read(agent/"task_contract.json")
        payload=joblib.load(agent/"primary.joblib")
        digest=sha(agent/"primary.joblib")
        require(digest==read(agent/"primary_split.json")["payload_sha256"],"Training panel changed")
        metric=MetricSpec.from_payload(payload)
        truth=np.asarray(payload["y_search"])
        with installed_methods():
            all_caps=registered_capabilities(contract,final["training_rows"])
        all_caps.update({k:capability(v) for k,v in read(agent/"candidate_registry.json")["entries"].items()})
        name=review["candidate"]
        all_caps[name]=dict(kind=contract["kind"],params=review["derived_configuration"],estimator="preference_analogy",
            classifier_fits=1,refit_classifier_fits=1,supports_probabilities=True,resource="local_frozen_encoder_cpu",
            origin="b_port")
        paths={r["candidate"]:agent/f"trial_{r['trial']}.joblib" for r in read(agent/"history.json")
               if r["split_id"]=="primary" and r["status"]=="completed"}
        paths[name]=agent/f"analogy_primary_{name}.joblib"
        obs,caps={},{}
        for m,p in paths.items():
            obj=joblib.load(p)
            facts=obj["metrics"]
            require(facts["resolved_configuration"]["params"]==all_caps[m]["params"],"Capability identity changed")
            rec=planner_observation(obj,facts,dict(sha256=digest,labels=truth),metric)
            if rec["probabilities"] is not None:
                obs[m],caps[m]=rec,all_caps[m]
        members=final["ensemble_components"] or [final["single_candidate"]]
        weights=final.get("weights",[1.])
        def decide(bank,catalog,anchors,budget=12):
            _,_,choice=nominate(bank,anchors,weights,catalog,budget,metric,(),set())
            return (choice["member"],choice["share"]) if choice else (None,0.)
        natural=decide(obs,caps,members)
        renamed=[]
        for seed in range(5):
            names={m:identity([seed,m])[:16] for m in obs}
            order=sorted(obs,key=names.get,reverse=bool(seed%2))
            replacement=decide({names[m]:obs[m] for m in order},{names[m]:caps[m] for m in order},[names[m] for m in members])
            equivalent=replacement==(names.get(natural[0]),natural[1])
            require(equivalent,"Renaming changed a production nomination")
            renamed.append(equivalent)
        base=evidence(obs,members,weights,metric)
        classes=np.asarray(metric.labels)
        neutral_obs=dict(base,predictions=classes[base["probabilities"].argmax(axis=1)])
        neutral={m:(rec if m in members else neutral_obs) for m,rec in obs.items()}
        require(decide(neutral,caps,members)==(None,0.),"Neutral nonmembers should add nothing")
        target=min((m for m in obs if m not in members),key=lambda m:identity(caps[m]["params"]))
        planted=dict(neutral)
        planted[target]=dict(obs[target],predictions=truth,probabilities=(truth[:,None]==classes).astype(float))
        trial=decide(planted,caps,members)
        cost=2*(sum(caps[m]["classifier_fits"] for m in members)+caps[target]["classifier_fits"])+caps[target]["refit_classifier_fits"]
        require(trial[0]==target and decide(planted,caps,members,cost-1)==(None,0.),"Counterfactual feedback or cost ignored")
        results[task]=dict(primary=digest,observed=len(obs),outside=len(obs)-len(members),natural=natural,
            alias_replays=len(renamed),matched=sum(renamed),neutral_retained=True,ideal_chose=trial,
            admissible_cost=cost,one_fit_short_retained=True)
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),tasks=results,new_fits=0,model_api_requests=0,
         hypothetical_scores_used_only_in_replay=True,course_designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(results,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
