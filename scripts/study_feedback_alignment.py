"""Test if measured probability residuals predict the best next representation."""
import argparse
import hashlib
from itertools import combinations
import os
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.study_representation_geometry import ARMS, elect, slice_record
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


PLAN = ROOT/"research/paired_encoder_search_1127/feedback_alignment_protocol.json"


def independent_probes(rows,groups):
    first = {}
    for i,(pair,group) in enumerate(zip(rows,groups)):
        key = policy.identity(["residual probe v1",*sorted(pair)])
        if group not in first or key<first[group][0]:
            first[group] = (key,i)
    return np.asarray([i for _,i in sorted(first.values())[:256]])


def normalized_cross_terms(matrix):
    values = matrix[np.triu_indices(len(matrix),k=1)].astype(np.float64)
    return values/max(np.linalg.norm(values),1e-12)


def run(root):
    out = root/"feedback-alignment"
    out.mkdir(exist_ok=False)
    archive = root/"representation-geometry/inputs.json"
    sources = read(archive)
    lock(out/"inputs.json",dict(protocol=sha(PLAN),archive=sha(archive),data=sources["data"],
                               eligible={e:v["config"] for e,v in sources["representations"].items()}))
    activate_resource_directory(root/"discovery-library")
    os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"] = str(root/"feature_bank")
    measurements = {}
    for size,path in ARMS.items():
        agent = root/path
        digest = sha(agent/"primary.joblib")
        require(digest==sources["data"][size],"Training panel changed")
        data = joblib.load(agent/"primary.joblib")
        metric = MetricSpec.from_payload(data)
        truth = np.asarray(data["y_search"])
        rows = data["X_search"]
        require(metric.name=="accuracy" and list(metric.labels)==[0,1] and data["pair_semantics"]=="preference",
                "Replay requires a two-sided probability residual")
        groups = sample_groups(TrainingTask(data["kind"],rows,truth),"shared_text")
        roles = np.asarray([int(hashlib.sha256(("B residual role v1/"+str(g)).encode()).hexdigest()[:8],16)%2 for g in groups])
        profile = read(agent/"training_profile.json")
        with installed_methods():
            catalog = registered_capabilities(read(agent/"task_contract.json"),len(data["y_train"])+len(truth))
        bank,caps,keys,gaps,providers = {},{},{},{},{}
        for encoder,entry in sources["representations"].items():
            saved = entry["trials"][size]
            name = saved["name"]
            trial = agent/f"trial_{saved['trial']}.joblib"
            require(sha(trial)==saved["sha256"],"Archived worker differs")
            bundle = joblib.load(trial)
            facts = bundle["metrics"]
            caps[name] = catalog[name]
            require(caps[name]["params"]["encoder"]==encoder,"Wrong historical representation")
            keys[name] = entry["config"]
            gaps[name] = facts["train_score"]
            providers[name] = embedding_function(bundle["model"].named_steps["features"])
            bank[name] = planner_observation(bundle,facts,dict(sha256=digest,labels=truth),metric)
        details = []
        for role in (0,1):
            learn_ix = np.flatnonzero(roles!=role)
            learn = {m:slice_record(v,roles!=role) for m,v in bank.items()}
            held = {m:slice_record(v,roles==role) for m,v in bank.items()}
            selected = independent_probes([rows[i] for i in learn_ix],groups[learn_ix])
            probes = [rows[i] for i in learn_ix[selected]]
            kernel = {}
            for m,provider in providers.items():
                difference,_ = views(probes,provider)
                kernel[m] = normalized_cross_terms(difference @ difference.T)
            cases = []
            for pair in combinations(sorted(learn,key=keys.get),2):
                incumbent,_ = elect(pair,learn,learn,keys)
                p = np.mean([learn[m]["probabilities"][selected,1] for m in incumbent],axis=0)
                residual = truth[learn_ix[selected]]-p
                target = normalized_cross_terms(np.outer(residual,residual))
                seen = {"primary":{m:learn[m] for m in pair}}
                history = [dict(candidate=m,train_score=gaps[m],search_score=learn[m]["score"],split_id="primary",
                                status="completed") for m in pair]
                menu = [a["members"][0] for a in policy.actions(caps,profile,history,seen,metric,["primary"],4)
                        if a["action"]=="explore"]
                scores = {m:float(np.dot(kernel[m],target)) for m in menu}
                ordinary, proposed = menu[0], max(menu,key=scores.get)
                choices = {}
                for mode,name in (("ordinary",ordinary),("residual",proposed)):
                    winner,correct = elect([*pair,name],learn,held,keys)
                    choices[mode] = dict(next=name,elected=winner,held_correct=correct)
                cases.append(dict(start=list(pair),incumbent=incumbent,alignment=scores,choices=choices,
                    changed=ordinary!=proposed,net=choices["residual"]["held_correct"]-choices["ordinary"]["held_correct"]))
            held_count = int((roles==role).sum())
            summary = dict(role=role,held_rows=held_count,probed_groups=len(selected),cases=len(cases),
                           changed=sum(c["changed"] for c in cases),mean_held_delta=float(np.mean([c["net"] for c in cases])/held_count))
            lock(out/f"{size}_{role}.json",dict(summary=summary,probe_key=fingerprint(learn_ix[selected]),cases=cases))
            details.append(summary)
        measurements[size] = dict(roles=details,changed=sum(s["changed"] for s in details)/sum(s["cases"] for s in details),
            held_delta=float(np.average([s["mean_held_delta"] for s in details],weights=[s["held_rows"] for s in details])))
    passed = all(s["changed"]>=.1 and s["held_delta"]>=.0025 and min(r["mean_held_delta"] for r in s["roles"])>=0
                 for s in measurements.values())
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),by_training_size=measurements,
         further_solve_warranted=passed,overlapping_replays=True,new_fits=0,model_api_requests=0,
         designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(measurements,passed,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args().root)
