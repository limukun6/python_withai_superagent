"""Identify the representation a real elected encoder would actually lend to a derived voter."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

import superagent
from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import measure
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import env, read, require, sha
from superagent.external_assets import activate_resource_directory, model_resources
from superagent.local_sentence import LocalSentenceFeatures, describe
from superagent.metrics import MetricSpec
from superagent.preference_analogy import derived_candidate, embedding_function, views


PLAN=Path(__file__).resolve().parents[1]/"research/paired_encoder_search_1127/local_adviser_identity_protocol.json"


def evidence(model, texts):
    original=model.named_steps["features"]
    source=original.resource()
    candidate=derived_candidate(model)
    require(candidate is not None,"Missing derived candidate")
    feature=candidate.fit_predict.keywords["feature"]
    target=model_resources(feature)[0]
    require(len(target)==1,"Unresolved actual adviser weights")
    expected=embedding_function(original)(texts)
    actual=embedding_function(feature)(texts)
    def vector_hash(matrix):
        return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    return candidate,dict(expected_revision=source["revision"],declared_revision=candidate.params["revision"],
         actual_revision=target[0]["revision"],expected_vector_hash=vector_hash(expected),actual_vector_hash=vector_hash(actual),
         dimensions=[expected.shape[1],actual.shape[1]],candidate=candidate.family,
         exact_match=bool(np.array_equal(expected,actual)))


def run(root, arm):
    out=root/"local-adviser-identity"/arm
    out.mkdir(parents=True,exist_ok=False)
    impl=Path(superagent.__file__).resolve().parents[1]
    asset=root/"arctic-library"
    activate_resource_directory(asset)
    os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(root/"feature_bank")
    base=root/"last-token-subset/original/run/agent"
    payload=base/"primary.joblib"
    data=joblib.load(payload)
    texts=list(dict.fromkeys(s for row in data["X_train"] for s in row))[:12]
    history=[r for r in read(base/"history.json") if r["split_id"]=="primary" and r["status"]=="completed"
             and str(r["resolved_configuration"]["params"].get("encoder","")).startswith("local-sentence-")]
    require(history,"Archive contains no previously fitted, discovered encoders")
    evidence_files={r["candidate"]:sha(base/f"trial_{r['trial']}.joblib") for r in history}
    lock(out/"inputs.json",dict(protocol=sha(PLAN),runtime=source_identity(impl),payload=sha(payload),trials=evidence_files,
         probe_strings=len(texts),expanded_publisher=sha(root/"arctic-installed.json")))
    rows=[]
    first=None
    for trial in history:
        model=joblib.load(base/f"trial_{trial['trial']}.joblib")["model"]
        candidate,row=evidence(model,texts)
        require(row["expected_revision"]==trial["resolved_configuration"]["params"]["revision"],"Wrong stored classifier")
        rows.append(dict(trial=trial["trial"],member=trial["candidate"],**row))
        if first is None:
            first=(candidate,model.named_steps["features"])
    spec=describe(asset/"snowflake-arctic-embed-s")
    _, added=evidence(Pipeline([("features",LocalSentenceFeatures("pair_text",spec))]),texts)
    verified_fit=None
    if arm=="after":
        require(all(r["exact_match"] and r["expected_revision"]==r["declared_revision"]==r["actual_revision"] for r in [*rows,added]),
                "Corrected source still differs")
        require(len({r["candidate"] for r in [*rows,added]})==len(rows)+1,"Two distinct local sources map to the same adviser")
        candidate,original=first
        observed=out/"physical.jsonl"
        settings=env(asset,observed)
        labels=np.asarray(data["y_search"])
        folder=out/"worker"
        probability=measure(candidate.family,payload,base/"runtime_registry_0.json",folder,settings,
                            MetricSpec.from_payload(data),labels)
        model=joblib.load(folder/"trial.joblib")["model"]
        learner=model.named_steps["classifier"]
        resources=model_resources(model)[0]
        change,context=views(data["X_train"][:16],embedding_function(original))
        require(learner.features.spec==original.spec and len(resources)==1
                and resources[0]["revision"]==original.spec["signature"] and len(observed.read_text().splitlines())==1
                and np.allclose(learner.index_.change[:16],change,rtol=0,atol=2e-5)
                and np.allclose(learner.index_.context[:16],context,rtol=0,atol=2e-5),
                "Fitted voter or exported resources do not match its source")
        verified_fit=dict(input_member=rows[0]["member"],revision=resources[0]["revision"],training_rows=len(data["y_train"]),
                          held_training_rows=len(probability),stored_geometry_matches_training_source=True,
                          portable_resources=1,supervised_fits=1)
    result=dict(inputs=sha(out/"inputs.json"),fitted_sources=rows,added_resource_only=added,
                verified_worker=verified_fit,correct_sources=sum(r["exact_match"] for r in rows),
                supervised_fits=int(verified_fit is not None),model_api_requests=0,designated_validation_opened=False,
                external_movie_labels_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(arm,result["correct_sources"],len(rows),added["exact_match"],verified_fit,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=("before","after"))
    parser.add_argument("--root",type=lambda v:Path(v).resolve(),required=True)
    args=parser.parse_args()
    run(args.root,args.arm)
