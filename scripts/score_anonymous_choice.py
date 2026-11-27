"""Score the sealed ordinary alias-invariant run, reusing an existing score for identical predictions."""
import argparse
from pathlib import Path

import numpy as np

from scripts.score_followup_entry import prediction_file
from scripts.summarize_encoder_experiments import trace
from scripts.unified_released_study import archive,read,require,save,sha,title_rows

PATHS={"legacy_long":"long-cold/original","frontier":"frontier-raw/frontier","frontier_scale":"frontier-raw/frontier_scale",
       "auto_portfolio":"auto-raw/auto_portfolio","resource_default":"resource-raw/resource_default"}


def score(root,course):
    output=root/"anonymous-choice-scores"
    output.mkdir(exist_ok=True)
    name="anonymous_default"
    run=root/"anonymous-choice"/name
    audit_path=root/"anonymous-choice/alias_audit.json"
    audit=read(audit_path)["new_cases"]["task2"]
    require(audit["matched_opaque_replays"]==3*audit["decisions"] and audit["terminal_feedback_intervention"],
            "Ordinary entry did not complete its decision audit")
    launch=read(run/"launch.json")
    require(launch==dict(individual_model_variables=[],model_hint=None,review_flag=[],limits=[],resource_directory_argument=True),
            "New solver was given extra instructions")
    plan=read(run/"protocol.json")
    old_plan=read(root/PATHS["resource_default"]/"protocol.json")
    require(plan["features_sha256"]==old_plan["features_sha256"] and plan["raw_sha256"]==old_plan["raw_sha256"]
            and plan["candidate_hint"] is None,"Comparison changed the inputs")
    frozen, predicted=prediction_file(run)
    require(frozen["rows"]==8301 and frozen["fits"]==audit["actual_classifier_fits"]<=10 and frozen["api"]==0,
            "Incomplete ordinary run")
    previous=root/"delivery_scores"
    lock=read(previous/"evaluation_lock.json")
    historical=read(previous/"results.json")
    require(historical["lock_sha256"]==sha(previous/"evaluation_lock.json"),"Old score lock differs")
    inputs=dict(seals={name:sha(run/"sealed.json")},prior_score_sha256=sha(previous/"results.json"),alias_audit_sha256=sha(audit_path))
    new_lock=output/"evaluation_lock.json"
    if new_lock.exists():
        require(read(new_lock)==inputs,"Prospective score lock has changed")
    else:
        save(new_lock,inputs)
    if (output/"results.json").exists():
        saved=read(output/"results.json")
        require(saved["lock_sha256"]==sha(new_lock),"Scored output changed")
        return saved
    matched=None
    baseline=None
    for prior,relative in PATHS.items():
        old_folder=root/relative
        require(sha(old_folder/"sealed.json")==lock["seals"][prior],"Prior model seal differs")
        seal, values=prediction_file(old_folder)
        if prior=="resource_default":
            baseline=values
        if seal["prediction_sha256"]==frozen["prediction_sha256"]:
            require(np.array_equal(values,predicted),"Prediction identity differs")
            matched=prior
    if matched is None:
        raw=archive(course,"titles")
        require(sha(raw)==plan["raw_sha256"],"Unexpected course archive")
        truth=np.asarray([r[2] for r in title_rows(raw,"val",labels=True)])
        require(len(truth)==len(predicted) and np.isin(truth,[0,1]).all(),"Invalid reference classes or size")
        correct=int(np.sum(truth==predicted))
    else:
        correct=historical["completed"][matched]["correct"]
    base_correct=historical["completed"]["resource_default"]["correct"]
    disagreement=int(np.sum(predicted!=baseline))
    # On a binary task every disagreement changes exactly one of two correctness indicators.
    numerator=disagreement+correct-base_correct
    require(numerator%2==0 and 0<=numerator<=2*disagreement,"Inconsistent paired binary counts")
    gained=numerator//2
    receipt=dict(correct=correct,rows=len(predicted),accuracy=correct/len(predicted),gained_vs_previous_default=gained,
                 lost_vs_previous_default=disagreement-gained,identical_to=matched,trace=trace(run))
    result=dict(lock_sha256=sha(new_lock),completed={name:receipt},new_designated_validation_read=matched is None,
                formal_test_opened=False,H2_opened=False,new_model_requests=0)
    save(output/"results.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","course"):
        p.add_argument("--"+key,type=lambda s:Path(s).resolve(),required=True)
    args=p.parse_args()
    row=score(args.root,args.course)["completed"]["anonymous_default"]
    print(row["correct"],row["rows"],row["identical_to"],row["trace"]["members"])
