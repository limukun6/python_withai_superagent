"""Export compact reproducible research receipts without examples, weights or label files."""
import argparse
from pathlib import Path

from scripts.unified_released_study import read,save,sha


def run(out,destination):
    dev=out/"development"
    folds=out/"fusion_match"
    rounds=read(folds/"result.json")
    primary=read(dev/"outer_results.json")
    anchor=read(out/"anchor/result.json")
    retained=read(out/"default_compatibility/result.json")
    same=(retained["local_prediction_sha256"]==anchor["local_prediction_sha256"] and retained["replayed"])
    evidence={}
    mapping={
        "compact_export":out/"portable_table/result.json",
        "initial_election":dev/"election/results.json",
        "initial_nomination":dev/"election/nomination.json",
        "outer":dev/"outer_results.json",
        "confidence":dev/"feedback_fusion/results.json",
        "contextual_icl":dev/"icl_context_election/results.json",
        "contextual_icl_nomination":dev/"icl_context_election/locked.json",
        "tabm":dev/"tabm_election/results.json",
        "tabm_nomination":dev/"tabm_election/locked.json",
        "boosting":dev/"boost_election/results.json",
        "boosting_nomination":dev/"boost_election/locked.json",
        "conditional_trust":dev/"disagreement/results.json",
        "three_fold_match":folds/"result.json",
        "first_optional_probe":dev/"coverage32/run/agent/coverage_review.json",
    }
    receipts={name:sha(p) for name,p in mapping.items()}
    for name,p in mapping.items():
        evidence[name]=read(p)
    receipts["designated_replay"]=sha(out/"anchor/result.json")
    fits=dict(native_reference=anchor["actual_fits"],retained_default_compatibility=retained["actual_fits"],ordinary_outer=sum(m["fits"] for m in primary["ordinary"].values()),
              initial_candidate_screen=3,frozen_recipe_diagnostic=read(dev/"diagnostics/sealed.json")["fits"],
              confidence_gate=1,conditional_trust_gate=evidence["conditional_trust"]["fits"],
              contextual_table_models=sum(read(dev/("icl_"+n)/"sealed.json")["physical_fits"] for n in
                                          ("global_structure","conditional_context")),
              tabm_optimizer_trainings=2,alternative_boosting_fits=3,
              new_three_fold_entries=sum(item["fits"] for row in rounds["folds"] for item in row["methods"].values()))
    if (out/"designated_result.json").exists():
        evidence["qualified_designated"]=read(out/"designated_result.json")
        fits["qualified_designated"]=read(out/"release/result.json")["actual_fits"]
    result=dict(provenance=dict(teammate_runtime="534d5992870a8be952d1cd7be3afb2ca23512b9c",
         cap_only_control="b20bbbca8b47dff7b1fafc42d32f8c6ac8787e45",expanded_menu="ff5edb63",
         coverage="55b4ab41",final_holdout="e85c4337"),source_receipts=receipts,
         historical_designated=dict(correct=1549,rows=2030,prediction=anchor["historical_prediction_sha256"]),
         retained_default=dict(matches_local_anchor=same,source=read(out/"default_compatibility/inputs.json")["source"],
                               predictions=retained["local_prediction_sha256"],methods=retained["members"]),
         mac_anchor=dict(methods=anchor["members"],prediction=anchor["local_prediction_sha256"],
                         matches_historical_bytes=anchor["historical_bytes_match"],refit_and_reloaded=anchor["replayed"]),
         completed_task_estimator_or_router_fits=fits,completed_task_fits_total=sum(fits.values()),
         separate_engineering=dict(tabicl_failed_real_preprocessing_attempts=1,named_optional_table_smoke_fits=2,
                                   optional_tabicl_ensemble_views=4,tabm_shared_heads=8,ebm_internal_bags=4),
         language_model_api_requests=0,input_tokens=0,output_tokens=0,incremental_course_usage_rmb=0,
         official_test_opened=False,h2_opened=False,results=evidence)
    save(destination,result)
    print(fits,sum(fits.values()),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    parser.add_argument("--destination",required=True,type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    run(a.out,a.destination)
