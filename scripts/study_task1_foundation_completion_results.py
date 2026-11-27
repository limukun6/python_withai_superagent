"""Publish compact evidence for the two remaining folds and frozen encoder ablations."""
import argparse
from pathlib import Path

from scripts.study_task1_seed_completion import origin
from scripts.unified_released_study import ROOT, read, require, save, sha

REGISTRY = ROOT / "research/task1_feedback_extension_1127"


def export(out):
    dev = out / "development"
    paths = dict(cold_five_fold=out / "seed_completion/result.json", paired_training_weights=out / "paired_weights/result.json",
                 five_seed_stopping=out / "seed_stopping/result.json",
                 profile_clusters=out / "profile_uncertainty.json",
                 early_rows=dev / "frozen_row_representations/sealed.json",
                 early_nomination=dev / "rowembed_election/locked.json", early_election=dev / "rowembed_election/results.json",
                 early_matched_heads=dev / "rowembed_election/standalone.json",
                 late_rows=dev / "adapted_readout_contexts/sealed.json", late_nomination=dev / "adaptread_election/locked.json",
                 late_election=dev / "adaptread_election/results.json")
    data = {k:read(v) for k,v in paths.items()}
    matched = read(dev / "frozen_row_representations/heads.json")
    fits = dict(remaining_two_cold_solves=data["cold_five_fold"]["further_cold_fits"],
                matched_weighted_repeats=data["paired_training_weights"]["fits"],
                five_seed_pilots_and_full_fits=data["five_seed_stopping"]["fits"],
                early_frozen_context=read(dev / "frozen_row_representations/fitted.json")["foundation_context_fits"],
                early_supervised_heads=matched["supervised_heads"],
                late_frozen_contexts=data["late_rows"]["new_contexts"], late_adapted_heads=data["late_rows"]["adapted_heads"])
    require(data["cold_five_fold"]["rows"] == data["paired_training_weights"]["rows"] == 8117
            and data["early_rows"]["fit"] == sha(dev / "frozen_row_representations/fitted.json"),
            "Missing paired controls or encoder")
    prior = REGISTRY / "seed_confirmation_results.json"
    before = read(prior)["cumulative"]
    value = dict(previous=sha(prior), previous_cumulative=before, recording_rule="One count per classifier, context construction or supervised decision head. Feature-only PCA, scaling, and vocabulary operations are listed separately.",
                 results=data, receipts={k:sha(p) for k,p in paths.items()}, fits=fits, stage_total=sum(fits.values()),
                 cumulative=before+sum(fits.values()), additional_feature_operations=dict(early_fit_only_pca=matched["unsupervised_pca"],
                     raw_one_hot_vocabulary=matched["one_hot_vocabulary"], early_hidden_scaler=matched["feature_scaler"],
                     late_fit_only_whitened_pca=data["late_rows"]["pca_fits"]),
                 model_api_requests=0,new_model_api_currency_rmb=0,official_test_opened=False,h2_opened=False)
    save(REGISTRY / "foundation_completion_results.json",value)
    actions = []
    for i in range(5):
        side = origin(out,i) / "replicated"
        report = side / "run/agent/seed_replication.json"
        seal = read(side / "sealed.json")
        decision = read(report)
        require(seal["model"] == (decision["model"] if decision["action"]=="adopt" else decision["incumbent"]),
                "Recorded decision differs from the exported model")
        actions.append(dict(fold=i, final=seal["model"], actual_fits=seal["fits"], independent_reload=seal["independent_reload"],
                            source_commit="f214f8b33b7066fb4c670379fc6bbb4920553dab",
                            decision_sha256=sha(report), decision=decision))
    save(REGISTRY / "seed_policy_actions.json",dict(folds=actions,summary_sha256=sha(paths["cold_five_fold"])))
    print(fits,value["stage_total"],value["cumulative"],flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    export(parser.parse_args().out)
