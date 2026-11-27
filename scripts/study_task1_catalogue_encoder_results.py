"""Collect paired results, environment receipts and fit counts from the catalogue and encoder studies."""
import argparse
from pathlib import Path

from scripts.unified_released_study import ROOT, read, require, save, sha

REGISTRY = ROOT / "research/task1_feedback_extension_1127"


def export(out):
    dev = out / "development"
    paths = {
        "ordinary_catalogue": out / "ordinary_catalogue/result.json",
        "leaf": dev / "leaf_election/results.json", "leaf_vs_single": dev / "leaf_election/vs_original.json",
        "leaf_nomination": dev / "leaf_election/locked.json", "leaf_components": dev / "leaf_encoder_components_02/sealed.json",
        "masked": dev / "masked_election/results.json", "masked_controls": dev / "masked_election/controls.json",
        "masked_components": dev / "masked_components/sealed.json", "masked_nomination": dev / "masked_election/locked.json",
        "adaptive_memory": dev / "adaptmem_election/results.json", "adaptive_single": dev / "adaptmem_election/anchor.json",
        "adaptive_teachers": dev / "adaptive_memory_components/teachers.json",
        "tabdpt": dev / "tabdpt_election/results.json", "tabdpt_nomination": dev / "tabdpt_election/locked.json",
        "tabdpt_probe": dev / "tabdpt_probe.json", "decision_threshold": out / "threshold_replay/result.json",
        "search_nominee_blending": out / "policy_nominees/result.json",
    }
    records = {k:read(p) for k,p in paths.items()}
    counts = {}
    catalogue = records["ordinary_catalogue"]
    counts["four_ordinary_solutions"] = sum(m["fits"] for r in catalogue["folds"] if r["fold"] in (3,4)
                                             for m in r["methods"].values())
    first_leaf = dev / "leaf_encoder_components"
    require((first_leaf / "fit_0.json").is_file() and (first_leaf / "fit_1.json").is_file()
            and not (first_leaf / "sealed.json").exists(), "Unverified partial first attempt")
    counts["leaf_initial_completed_before_next_applicability_failure"] = 1
    counts["leaf_complete_comparison"] = records["leaf_components"]["classifier_fits"]
    counts["masked_field_pretraining"] = records["masked_components"]["self_supervised_fits"]
    counts["matched_masked_or_untrained_supervised_heads"] = records["masked_components"]["supervised_fits"]
    counts["aligned_out_of_fold_teachers"] = records["adaptive_teachers"]["new_classifier_fits"]
    counts["learned_metric_corrections"] = sum(read(dev / f"adaptmem_{n}/sealed.json")["supervised_metric_fits"]
                                              for n in ("anchor", "pair"))
    counts["downstream_tabdpt_frozen_contexts"] = sum(read(dev / f"tabdpt_{n}/sealed.json")["frozen_context_fits"]
                                                     for n in ("ordinal", "onehot"))
    counts["threshold_and_dual_replay_new_classifiers"] = 0
    require(not catalogue["official_test_opened"], "Unexpected formal test access")
    if catalogue["advance"]:
        paths["designated"] = out / "designated_result.json"
        records["designated"] = read(paths["designated"])
        counts["qualified_raw_entry"] = read(out / "release/result.json")["actual_fits"]
    prior = REGISTRY / "expanded_budget_results.json"
    from_previous = read(prior)["accumulated_task_and_router_fits"]
    value = dict(previous=sha(prior), previous_accumulated_fits=from_previous, results=records,
                 receipt_sha256={k:sha(p) for k,p in paths.items()}, new_fits=counts, stage_total=sum(counts.values()),
                 cumulative=from_previous + sum(counts.values()), separately=dict(
                     failed_leaf_applicability_checks=1, reused_residual_teacher_models=8,
                     synthetic_tabdpt_context_fits=1, token_api_requests=0, currency_rmb_new_api=0),
                 official_test_opened=False, designated_opened=bool(catalogue["advance"]), h2_opened=False)
    save(REGISTRY / "catalogue_encoder_results.json", value)
    print(counts, value["stage_total"], value["cumulative"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    export(parser.parse_args().out)
