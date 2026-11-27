"""Collect the follow-up's locked choices, paired grades, fitting counts and public artifact hashes."""
import argparse
from pathlib import Path

from scripts.unified_released_study import ROOT, read, require, save, sha

PREVIOUS = ROOT / "research/task1_feedback_extension_1127/results.json"
SCREENS = {"rank": ("pairwise",), "rules": ("pairs", "triples"), "factor": ("fields", "recurrence"),
           "measure": ("measured", "relative"), "mitra": ("global", "local")}


def export(out, destination):
    previous = read(PREVIOUS)
    near = out / "near_full"
    matched = read(near / "result.json")
    contestants = {}
    fits = {"six_cold_entries": 0}
    for fold in matched["folds"]:
        for arm, method in fold["methods"].items():
            side = near / str(fold["fold"]) / arm
            launch = read(side / "launch.json")
            seal = read(side / "sealed.json")
            require(seal["launch"] == sha(side / "launch.json") and seal["independent_reload"]
                    and seal["fits"] == method["fits"] and not seal["model_api_requests"], "Unverified solve")
            contestants[f'{fold["fold"]}-{arm}'] = dict(source=launch["source"], cap=launch["cap"],
                 elapsed_seconds=round(seal["seconds"], 2), seal_sha256=sha(side / "sealed.json"), **method)
            fits["six_cold_entries"] += seal["fits"]
    dev = out / "development"
    screens = {}
    for family, methods in SCREENS.items():
        election = dev / (family + "_election")
        result = read(election / "results.json")
        locked = read(election / "locked.json")
        require(result["lock"] == sha(election / "locked.json"), "Nomination changed")
        details = {}
        count = 0
        for method in methods:
            path = dev / (family + "_" + method) / "sealed.json"
            evidence = read(path)
            require(evidence["protocol"] == locked["protocol"] and not evidence["api_requests"], "Mismatched experiment")
            details[method] = dict(sha256=sha(path), **evidence)
            if family == "rank":
                count += evidence["rank_fits"] + evidence["calibrator_fits"]
            elif family == "rules":
                count += evidence["model_fits"]
            elif family == "mitra":
                count += evidence["trainable_fits"]
            else:
                count += evidence["fits"]
        fits[family] = count
        screens[family] = dict(protocol_sha256=locked["protocol"], lock_sha256=sha(election / "locked.json"),
                result_sha256=sha(election / "results.json"), nomination=locked["alternatives"], checks=result["checks"],
                reference=locked["reference"], rows=locked["rows"], winner=result["winner"],
                confirmed_gain=result["gain"], eligible=result["eligible"], candidates=details)
    contexts = sum(len(record["contexts"]) for record in screens["mitra"]["candidates"].values())
    result = dict(previous_report_sha256=sha(PREVIOUS), previous_completed_fits=previous["completed_task_fits_total"],
                  completed_increment=fits, completed_increment_total=sum(fits.values()),
                  accumulated_completed_fits=previous["completed_task_fits_total"]+sum(fits.values()),
                  separate_frozen_mitra_contexts=contexts, mitra_weight_optimizer_steps=0,
                  frozen_pretrained_weights=read(ROOT/"research/task1_feedback_extension_1127/mitra_context.json")["provenance"],
                  external_folds_sha256=sha(near / "result.json"), external_solutions=contestants,
                  two_remaining_folds=matched, training_only_screens=screens,
                  designated_validation_opened=False, official_test_opened=False, h2_opened=False,
                  language_model_requests=0, input_tokens=0, output_tokens=0, course_api_cost_rmb=0)
    save(destination, result)
    print(fits, result["accumulated_completed_fits"], "plus fixed contexts", contexts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    parser.add_argument("--destination", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    export(args.out, args.destination)
