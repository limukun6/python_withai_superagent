"""Localize disagreement and promotion errors after the five matched predictions are sealed."""
import argparse
from pathlib import Path

import numpy as np

from scripts.run_conditional_handoff import load_export
from scripts.study_crossfit_challenger import study_splits
from scripts.unified_released_study import ROOT, read, require, save, sha
from tools.challenger_training import ties
from tools.preference_handoff import binary, changes


PLAN = ROOT / "docs/batches/crossfit_challenger_diagnostics_20261124_zh.md"


def diagnose(out, archive):
    result = read(out / "result.json")
    require(sha(out / "sealed.json") == result["fold_seal"], "Score does not correspond to the frozen controls")
    task, _, splits = study_splits(archive)
    folds, totals = [], {}
    for i, (_, held) in enumerate(splits):
        folder = out / f"fold_{i}"
        decision = read(folder / "run/decision.json")
        screen = read(folder / "run/nomination.json")
        ranking = sorted(screen["scores"], key=lambda name: (-screen["scores"][name], name))
        record = {"fold": i, "nominee": screen["candidate"], "ranking": ranking, "screen": screen["scores"],
                  "chosen": decision["chosen"], "outer_rows": len(held)}
        folds.append(record)
        if screen["candidate"] is None:
            continue
        sealed = read(folder / "sealed.json")["records"]
        require(all(sha(folder / k) == v for k, v in sealed.items()), "Fold evidence changed")
        t = ties([task.X_train[j] for j in held])
        y = task.y_train[held]
        with np.load(folder / "outer_scores.npz", allow_pickle=False) as frozen:
            p = frozen["probabilities"]
        predictions = load_export(folder / "rows.csv")
        base = predictions["baseline"]
        third = binary(p[:, 2], t)
        agree = binary(p[:, 0], t) == binary(p[:, 1], t)
        conflict = base != third
        strata = {}
        for label, mask in (("agree", agree), ("disagree", ~agree)):
            active = mask & conflict
            counts = {"rows": int(mask.sum()), "third_conflicts": int(active.sum())}
            for arm, pred in (("third", third), ("forced", predictions["learned_weight"]), ("agent", predictions["agent"])):
                c = changes(base[active], pred[active], y[active])
                counts[arm + "_gained"] = c["gained"]
                counts[arm + "_lost"] = c["lost"]
            require(counts["third_gained"] + counts["third_lost"] == counts["third_conflicts"],
                    "Binary conflicts miscounted")
            strata[label] = counts
            summed = totals.setdefault(label, {})
            for k, v in counts.items():
                summed[k] = summed.get(k, 0) + v
        arms = result["folds"][i]["arms"]
        record.update(oof_rows=sum(e["rows"] for e in decision["evidence"].values()), oof_net=decision["net"],
                      weights=decision["weights"], outer_forced_net=arms["learned_weight"]["net"],
                      outer_delivered_net=arms["agent"]["net"], by_agreement=strata)
    report = {"plan": sha(PLAN), "source_result": sha(out / "result.json"), "folds": folds,
              "by_agreement": totals, "new_classifier_fits": 0, "model_api_calls": 0}
    save(out / "diagnostics.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    data = diagnose(args.out, args.archive)
    for fold in data["folds"]:
        print(fold["fold"], fold["nominee"], fold.get("oof_net"), fold.get("outer_forced_net"), fold["chosen"])
    print(data["by_agreement"])
