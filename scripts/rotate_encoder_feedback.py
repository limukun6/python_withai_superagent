"""Rotate learning and evaluation groups and test evidence pooling against the original selection rule."""
import argparse
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.fit_portfolio_fold import PRIOR
from scripts.run_conditional_handoff import load_export
from scripts.study_encoder_fusion import NAMES, measures, save_csv
from scripts.unified_released_study import ROOT, read, require, sha
from tools.feedback_policy_pool import evidence_pool, pooled_probability
from tools.preference_handoff import binary
from tools.preference_portfolio import choose, outputs, panel
from unified_agent import save


PLAN = ROOT/"docs/batches/unified_encoder_feedback_rotation_plan_20261124_zh.md"


def inputs_at(study, prior, i):
    part, old = study/f"fold_{i}", prior/f"fold_{i}"
    spec = read(PRIOR/"protocol.json")["splits"][i]
    require(sha(part/"all_probabilities.npz") == read(part/"sealed.json")["probabilities"]
            and sha(old/"features.joblib") == spec["features"], "Existing base evidence changed")
    return part, old, spec, joblib.load(old/"features.joblib")


def predict(study, prior, i, output):
    part, _, _, views = inputs_at(study, prior, i)
    rec = read(part/"rotation/decision.json")
    with np.load(part/"all_probabilities.npz", allow_pickle=False) as p:
        data = panel(p["learn"], views["learn"]["pairs"])
    choice, pool = rec["choice"], rec["pool"]
    cols = outputs(choice, data)
    cols["original_agent"] = cols["agent"]
    for key in ("proposed", "deployed"):
        prob = pooled_probability(pool[key+"_weights"], choice["rules"], data)
        cols[key] = binary(prob, data["ties"])
    cols["agent"] = cols["deployed"]
    save_csv(output, views["learn"]["ids"], cols)


def freeze(study, prior, i):
    part, old, spec, views = inputs_at(study, prior, i)
    folder = part/"rotation"
    folder.mkdir(exist_ok=True)
    first_labels = prior/"scoring_only"/f"fold_{i}.json"
    require(sha(first_labels) == spec["targets"] and sha(old/"feedback.npz") == spec["feedback"],
            "Rotated feedback belongs to other rows")
    identity = {"plan": sha(PLAN), "policy": sha(ROOT/"tools/feedback_policy_pool.py"),
                "parent": sha(part/"sealed.json"), "runner": sha(Path(__file__))}
    if not (folder/"decision.json").exists():
        source = read(first_labels)
        require(source["ids"] == views["outer"]["ids"], "Rotated learning labels have different row IDs")
        y0 = np.asarray(source["labels"])
        with np.load(old/"feedback.npz", allow_pickle=False) as ys:
            y1 = ys["confirm"]
        with np.load(part/"all_probabilities.npz", allow_pickle=False) as ps:
            learning = panel(ps["outer"], views["outer"]["pairs"])
            confirm = panel(ps["confirm"], views["confirm"]["pairs"])
        save(folder/"admission.json", {"continuous": 2, "shortlists": 2, "pooling": 1, "previous_heads": 6})
        choice = choose(NAMES, learning, y0, confirm, y1)
        pool = evidence_pool(choice, learning, y0, views["outer"]["groups"], confirm, y1, views["confirm"]["groups"])
        save(folder/"decision.json", {"identity": identity, "choice": choice, "pool": pool})
    require(read(folder/"decision.json")["identity"] == identity, "Saved rotated decision belongs to different evidence")
    hashes = []
    for j in range(2):
        destination = folder/f"rows_{j}.csv"
        if not destination.exists():
            subprocess.run([sys.executable,"-m","scripts.rotate_encoder_feedback","predict","--prior",str(prior),
                "--study",str(study),"--fold",str(i),"--output",str(destination)], cwd=ROOT, check=True)
        hashes.append(sha(destination))
    require(hashes[0] == hashes[1], "Rotated cross-process prediction differs")
    seal = {"selection": sha(folder/"decision.json"), "rows": hashes, "outer_row_ids": spec["roles"]["learn"]["ids"]}
    if (folder/"sealed.json").exists():
        require(read(folder/"sealed.json") == seal, "Changed sealed decision")
    else:
        save(folder/"sealed.json", seal)


def score(study, prior):
    seals = [read(study/f"fold_{i}/rotation/sealed.json") for i in range(5)]
    if not (study/"rotation_frozen.json").exists():
        save(study/"rotation_frozen.json", seals)
    require(read(study/"rotation_frozen.json") == seals, "Reopened rotated evaluation")
    by_fold, combined, ys, gs = [], {}, [], []
    for i, seal in enumerate(seals):
        part, old, spec, views = inputs_at(study, prior, i)
        directory, target = part/"rotation", old/"feedback.npz"
        require(sha(directory/"rows_0.csv") == sha(directory/"rows_1.csv") == seal["rows"][0]
                and sha(directory/"decision.json") == seal["selection"] and sha(target) == spec["feedback"],
                "Unsealed rotation")
        with np.load(target, allow_pickle=False) as labels:
            y = labels["learn"]
        cols = load_export(directory/"rows_0.csv")
        outer = views["learn"]
        require(cols["row_id"].tolist() == outer["ids"] and not set(outer["groups"]).intersection(gs),
                "Rotated evaluation groups repeat")
        rec = read(directory/"decision.json")
        by_fold.append({"fold": i, "original_choice": rec["choice"]["chosen"], "pool_adopted": rec["pool"]["adopted"],
                        **measures(cols, y, outer["groups"])})
        ys.extend(y)
        gs.extend(outer["groups"])
        for n, v in cols.items():
            combined.setdefault(n, []).append(v)
    cols = {n: np.concatenate(v) for n,v in combined.items()}
    require(len(set(cols["row_id"])) == len(ys) == 6608, "Rotated five-fold coverage differs")
    pooled = measures(cols, np.array(ys), gs)
    wins = sum(f["methods"]["agent"]["net"] > 0 for f in by_fold)
    a = pooled["methods"]["agent"]
    result = {"folds": by_fold, "pooled": pooled, "positive_folds": wins,
              "passes": bool(wins>=4 and a["net"]>=5 and a["paired_95_fraction"][0]>0), "continuous_fits": 10,
              "discrete_selections": 10, "evidence_pools": 5, "added_base_fits": 0, "model_api_calls": 0}
    if (study/"rotation_result.json").exists():
        require(read(study/"rotation_result.json") == result, "Recomputed results differ")
    else:
        save(study/"rotation_result.json", result)
    print("rotation", {k: (v["correct"], v["net"]) for k,v in pooled["methods"].items()
                       if k in ("baseline", "contextual", "constant", "original_agent", "agent")}, wins, result["passes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze","predict","score"))
    p.add_argument("--prior",required=True,type=lambda x:Path(x).resolve())
    p.add_argument("--study",required=True,type=lambda x:Path(x).resolve())
    p.add_argument("--fold",type=int,choices=range(5))
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.stage == "predict":
        predict(a.study,a.prior,a.fold,a.output)
    elif a.stage == "freeze":
        for i in (range(5) if a.fold is None else [a.fold]):
            freeze(a.study,a.prior,i)
    else:
        score(a.study,a.prior)
