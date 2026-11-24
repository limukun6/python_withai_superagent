"""Compare residual screening and observed incremental accuracy on identical archived feedback."""
import argparse
import math
from pathlib import Path

import numpy as np

from scripts.fit_portfolio_fold import PRIOR
from scripts.prepare_conditional_handoff import partition
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_encoder_fusion import NAMES, save_csv
from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import screen_vectors, ties
from tools.crossfit_challenger import nominate, residual_directions
from tools.portfolio_training import configure
from tools.preference_handoff import binary, changes

PLAN = ROOT / "docs/batches/challenger_nomination_replay_plan_20261124_zh.md"
HISTORY = ROOT / "research/unified_encoder_fusion_1124"


def context(archive):
    task = load_training_task(archive)
    g = sample_groups(task, "shared_text")
    return task, g, list(partition(g, np.asarray(read(EVIDENCE / "title_reserve_ids.json"))))


def probabilities(fold, roles):
    src = HISTORY / f"fold_{fold}"
    binding = read(HISTORY / "frozen.json")[fold]
    require(binding == read(src / "sealed.json") and sha(src / "all_probabilities.npz") == binding["probabilities"]
            and NAMES == read(src / "decision.json")["choice"]["members"], "Archived learners differ")
    row_spec = read(PRIOR / "protocol.json")["splits"][fold]["roles"]
    require(all(fingerprint(v.tolist()) == row_spec[k]["ids"] for k, v in roles.items()), "Archived split differs")
    with np.load(src / "all_probabilities.npz", allow_pickle=False) as stored:
        return {k: stored[k] for k in ("learn", "confirm", "outer")}


def votes(p):
    base = p[:, :2].mean(1)
    return {"baseline": base, **{name: (2 * base + p[:, j]) / 3 for j, name in enumerate(NAMES[2:], 2)}}


def feedback(p, rows, y):
    prediction = votes(p)
    tie = ties(rows)
    base = prediction["baseline"]
    return {name: {"net": changes(binary(base, tie), binary(prediction[name], tie), y)["net"],
                   "brier_gain": float(np.mean((base - y) ** 2 - (prediction[name] - y) ** 2))}
            for name in NAMES[2:]}


def decide(screen, learn, confirm, n):
    direct = min(learn, key=lambda name: (-learn[name]["net"], -learn[name]["brier_gain"], name))
    choices = {"proxy": nominate(screen), "observed": direct if learn[direct]["net"] > 0 else None}
    threshold = max(2, math.ceil(.002 * n))
    approved = {k: name if name and learn[name]["net"] > 0 and confirm[name]["net"] >= threshold
                and confirm[name]["brier_gain"] > 0 else None for k, name in choices.items()}
    return dict(nominations=choices, approved=approved, minimum_confirm=threshold, screen=screen,
                learn=learn, confirm=confirm)


def evaluate_rows(p, rows, choice):
    tie = ties(rows)
    c = {k: binary(v, tie) for k, v in votes(p).items()}
    for key in ("proxy", "observed"):
        c[key + "_forced"] = c[choice["nominations"][key] or "baseline"]
        c[key] = c[choice["approved"][key] or "baseline"]
    return c


def freeze(out, archive, weights=None, cache=None):
    out.mkdir(parents=True, exist_ok=True)
    task, groups, splits = context(archive)
    if weights is not None:
        configure(weights, cache)
    records = []
    with reuse_frozen_vectors():
        for i, roles in enumerate(splits):
            dest = out / f"fold_{i}"
            dest.mkdir(exist_ok=True)
            p = probabilities(i, roles)
            learn = roles["learn"]
            blocks = {k: [task.X_train[j] for j in ids] for k, ids in roles.items() if k != "base"}
            gradients = dest / "screen.npz"
            if weights is not None:
                residual = task.y_train[learn] - p["learn"][:, :2].mean(1)
                fresh = {n: residual_directions(screen_vectors(n, blocks["learn"]), residual, groups[learn]) for n in NAMES[2:]}
                if gradients.exists():
                    with np.load(gradients, allow_pickle=False) as stored:
                        require(all(np.allclose(stored[n], fresh[n], rtol=1e-6, atol=1e-9) for n in fresh),
                                "Stored residual directions differ")
                else:
                    np.savez_compressed(gradients, **fresh)
            with np.load(gradients, allow_pickle=False) as dep:
                require(set(dep.files) == set(NAMES[2:]), "Screen is missing a candidate")
                screen = {name: float(dep[name][0] @ dep[name][1]) for name in NAMES[2:]}
            labels = {k: task.y_train[roles[k]] for k in ("learn", "confirm")}
            m = {k: feedback(p[k], blocks[k], y) for k, y in labels.items()}
            choice = decide(screen, m["learn"], m["confirm"], len(labels["confirm"]))
            save(dest / "choice.json", choice)
            columns = evaluate_rows(p["outer"], blocks["outer"], choice)
            rows = dest / "rows.csv"
            if rows.exists():
                expected = {"row_id": roles["outer"], **columns}
                saved = load_export(rows)
                require(expected.keys() == saved.keys() and all(np.array_equal(saved[k], v) for k, v in expected.items()),
                        "Replayed decisions differ")
            else:
                save_csv(rows, roles["outer"], columns)
            records.append({k: sha(dest / k) for k in ("screen.npz", "choice.json", "rows.csv")})
    save(out / "frozen.json", dict(folds=records, plan=sha(PLAN), archive_seal=sha(HISTORY / "frozen.json"),
                                  new_classifier_fits=0, model_api_calls=0, supervised_nominations=10))


def score(out, archive):
    task, groups, splits = context(archive)
    sealed = read(out / "frozen.json")
    require(sealed["plan"] == sha(PLAN), "Replay plan changed")
    cols, truth, used_groups, folds = {}, [], [], []
    for i, roles in enumerate(splits):
        folder = out / f"fold_{i}"
        require(all(sha(folder / name) == v for name, v in sealed["folds"][i].items()), "Replay output changed")
        pred = load_export(folder / "rows.csv")
        ids = pred.pop("row_id")
        require(np.array_equal(ids, roles["outer"]), "Outer row IDs differ")
        y, g = task.y_train[ids], groups[ids]
        require(not set(g) & set(used_groups), "Duplicated outer groups")
        truth.extend(y)
        used_groups.extend(g)
        for k, v in pred.items():
            cols.setdefault(k, []).append(v)
        m = {k: changes(pred["baseline"], v, y) for k, v in pred.items()}
        folds.append(dict(fold=i, arms=m))
    y = np.asarray(truth)
    predictions = {k: np.concatenate(v) for k, v in cols.items()}
    arms = {k: comparison(predictions["baseline"], v, y, used_groups) for k, v in predictions.items()}
    require(len(y) == 6608 and arms["baseline"]["correct"] == read(HISTORY / "result.json")["pooled"]["methods"]["baseline"]["correct"],
            "Original archived baseline or coverage differs")
    direct = comparison(predictions["proxy"], predictions["observed"], y, used_groups)
    save(out / "result.json", dict(frozen=sha(out / "frozen.json"), rows=len(y), groups=len(set(used_groups)),
                                  folds=folds, arms=arms, observed_versus_proxy=direct))
    print({k: (v["correct"], v["net"]) for k, v in arms.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "score"))
    p.add_argument("out", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--cache", type=Path)
    args = p.parse_args()
    if args.stage == "freeze":
        freeze(args.out, args.archive, args.weights, args.cache)
    else:
        score(args.out, args.archive)
