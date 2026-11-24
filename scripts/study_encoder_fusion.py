"""Freeze decisions and cross-process native exports for the registered six-expert comparison."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np

from scripts.fit_portfolio_fold import PLAN, PRIOR, ROLES
from scripts.run_conditional_handoff import comparison, load_export
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.prediction_evidence import fingerprint
from tools.encoder_assets import MODELS
from tools.portfolio_training import REFERENCE, configure, export
from tools.preference_handoff import binary
from tools.preference_portfolio import choose, outputs, panel, weights
from unified_agent import pair_packet, save


NAMES = [*REFERENCE, *MODELS]
SPEC = {"input_schema": {"modality": "pair_text", "columns": [{"name": "first", "type": "text"},
        {"name": "second", "type": "text"}], "target": "preference", "pair_label_one": "left", "pair_semantics": "preference"}}


def observed(directory, name, digest):
    with (directory/name).open("rb") as stream:
        require(sha(directory/name) == digest, "Frozen probability array changed")
        with np.load(stream, allow_pickle=False) as source:
            return {r: source[r] for r in ROLES}


def source_panels(prior, out, index):
    spec = read(PRIOR/"protocol.json")
    require(sha(prior/"protocol.json") == sha(PRIOR/"protocol.json") and spec["members"] == list(REFERENCE),
            "Reference portfolio or grouped protocol differs")
    basis, source, folder = spec["splits"][index], prior/f"fold_{index}", out/f"fold_{index}"
    require(sha(source/"features.joblib") == basis["features"], "Feedback or evaluation features changed")
    panels = joblib.load(source/"features.joblib")
    for role in ROLES:
        require(fingerprint(panels[role]["ids"]) == basis["roles"][role]["ids"], "Role membership differs")
    ref = read(PRIOR/f"fold_{index}/scores_frozen.json")
    cols = [observed(source, "probabilities.npz", ref["scores"])]
    receipts = []
    for name in MODELS:
        dest = folder/name
        receipt = read(dest/"scores.json")
        require(receipt["name"] == name and receipt["identity"]["plan"] == sha(PLAN)
                and receipt["identity"]["base"] == basis["payload"] and receipt["identity"]["features"] == basis["features"]
                and sha(dest/"fits.json") == receipt["ledger"] and read(dest/"fits.json")["completed_classifier_fits"] == 1,
                "New expert fit or row identity differs")
        cols.append(observed(dest, "probabilities.npz", receipt["scores"]))
        receipts.append(receipt)
    values = {r: np.column_stack([v[r] for v in cols]) for r in ROLES}
    return source, folder, basis, panels, values, receipts


def save_csv(path, ids, columns):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", *columns])
        for i, row in enumerate(ids):
            writer.writerow([int(row), *(int(p[i]) for p in columns.values())])


def predict_scores(prior, out, index, destination):
    _, folder, _, panels, values, _ = source_panels(prior, out, index)
    decision = read(folder/"decision.json")["choice"]
    current = panel(values["outer"], panels["outer"]["pairs"])
    save_csv(destination, panels["outer"]["ids"], outputs(decision, current))


def freeze(prior, out, index, weight_root, cache):
    source, folder, basis, panels, values, receipts = source_panels(prior, out, index)
    identity = {"reference": sha(PRIOR/f"fold_{index}/scores_frozen.json"),
                "new": [r["scores"] for r in receipts], "feature_rows": basis["features"],
                "feedback": basis["feedback"], "policy_source": sha(ROOT/"tools/preference_portfolio.py"), "plan": sha(PLAN)}
    require(sha(source/"feedback.npz") == basis["feedback"], "Learning/confirmation feedback differs")
    if (folder/"decision.json").exists():
        record = read(folder/"decision.json")
        require(identity == record["input"], "Saved choice belongs to different inputs")
        decision = record["choice"]
    else:
        with np.load(source/"feedback.npz", allow_pickle=False) as labels:
            first = panel(values["learn"], panels["learn"]["pairs"])
            check = panel(values["confirm"], panels["confirm"]["pairs"])
            save(folder/"routing_admission.json", {"new_head_fits": 3, "previous_head_fits": 3, "continuous_fits": 2,
                 "supervised_shortlists": 2, "maximum_learning_operations": 10})
            decision = choose(NAMES, first, labels["learn"], check, labels["confirm"])
        save(folder/"decision.json", {"input": identity, "choice": decision})
    if not (folder/"all_probabilities.npz").exists():
        with (folder/"all_probabilities.npz").open("xb") as stream:
            np.savez(stream, **values)
    with np.load(folder/"all_probabilities.npz", allow_pickle=False) as packed:
        require(all(np.array_equal(packed[r], values[r]) for r in ROLES), "Serialized evidence differs")
    if not (folder/"agent/final_selection.json").exists():
        ref_models = read(PRIOR/f"fold_{index}/models_frozen.json")["models"]
        models = []
        for i, m in enumerate(ref_models):
            path = source/f"member_{i}.joblib"
            require(sha(path) == m["model"] and m["name"] == REFERENCE[i], "Reference model changed")
            models.append(joblib.load(path)["model"])
        for r in receipts:
            path = folder/r["name"]/"model.joblib"
            require(sha(path) == r["model"], "New expert artifact changed")
            models.append(joblib.load(path))
        export(folder/"agent", models, decision, SPEC)
    features = folder/"outer_features.zip"
    if not features.exists():
        rows = panels["outer"]["pairs"]
        pair_packet(features, SimpleNamespace(spec=SPEC, X_train=rows), range(len(rows)), training=False)
    raw = folder/"rows.csv"
    if not raw.exists():
        subprocess.run([sys.executable, "-m", "scripts.study_encoder_fusion", "predict", "--prior", str(prior),
                        "--out", str(out), "--fold", str(index), "--output", str(raw)], cwd=ROOT, check=True)
    configure(weight_root, cache)
    native = folder/"delivery/predictions.csv"
    subprocess.run([sys.executable, "-m", "unified_agent", "predict", str(features), "--run-dir", str(folder),
                    "--output", str(native)], cwd=ROOT, check=True)
    expected, real = load_export(raw), load_export(native)
    require(expected["row_id"].tolist() == panels["outer"]["ids"]
            and real["row_id"].tolist() == list(range(len(expected["row_id"])))
            and np.array_equal(expected["agent"], real["prediction"]), "Independent raw-input delivery differs")
    code = sha(Path(__file__))
    seal = {"predictions": sha(raw), "native": sha(native), "features": sha(features),
            "model": read(folder/"agent/final_selection.json")["model_sha256"], "selection": sha(folder/"decision.json"),
            "probabilities": sha(folder/"all_probabilities.npz"), "runner": code,
            "outer_rows": len(real["prediction"]), "new_head_fits": len(receipts), "model_api_calls": 0}
    if (folder/"sealed.json").exists():
        require(seal == read(folder/"sealed.json"), "Repeated export differs")
    else:
        save(folder/"sealed.json", seal)
    print("sealed", index, decision["chosen"], seal["outer_rows"], "raw predictions", flush=True)


def measures(columns, y, groups):
    base = columns["baseline"]
    both_wrong = (columns[REFERENCE[0]] != y) & (columns[REFERENCE[1]] != y)
    result = {}
    for name, col in columns.items():
        if name != "row_id":
            result[name] = {**comparison(base, col, y, groups), "both_wrong_recovered": int((both_wrong & (col==y)).sum())}
    new_matrix = np.asarray([columns[n] for n in MODELS])
    return {"methods": result, "groups": len(set(groups)), "rows": len(y), "baseline_members_both_wrong": int(both_wrong.sum()),
            "any_new_correct_on_shared_error": int((both_wrong & (new_matrix == y).any(axis=0)).sum())}


def summarize(prior, out):
    spec = read(PRIOR/"protocol.json")
    seals = [read(out/f"fold_{i}/sealed.json") for i in range(5)]
    if not (out/"frozen.json").exists():
        save(out/"frozen.json", seals)
    require(read(out/"frozen.json") == seals, "Changed predictions after evaluation freeze")
    details, combined, all_y, all_groups = [], {}, [], []
    for i, seal in enumerate(seals):
        local, source = out/f"fold_{i}", prior/"scoring_only"/f"fold_{i}.json"
        require(sha(local/"rows.csv") == seal["predictions"] and sha(local/"decision.json") == seal["selection"]
                and sha(local/"all_probabilities.npz") == seal["probabilities"]
                and sha(source) == spec["splits"][i]["targets"], "Unmatched outer targets or selection")
        truth = read(source)
        y, groups = np.array(truth["labels"]), truth["groups"]
        cols = load_export(local/"rows.csv")
        require(cols["row_id"].tolist() == truth["ids"] and not set(groups).intersection(all_groups),
                "One headline group crosses outer folds")
        result = measures(cols, y, groups)
        choice = read(local/"decision.json")["choice"]
        info = joblib.load(prior/f"fold_{i}/features.joblib")["outer"]
        with np.load(local/"all_probabilities.npz", allow_pickle=False) as packed:
            inputs = panel(packed["outer"], info["pairs"])
        selected_rule = choice["rules"][choice["chosen"]]
        w = weights(selected_rule, inputs)
        details.append({"fold": i, "chosen": choice["chosen"], "members": NAMES,
                "mean_actual_weight": w.mean(0).tolist(), "std_actual_weight": w.std(0).tolist(), **result})
        all_y.extend(y)
        all_groups.extend(groups)
        for name, col in cols.items():
            combined.setdefault(name, []).append(col)
    cols = {name: np.concatenate(values) for name, values in combined.items()}
    require(len(cols["row_id"]) == len(set(cols["row_id"])) == spec["outer_rows"], "Lost or repeated outer rows")
    pooled = measures(cols, np.array(all_y), all_groups)
    baseline = read(PRIOR/"five_fold_results.json")["pooled"]["methods"]["baseline"]["correct"]
    require(pooled["methods"]["baseline"]["correct"] == baseline, "Previous paired baseline differs")
    agent = pooled["methods"]["agent"]
    wins = sum(f["methods"]["agent"]["net"] > 0 for f in details)
    result = {"folds": details, "pooled": pooled, "positive_folds": wins,
              "passes": bool(wins >= 4 and agent["net"] >= 5 and agent["paired_95_fraction"][0] > 0),
              "new_classifier_fits": 15, "new_continuous_fits": 10, "new_discrete_selections": 10,
              "previous_classifier_fits_reused": 15, "model_api_calls": 0, "input_tokens_billed": 0,
              "output_tokens_billed": 0, "model_api_rmb": 0, "course_validation_used": False, "formal_test_used": False}
    if (out/"result.json").exists():
        require(result == read(out/"result.json"), "Repeated result differs")
    else:
        save(out/"result.json", result)
    print({k: (v["correct"], v["gained"], v["lost"]) for k,v in pooled["methods"].items()}, flush=True)
    print("Agent", [f["chosen"] for f in details], wins, "positive folds; promoted", result["passes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "predict", "score"))
    for key in ("prior", "out", "cache", "weights", "output"):
        p.add_argument("--"+key, type=lambda s: Path(s).resolve(), required=key in ("prior", "out"))
    p.add_argument("--fold", type=int, choices=range(5))
    a = p.parse_args()
    if a.stage == "predict":
        predict_scores(a.prior, a.out, a.fold, a.output)
    elif a.stage == "freeze":
        for n in (range(5) if a.fold is None else (a.fold,)):
            freeze(a.prior, a.out, n, a.weights, a.cache)
    else:
        summarize(a.prior, a.out)
