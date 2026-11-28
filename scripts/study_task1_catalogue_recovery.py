"""Prepare a fresh, matched resource trial using hash-verified copies of the original packets."""
import argparse
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_conditional_opportunity import ordered
from scripts.study_task1_extensions import key
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_seed_completion import origin
from scripts.study_task1_unseen_profiles import FOLDS, STUDY
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.table_foundation import FrozenTableClassifier, NAME, SHA256, checkpoint
from superagent.joint_context import tokens
from superagent.models import _clean_categories
from superagent.profile_history import discover_columns
from superagent.splits import sample_groups

STAGE = "matched_catalogue_1128"
PLAN = ROOT / "research/task1_feedback_extension_1127/matched_catalogue_recovery.json"


def copy_verified(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    require(not dst.exists(), "New comparison needs unused paths")
    shutil.copyfile(src, dst)
    require(sha(src) == sha(dst), "Comparison input differs")


def prepare(out):
    root = out / STAGE
    root.mkdir(exist_ok=False)
    copy_verified(out / "release/inputs.json", root / "release/inputs.json")
    copy_verified(out / STUDY / "prepared.json", root / STUDY / "prepared.json")
    source = {}
    for i in FOLDS:
        fold = out / STUDY / str(i)
        side = fold / "replicated_serial"
        h = side / "run/agent/history.json"
        unavailable = [t["trial"] for t in read(h) if t["candidate"] == NAME and t["status"] == "failed"
                       and "invalid low watermark ratio" in t.get("error","")]
        require(unavailable and (side / "sealed.json").is_file(), "Impaired-resource arm was not identified")
        for name in ("inputs.json","anonymous/training.zip","anonymous/queries.zip","outer_answers.joblib"):
            copy_verified(fold / name, root / STUDY / str(i) / name)
        source[str(i)] = dict(failed_optional_trials=unavailable, history=sha(h), impaired=sha(side / "sealed.json"),
                              inputs=sha(fold / "inputs.json"))
    # The known-profile audit only needs features from these ordinary-fold packets.
    for i in range(5):
        for name in ("training.zip","queries.zip"):
            src = origin(out, i) / "anonymous" / name
            copy_verified(src, root / src.relative_to(out))
    save(root / "registered.json", dict(protocol=sha(PLAN), originals=source))


def probe(out):
    root = out / STAGE
    require(all(k not in os.environ for k in ("PYTORCH_MPS_HIGH_WATERMARK_RATIO","PYTORCH_MPS_LOW_WATERMARK_RATIO")),
            "The comparison requires original allocator settings")
    import torch
    from superagent.table_foundation import device
    hardware = device()
    require(checkpoint() is not None and sha(checkpoint()) == SHA256, "Missing verified optional weights")
    allocation_probe = torch.ones(2, device=hardware).sum().item()
    require(allocation_probe == 2., "Accelerator cannot compute")
    fold = origin(out, 3)
    side = fold / "replicated"
    rc = read(side / "sealed.json")
    j = rc["members"].index(NAME)
    path = side / f"run/agent/refit_component_{j}.joblib"
    m = joblib.load(path)["model"]
    require(isinstance(m, FrozenTableClassifier), "Expected a fitted frozen table reference")
    x, = arrays(fold / "anonymous/queries.zip", ["C_test.npy"])
    p = ordered(m, x[:6], m.classes_)
    require(len(p) == 6 and np.isfinite(p).all(), "Frozen-reference inference failed")
    save(root / "resource_probe.json", dict(protocol=sha(PLAN), registered=sha(root / "registered.json"), device=hardware,
         checkpoint=SHA256, historical_refit=sha(path), finite_rows=len(p), classifier_fits=0, model_api_requests=0))
    print("full-resource inference", hardware, len(p), flush=True)


def audit(out):
    root = out / STAGE
    probe = read(root / "resource_probe.json")
    require(probe["registered"] == sha(root / "registered.json"), "Incorrect environment probe")
    results = []
    for i in FOLDS:
        fold = root / STUDY / str(i)
        side = fold / "replicated"
        a = side / "run/agent"
        hist = read(a / "history.json")
        trials = [dict(trial=t["trial"], status=t["status"]) for t in hist if t["candidate"] == NAME]
        for t in hist:
            require("invalid low watermark ratio" not in str(t.get("error", "")), "Invalid resource configuration")
        old = out / STUDY / str(i) / "replicated_serial"
        native = read(a / "run_outcome.json")
        launch = read(side / "launch.json")
        require(read(old / "launch.json") == launch and launch["optional_resources"].get("tabicl_v2") == probe["checkpoint"],
                "Resource or catalogue differs from declared full solve")
        if trials:
            require(any(t["status"] == "completed" for t in trials), "The nominated optional method never finished")
        results.append(dict(fold=i, ordinary=sha(side / "sealed.json"), optional_trials=trials,
                            attempted_fits=native["attempted_classifier_fits"], completed_fits=native["completed_classifier_fits"],
                            failure_count=sum(t["status"] == "failed" for t in hist)))
    save(root / "resource_outcomes.json", dict(protocol=sha(PLAN), probe=sha(root / "resource_probe.json"), folds=results))
    print("full-resource audit", results, flush=True)


def same_rows(out):
    """Descriptively compare assignments on the very same source rows across the two training designs."""
    root = out / STAGE
    full_result = root / STUDY / "result.json"
    require(full_result.is_file() and (root / "resource_outcomes.json").is_file(), "Score completed full-resource solves first")
    train = out / "remaining_folds_source/anonymous/training.zip"
    source = sha(train)
    x,y = arrays(train, ["C_train.npy","y_train.npy"])
    class_rows = y.astype(str)
    x_clean = _clean_categories(x)
    profile = np.asarray(tokens(x_clean[:,discover_columns(x_clean)]))
    exact = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=x))
    seed = read(origin(out,0) / "inputs.json")["seed"]
    old_hits = np.zeros(len(y), dtype=bool)
    assigned = np.zeros(len(y), dtype=bool)
    for i,(fit,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(x,y,exact)):
        f = origin(out,i)
        info = read(f / "inputs.json")
        require(info["source"] == source and key(held) == info["roles"]["external"]["digest"]
                and not assigned[held].any() and set(profile[held]).issubset(profile[fit]), "Unmatched original evaluation rows")
        pred = f / "replicated/solve.csv"
        require(sha(pred) == read(f/"replicated/sealed.json")["prediction"], "Original predictions changed")
        old_hits[held] = prediction_csv(pred,len(held)) == class_rows[held]
        assigned[held] = True
    require(assigned.all() and int(old_hits.sum()) == read(out/"seed_completion/result.json")["correct"]["replicated"],
            "Original out-of-fold predictions do not cover the training source")
    prior, fresh, groups, per_fold = [], [], [], []
    for i, (fit,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=481936).split(x,y,profile)):
        if i not in FOLDS:
            continue
        folder = root / STUDY / str(i)
        info = read(folder / "inputs.json")
        require(info["source"] == source and key(held) == info["roles"]["external"]["digest"]
                and set(profile[held]).isdisjoint(profile[fit]), "Unmatched novel-profile rows")
        pred = folder / "replicated/solve.csv"
        require(sha(pred) == read(folder/"replicated/sealed.json")["prediction"], "Stress predictions changed")
        baseline = old_hits[held]
        cold = prediction_csv(pred,len(held)) == class_rows[held]
        prior.extend(baseline)
        fresh.extend(cold)
        groups.extend(profile[held])
        per_fold.append(dict(fold=i, rows=len(held), known=int(baseline.sum()), novel=int(cold.sum())))
    paired = compare(np.asarray(fresh), np.asarray(prior), groups)
    result = dict(rows=len(groups), profiles=len(set(groups)), folds=per_fold,
                  correct=dict(known=int(np.sum(prior)), novel=int(np.sum(fresh))), paired=paired,
                  source=source, ordinary_fivefold=sha(out / "seed_completion/result.json"), stress=sha(full_result),
                  new_fits=0, model_api_requests=0, official_test_opened=False, designated_validation_opened=False, h2_opened=False)
    require(result["correct"]["novel"] == read(full_result)["correct"]["frozen"], "Stress score differs")
    save(root / "same_rows.json", result)
    print("matched rows", result["correct"], paired, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","probe","audit","same-rows"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    {"prepare":prepare, "probe":probe, "audit":audit, "same-rows":same_rows}[args.action](args.out)
