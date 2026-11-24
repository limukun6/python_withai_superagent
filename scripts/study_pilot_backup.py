"""Run a pre-registered backup intervention and its control through the same native CLI."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.evaluate_pilot_nomination import project, store_arrays
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_crossfit_challenger import native_predict, study_splits
from scripts.study_encoder_fusion import save_csv
from scripts.study_pilot_nomination import current_components, link_file, run_arm
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_prediction_features
from superagent.prediction_evidence import fingerprint
from tools.challenger_training import score_members
from tools.pilot_challenger import BACKUP_STRATEGIES, STRATEGIES
from tools.portfolio_training import configure
from tools.preference_handoff import changes

PLAN = ROOT / "docs/batches/pilot_backup_plan_20261124_zh.md"


def parent_components(folder):
    sources = [p for s in STRATEGIES for p in current_components(folder / s)]
    for receipt in sorted((folder / "shadow").glob("*/trained.json")):
        r = read(receipt)
        require(r["model"] == read(folder / "sealed.json")["refit_records"][r["name"]]["model"]
                and sha(receipt.parent / "model.joblib") == r["model"], "Unbound historical comparison fit")
        sources.append(receipt.parent)
    return sources


def run_pair(folder, parent, weights, features):
    for f in ("train.zip", "features.zip"):
        link_file(parent / f, folder / f)
    sources = parent_components(parent)
    spent = 0
    for strategy in BACKUP_STRATEGIES:
        perf = run_arm(folder, strategy, sources, weights, features)
        spent += perf["new_classifier_fits"]
        sources = current_components(folder / strategy) + sources
    require(spent <= 1, "Backup pair exceeded its additional fit allowance")
    stages = [read(folder / s / "run/first_check.json") for s in BACKUP_STRATEGIES]
    require(all(stages[0][k] == stages[1][k] for k in ("first", "backup", "first_check")), "Different first-stage inputs")
    print(folder.name, "new classifier fits", spent, "finalists", [row["advanced"] for row in stages], flush=True)


def smoke(out, parent, weights, features):
    src = read(parent / "smoke.json")
    require(src["train"] == sha(parent / "train.zip") and src["features"] == sha(parent / "features.zip"), "Smoke packet changed")
    run_pair(out, parent, weights, features)
    results = {}
    for s in BACKUP_STRATEGIES:
        arm = out / s
        cached = native_predict(arm, weights, features)
        direct = native_predict(arm, weights)
        require(len(direct) == 24 and np.array_equal(cached, direct), "Anonymous cross-process prediction failed")
        end = read(arm / "run/pilot_completion.json")
        results[s] = {"new": end["new_classifier_fits"], "cold": end["cold_classifier_fits"],
                      "model": end["model_sha256"], "decision": end["decision_sha256"],
                      "predictions": sha(arm / "direct/predictions.csv")}
    save(out / "smoke.json", dict(parent=sha(parent/"smoke.json"), results=results, rows=24, model_api_calls=0))


def train(out, parent, smoke_path, weights, features):
    source = read(parent / "protocol.json")
    frozen = read(parent / "frozen.json")
    require(frozen["protocol"] == sha(parent/"protocol.json") and read(smoke_path/"smoke.json")["rows"] == 24,
            "Source evidence or small-task check missing")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "protocol.json", dict(plan=sha(PLAN), parent=sha(parent/"frozen.json"), smoke=sha(smoke_path/"smoke.json"),
                                    maximum_new_fits=5, cold_limit_per_strategy=9, model_api_limit=0))
    for i in range(5):
        src, dst = parent / f"fold_{i}", out / f"fold_{i}"
        binding = source["folds"][str(i)]
        require(sha(src/"sealed.json") == frozen["folds"][i] and sha(src/"train.zip") == binding["train_packet"]
                and sha(src/"features.zip") == binding["features_packet"], "Different parent packet")
        run_pair(dst, src, weights, features)


def freeze_fold(folder, src, weights, features):
    parent_seal = read(src/"sealed.json")
    require(all(sha(src/k) == v for k,v in parent_seal["records"].items()), "Original prediction seal differs")
    scores = {}
    with np.load(src / "outer.npz", allow_pickle=False) as p:
        for i,n in enumerate(parent_seal["refit_members"]):
            scores[n] = (p["refit"][:,i], parent_seal["refit_records"][n]["model"])
    first_run = folder / BACKUP_STRATEGIES[0] / "run"
    spec = read(first_run / "pilot_protocol.json")
    rows = load_prediction_features(folder/"features.zip", "pair_text", read(first_run/"agent/task_spec.json"))
    names = list(spec["inventory"]["candidates"][:2])
    decisions, record, ids, costs = {}, {}, {}, {}
    configure(weights, features)
    with reuse_frozen_vectors():
        for s in BACKUP_STRATEGIES:
            arm = folder / s
            run = arm / "run"
            dec, end = read(run/"decision.json"), read(run/"pilot_completion.json")
            require(end["decision_sha256"] == sha(run/"decision.json") and end["model_sha256"] == sha(run/"agent/selected_model.joblib"),
                    "Unbound backup decision or export")
            export = native_predict(arm, weights)
            decisions[s] = (dec["approved"], export)
            costs[s] = {"new":end["new_classifier_fits"], "reused":end["reused_classifier_fits"],
                        "cold":end["cold_classifier_fits"], "train_seconds":end["elapsed_seconds"],
                        "native_seconds":read(arm/"direct/elapsed.json")["seconds"]}
            for f in ("run/pilot_protocol.json","run/nomination.json","run/first_check.json","run/decision.json",
                      "run/pilot_completion.json","run/agent/final_selection.json","run/feedback.npz","direct/predictions.csv"):
                record[f"{s}/{f}"] = sha(arm/f)
            for action in end["actions"]:
                if action["stage"] != "refit":
                    continue
                n, model = action["candidate"], action["model"]
                require(ids.setdefault(n, model) == model, "Alternate paths used different refits")
                if n not in names:
                    names.append(n)
                if n not in scores or scores[n][1] != model:
                    path = run / "refit" / n / "model.joblib"
                    require(sha(path) == model, "Changed independent comparison model")
                    scores[n] = (score_members([joblib.load(path)], rows)[:,0], model)
    p = np.column_stack([scores[n][0] for n in names])
    votes = project(names, p, rows)
    previous = load_export(src/"rows.csv")
    output = {"baseline":votes["baseline"], "pilot-accuracy":previous["pilot-accuracy_approved"],
              "pilot-brier":previous["pilot-brier_approved"]}
    require(np.array_equal(previous["baseline"], output["baseline"]), "Parent and backup reference differ")
    for s,(n,pred) in decisions.items():
        require(np.array_equal(votes[n or "baseline"], pred), "Fresh backup export differs from expected vote")
        output[s] = pred
    store_arrays(folder / "refit.npz", {"scores":p})
    rowfile = folder / "rows.csv"
    if rowfile.exists():
        old = load_export(rowfile)
        expected = {"row_id":np.arange(len(rows)), **output}
        require(old.keys()==expected.keys() and all(np.array_equal(old[k],v) for k,v in expected.items()), "Rows changed")
    else:
        save_csv(rowfile, range(len(rows)), output)
    record.update({f:sha(folder/f) for f in ("rows.csv","refit.npz")})
    save(folder / "sealed.json", dict(parent=sha(src/"sealed.json"), records=record, refit_members=names,
                                      model_hashes=ids, costs=costs))
    print(folder.name, "raw outputs", len(rows), flush=True)


def freeze(out, parent, weights, features):
    require(read(out/"protocol.json")["parent"] == sha(parent/"frozen.json"), "Changed parent study")
    seals = []
    for i in range(5):
        folder = out / f"fold_{i}"
        if not (folder/"sealed.json").exists():
            freeze_fold(folder, parent/folder.name, weights, features)
        require(all(sha(folder/k)==v for k,v in read(folder/"sealed.json")["records"].items()), "Backup export changed")
        seals.append(sha(folder/"sealed.json"))
    save(out/"frozen.json", dict(protocol=sha(out/"protocol.json"), folds=seals))


def score(out, parent, archive):
    frozen = read(out/"frozen.json")
    require(frozen["protocol"] == sha(out/"protocol.json") and read(out/"protocol.json")["plan"] == sha(PLAN),
            "Changed backup protocol")
    primary = read(parent/"protocol.json")
    task, groups, splits = study_splits(archive)
    collected, test_ids, seen, folds = {}, [], set(), []
    off, on = BACKUP_STRATEGIES
    for i,(_,ids) in enumerate(splits):
        folder = out / f"fold_{i}"
        seal = read(folder/"sealed.json")
        require(frozen["folds"][i] == sha(folder/"sealed.json") and seal["parent"] == sha(parent/folder.name/"sealed.json")
                and all(sha(folder/k)==v for k,v in seal["records"].items()), "Unsealed backup fold")
        require(primary["folds"][str(i)]["outer"]["ids"] == fingerprint(ids.tolist()) and
                not seen.intersection(groups[ids]), "Different or overlapping groups")
        test_ids.extend(ids)
        seen.update(groups[ids])
        p = load_export(folder/"rows.csv")
        require(np.array_equal(p.pop("row_id"), np.arange(len(ids))), "Native row order differs")
        y = task.y_train[ids]
        metrics = {k:changes(p["baseline"],v,y) for k,v in p.items()}
        info = {}
        for s in BACKUP_STRATEGIES:
            r = folder/s/"run"
            stage, d = read(r/"first_check.json"), read(r/"decision.json")
            info[s] = dict(primary=stage["first"], backup=stage["backup"], advanced=stage["advanced"],
                           first_net=stage["first_check"]["net"] if stage["first_check"] else None,
                           final_net=d["confirm"]["net"] if d["confirm"] else None, approved=d["approved"])
        folds.append(dict(fold=i, rows=len(ids), costs=seal["costs"], choices=info, arms=metrics,
                          backup_effect=changes(p[off], p[on], y)))
        for k,v in p.items():
            collected.setdefault(k,[]).append(v)
    ids = np.asarray(test_ids)
    require(len(ids) == len(set(ids)) == 6608 and len(seen) == 4874, "Incorrect fold coverage")
    full = {k:np.concatenate(v) for k,v in collected.items()}
    y,g = task.y_train[ids],groups[ids]
    arms = {k:comparison(full["baseline"],v,y,g) for k,v in full.items()}
    paired = comparison(full[off],full[on],y,g)
    new = sum(c["new"] for f in folds for c in f["costs"].values())
    require(new <= 5, "Unexpected backup intervention fits")
    save(out/"result.json", dict(frozen=sha(out/"frozen.json"),rows=len(ids),groups=len(seen),folds=folds,
          arms=arms, backup_effect=paired, actual_new_classifier_fits=new, model_api_calls=0, model_api_rmb=0))
    print({k:(v["correct"],v["net"]) for k,v in arms.items()}, "backup versus control", paired["net"], "fits", new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke","train","freeze","score"))
    for arg in ("out","parent","smoke","weights","cache","archive"):
        parser.add_argument("--"+arg, required=arg in ("out","parent"), type=lambda s:Path(s).resolve())
    a = parser.parse_args()
    if a.stage == "smoke":
        smoke(a.out,a.parent,a.weights,a.cache)
    elif a.stage == "train":
        train(a.out,a.parent,a.smoke,a.weights,a.cache)
    elif a.stage == "freeze":
        freeze(a.out,a.parent,a.weights,a.cache)
    else:
        score(a.out,a.parent,a.archive)
