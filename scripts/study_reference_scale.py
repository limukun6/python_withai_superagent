"""Confirm an approved challenger against the same two reference classifiers trained on more groups."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.evaluate_pilot_nomination import store_arrays
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_brier_replication import RULE
from scripts.study_consensus_veto import trials
from scripts.study_crossfit_challenger import native_predict
from scripts.study_encoder_fusion import save_csv
from scripts.study_pilot_nomination import link_file
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_brier_replication import ORIGINAL, PUBLIC as REGROUPED
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import rows_at, score_members, ties
from tools.pilot_challenger import approve, compare_vote, pilot_roles
from tools.portfolio_training import configure, fit_one, write_selection

PLAN = ROOT / "docs/batches/reference_scale_plan_20261124_zh.md"


def folders(out, first, second):
    for name, public, parent in (("first", ORIGINAL, first), ("second", REGROUPED, second)):
        for i in range(5):
            part = f"fold_{i}"
            yield name, out / name / part, public / part, parent / part


def previous(src):
    run = src / RULE / "run"
    old, nomination = read(run / "decision.json"), read(run / "nomination.json")
    require(old["nomination"] == sha(run / "nomination.json") and old["feedback"] == sha(run / "feedback.npz"),
            "Incorrect original nomination or feedback")
    return run, old, nomination


def reconsider(src, task, larger_scores):
    run, old, nomination = previous(src)
    third = old["approved"]
    result = dict(original=sha(run / "decision.json"), third=third, approved=third, previous_check=old["confirm"],
                  new_check=None, larger_train_ids=None, confirm_ids=None)
    if not third:
        require(larger_scores is None, "Unexpected confirmation for a rejected nominee")
        return result
    groups = sample_groups(task, "shared_text")
    role = pilot_roles(groups, task.y_train)
    ids = np.sort(np.r_[role["pilot"], role["learn"]])
    check = role["confirm"]
    require(not set(groups[ids]) & set(groups[check]), "Larger reference was trained on a confirmation group")
    require(larger_scores.shape == (len(check), 2) and np.isfinite(larger_scores).all()
            and ((larger_scores >= 0) & (larger_scores <= 1)).all(), "Invalid larger-reference probabilities")
    with np.load(run / "feedback.npz", allow_pickle=False) as fb:
        require(np.array_equal(fb["confirm_ids"], check) and fb["confirm"].shape == (len(check), 3),
                "Original challenge scores were evaluated on different rows")
        challenger = fb["confirm"][:, 2]
        tie = ties(rows_at(task, check))
        old_check = compare_vote(fb["confirm"][:, :2].mean(1), challenger, tie, task.y_train[check], groups[check])
    require(old_check == old["confirm"], "Wrong original confirmation")
    checked = compare_vote(larger_scores.mean(1), challenger, tie, task.y_train[check], groups[check])
    decision = approve(third, nomination["learning"], checked, len(check))
    result.update(approved=decision["approved"], new_check=checked, larger_train_ids=fingerprint(ids.tolist()),
                  confirm_ids=fingerprint(check.tolist()))
    return result


def train(out, first, second, weights, cache):
    configure(weights, cache)
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(plan=sha(PLAN), source=sha(Path(__file__)), first=sha(ORIGINAL / "frozen.json"),
                    second=sha(REGROUPED / "frozen.json"), extra_fits=16, model_api_limit=0)
    save(out / "protocol.json", protocol)
    for _, dest, src, local in folders(out, first, second):
        dest.mkdir(parents=True, exist_ok=True)
        run, old, nomination = previous(src)
        packet = local / "train.zip"
        package = sha(packet)
        require(package == read(run / "pilot_protocol.json")["package"], "Changed anonymous input")
        task = load_training_task(packet)
        stored = dest / "decision.json"
        if stored.exists():
            matrix = None
            if old["approved"]:
                with np.load(dest / "larger.npz", allow_pickle=False) as arrays:
                    matrix = arrays["scores"]
            require(reconsider(src, task, matrix) == read(stored), "Previous scaled decision differs")
            continue
        larger = None
        if old["approved"]:
            groups = sample_groups(task, "shared_text")
            role = pilot_roles(groups, task.y_train)
            ids = np.sort(np.r_[role["pilot"], role["learn"]])
            identity = dict(package=package, ids=fingerprint(ids.tolist()))
            models = []
            with reuse_frozen_vectors():
                for name in nomination["members"][:2]:
                    path = dest / "larger" / name
                    fit_one(path, name, rows_at(task, ids), task.y_train[ids], identity)
                    models.append(joblib.load(path / "model.joblib"))
                larger = score_members(models, rows_at(task, role["confirm"]))
            store_arrays(dest / "larger.npz", dict(scores=larger))
        selection = reconsider(src, task, larger)
        save(stored, selection)
        print(dest.relative_to(out), "additional fits", 2 if old["approved"] else 0,
              "retained", selection["approved"], flush=True)


def freeze(out, first, second, weights):
    require(read(out / "protocol.json")["source"] == sha(Path(__file__)), "Training source changed")
    records = {}
    for name, dest, src, local in folders(out, first, second):
        run, old, nomination = previous(src)
        d = read(dest / "decision.json")
        require(d["original"] == sha(run / "decision.json"), "Original decision differs")
        changed = d["third"] != d["approved"]
        prior_rows = load_export(src / "rows.csv")
        inherited = load_export(src / RULE / "direct/predictions.csv")
        base = prior_rows["baseline"]
        global_key = RULE + "_approved" if name == "first" else "agent"
        require(np.array_equal(prior_rows["row_id"], inherited["row_id"]) and
                np.array_equal(inherited["prediction"], prior_rows[global_key]), "Original delivered rows differ")
        final = base if changed else inherited["prediction"]
        if changed:
            task = load_training_task(local / "train.zip")
            names = nomination["members"][:2]
            members = []
            end = read(run / "pilot_completion.json")
            model_sha = {a["candidate"]: a["model"] for a in end["actions"] if a["stage"] == "refit"}
            for n in names:
                path = local / RULE / "run/refit" / n / "model.joblib"
                require(sha(path) == model_sha[n], "Wrong delivered reference")
                members.append(joblib.load(path))
            agent = dest / "run/agent"
            if not agent.exists():
                write_selection(agent, PreferenceProbabilityVote("pair_text", members, np.array([0, 1])),
                                names, task.spec, "baseline", "larger-reference-confirmation")
            require(read(agent / "final_selection.json")["model_sha256"] == sha(agent / "selected_model.joblib"),
                    "Changed model bundle differs")
            link_file(local / "features.zip", dest / "features.zip")
            require(np.array_equal(native_predict(dest, weights), final), "Changed native prediction differs")
        native = (dest if changed else src / RULE) / "direct/predictions.csv"
        cols = dict(baseline=base, previous=inherited["prediction"], scaled=final)
        if not (dest / "rows.csv").exists():
            save_csv(dest / "rows.csv", inherited["row_id"], cols)
        observed = load_export(dest / "rows.csv")
        require(np.array_equal(observed.pop("row_id"), inherited["row_id"]) and
                all(np.array_equal(observed[k], v) for k, v in cols.items()), "Changed sealed rows")
        fits = {}
        for path in sorted((dest / "larger").glob("*/trained.json")):
            r = read(path)
            require(r["model"] == sha(path.parent / "model.joblib") and r["ledger"] == sha(path.parent / "fits.json"),
                    "Reference fit missing")
            fits[path.parent.name] = {k: read(path.parent / f) for k, f in
                (("fit", "trained.json"), ("ledger", "fits.json"), ("admission", "admission.json"))}
        require(len(fits) == (2 if d["third"] else 0), "Incorrect training budget")
        save(dest / "receipts.json", fits)
        files = ["rows.csv", "decision.json", "receipts.json"]
        if fits:
            files.append("larger.npz")
        if changed:
            files += ["direct/predictions.csv", "run/agent/final_selection.json"]
        records[name + "/" + src.name] = dict(files={f: sha(dest / f) for f in files}, native=sha(native),
                                               changed=changed, fits=len(fits))
    require(sum(v["fits"] for v in records.values()) == 16, "Incorrect study budget")
    save(out / "frozen.json", dict(protocol=sha(out / "protocol.json"), records=records))


def score(out, archive):
    frozen = read(out / "frozen.json")
    require(frozen["protocol"] == sha(out / "protocol.json") and read(out / "protocol.json")["plan"] == sha(PLAN),
            "Different preregistration")
    panels, folds = {}, {}
    for name, src, task, groups, _, held in trials(archive):
        key = name + "/" + src.name
        dest = out / key
        rec = frozen["records"][key]
        require(all(sha(dest / f) == v for f, v in rec["files"].items()), "Different frozen outputs")
        p = load_export(dest / "rows.csv")
        require(np.array_equal(p.pop("row_id"), np.arange(len(held))), "Unexpected output order")
        d = read(dest / "decision.json")
        folds.setdefault(name, []).append(dict(fold=src.name, original=d["third"], final=d["approved"],
                                                prior_check=d["previous_check"], scaled_check=d["new_check"],
                                                cost=rec["fits"]))
        for k, v in dict(**p, y=task.y_train[held], groups=groups[held]).items():
            panels.setdefault(name, {}).setdefault(k, []).append(v)
    results = {}
    for name, panel in panels.items():
        d = {k: np.concatenate(v) for k, v in panel.items()}
        require(len(d["y"]) == 6608 and len(set(d["groups"])) == 4874, "Incorrect development population")
        results[name] = dict(versus_previous=comparison(d["previous"], d["scaled"], d["y"], d["groups"]),
                             versus_reference=comparison(d["baseline"], d["scaled"], d["y"], d["groups"]), folds=folds[name])
    deltas = [v["versus_previous"] for v in results.values()]
    passed = all(v["net"] > 0 for v in deltas) and any(v["paired_95_fraction"][0] > 0 for v in deltas)
    save(out / "result.json", dict(frozen=sha(out / "frozen.json"), results=results, passes=passed,
                                    actual_new_classifier_fits=16, model_api_calls=0))
    print("Stronger-reference effect:", {k: v["versus_previous"]["net"] for k, v in results.items()}, "passes", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    tr = stages.add_parser("train")
    fr = stages.add_parser("freeze")
    sc = stages.add_parser("score")
    for cmd in (tr, fr, sc):
        cmd.add_argument("out", type=lambda s: Path(s).resolve())
    for cmd in (tr, fr):
        for key in ("first", "second", "weights"):
            cmd.add_argument("--" + key, required=True, type=lambda s: Path(s).resolve())
    tr.add_argument("--cache", required=True, type=lambda s: Path(s).resolve())
    sc.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.stage == "train":
        train(args.out, args.first, args.second, args.weights, args.cache)
    elif args.stage == "freeze":
        freeze(args.out, args.first, args.second, args.weights)
    else:
        score(args.out, args.archive)
