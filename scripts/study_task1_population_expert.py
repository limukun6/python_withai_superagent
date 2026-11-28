"""Test an opt-in correction on genuinely unseen groups from released training rows."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_conditional_opportunity import metrics, ordered
from scripts.study_task1_extensions import key
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_population_tree import PopulationTree, novel_queries
from scripts.study_task1_seed_completion import origin
from scripts.study_task1_unseen_profiles import FOLDS, STUDY, completed_arm
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.candidate_refinement import resolve_registered
from superagent.joint_context import tokens
from superagent.models import _clean_categories
from superagent.profile_history import discover_columns
from superagent.training_randomness import fitted_seed, training_seed

STAGE = "population_expert_1128"
PLAN = ROOT / "research/task1_feedback_extension_1127/novel_profile_expert.json"
PHASES = ("nominate", "confirm")
SHARES = (0., .25, .5, 1.)


def combine(control, expert, eligible, weight):
    return np.where(eligible, control + weight * (expert - control), control)


def rebuild_control(src, members, fit, held, x, y, classes, area, name):
    registry = read(src / "candidate_registry.json")
    repeat = read(src / "seed_replication.json")
    components = []
    fits = 0
    for j, member in enumerate(members):
        seeds = [None]
        if repeat["action"] == "adopt" and member in repeat["eligible"]:
            seeds.extend(repeat["seeds"])
        method = resolve_registered("categorical_tabular", member, registry)
        per_seed = []
        for n, seed in enumerate(seeds):
            save(area / f"{name}-base-{j}-{n}.json", dict(fit=key(fit), held=key(held), member=member, seed=seed))
            with training_seed(seed):
                model, _ = method.fit_predict(x[fit], y[fit], x[held])
            require(seed is None or fitted_seed(model) == seed, "Requested training seed was not applied")
            per_seed.append(ordered(model, x[held], classes))
            del model
            fits += 1
        components.append(np.mean(per_seed, axis=0))
    return np.mean(components, axis=0), fits


def build(out, i):
    fold = out / STUDY / str(i)
    side = completed_arm(fold)
    src = side / "run/agent"
    rc = read(side / "sealed.json")
    info = read(fold / "inputs.json")
    train, query = (fold / "anonymous" / f for f in ("training.zip", "queries.zip"))
    require(sha(train) == info["entry_inputs"] and sha(query) == info["entry_queries"]
            and rc["independent_reload"] and sha(side / "solve.csv") == rc["prediction"] == sha(side / "reload.csv"),
            "Unverified original solve or input")
    x, y = arrays(train, ["C_train.npy", "y_train.npy"])
    q, = arrays(query, ["C_test.npy"])
    classes = np.unique(y)
    table = _clean_categories(x)
    cols = discover_columns(table)
    require(len(classes) == 2 and cols, "Stress-test panels need a binary task with a discoverable group")
    groups = np.asarray(tokens(table[:, cols]))
    area = out / STAGE / str(i)
    area.mkdir(exist_ok=False, parents=True)
    save(area / "launch.json", dict(protocol=sha(PLAN), parent=sha(side / "sealed.json"), train=sha(train), query=sha(query)))
    kept = [pair for j, pair in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=39013)
                                          .split(x, y, groups)) if j in (0, 3)]
    require(set(groups[kept[0][1]]).isdisjoint(groups[kept[1][1]]), "Inner holdouts overlap")
    scores, audits = {}, {}
    extra_fits = 0
    for name, (fit, held) in zip(PHASES, kept, strict=True):
        require(set(groups[held]).isdisjoint(groups[fit]), "Inner training contains held profiles")
        base, n_fits = rebuild_control(src, rc["members"], fit, held, x, y, classes, area, name)
        mask = novel_queries(x[fit], x[held])
        pred = base
        extra_fits += n_fits
        if mask.any():
            save(area / f"{name}-expert.json", dict(fit=key(fit), held=key(held)))
            expert = PopulationTree().fit(x[fit], y[fit])
            require(np.array_equal(mask, expert.novel(x[held])), "Model and eligibility disagree")
            pred = ordered(expert, x[held], classes)
            extra_fits += 1
            audits[name] = expert.audit
        truth = y[held] == classes[1]
        scores[name] = {w: metrics(combine(base, pred, mask, w), truth, base) for w in SHARES}
        save(area / f"{name}-feedback.json", dict(rows=len(held), eligible=int(mask.sum()),
             reference_fits=n_fits, feedback={str(w): v for w, v in scores[name].items()}))
    first = scores["nominate"]
    nominee = min(SHARES, key=lambda w: (-first[w]["correct"], first[w]["brier"], w))
    panels = [scores[k][nominee] for k in PHASES]
    accept = bool(nominee and min(r["net"] for r in panels) >= 0
                  and sum(r["net"] for r in panels) >= .002 * sum(r["rows"] for r in panels)
                  and sum(r["brier_change"] * r["rows"] for r in panels) <= 0)
    share = nominee if accept else 0.
    save(area / "decision.json", dict(protocol=sha(PLAN), nominee=nominee, chosen=share,
         feedback={k: sha(area / f"{k}-feedback.json") for k in PHASES}))
    final = read(src / "final_selection.json")
    parent_path = src / final["model_file"]
    require(sha(parent_path) == final["model_sha256"] == rc["model"], "Wrong incumbent export")
    base = ordered(joblib.load(parent_path)["model"], q, classes)
    require(np.array_equal(classes[(base > .5).astype(int)].astype(str), prediction_csv(side / "solve.csv", len(q))),
            "Incumbent labels differ")
    model_hash = None
    mask = novel_queries(x, q)
    pred = base
    if mask.any():
        save(area / "deployment-expert.json", dict(fit=sha(train), query=sha(query)))
        model = PopulationTree().fit(x, y)
        require(np.array_equal(mask, model.novel(q)), "Exported model eligibility differs")
        pred = ordered(model, q, classes)
        path = area / "expert.joblib"
        joblib.dump(model, path, compress=3)
        model_hash = sha(path)
        extra_fits += 1
        audits["deployment"] = model.audit
    file = area / "sealed_features.npz"
    np.savez(file, base=base, expert=pred, novel=mask, classes=classes.astype(str))
    all_fits = extra_fits + rc["fits"]
    require(all_fits <= 64, "Per-arm training ceiling exceeded")
    save(area / "built.json", dict(decision=sha(area / "decision.json"), probabilities=sha(file), model=model_hash,
         query=sha(query), base_fits=rc["fits"], new_fits=extra_fits, upper_bound_path_fits=all_fits,
         eligible=int(mask.sum()), expert_audits=audits))
    print("population", i, "share", share, "panel nets", [r["net"] for r in panels], "fits", extra_fits, flush=True)


def seal(out, folds):
    root = out / STAGE
    locks = {}
    for i in folds:
        area, src = root / str(i), out / STUDY / str(i)
        state = read(area / "built.json")
        file = area / "sealed_features.npz"
        decision = read(area / "decision.json")
        query = src / "anonymous/queries.zip"
        require(sha(file) == state["probabilities"] and sha(query) == state["query"]
                and sha(area / "decision.json") == state["decision"], "Build evidence changed")
        with np.load(file, allow_pickle=False) as f:
            base, pred, mask, classes = (f[k] for k in ("base", "expert", "novel", "classes"))
        if state["model"]:
            path = area / "expert.joblib"
            require(sha(path) == state["model"], "Expert model changed")
            x, = arrays(query, ["C_test.npy"])
            model = joblib.load(path)
            np.testing.assert_allclose(ordered(model, x, model.classes_), pred, rtol=1e-7, atol=1e-8)
            require(np.array_equal(model.novel(x), mask), "Replayed novelty mask differs")
        hashes = {}
        for tag, w in dict(control=0., chosen=decision["chosen"], quarter=.25, population=1.).items():
            result = classes[(combine(base, pred, mask, w) > .5).astype(int)]
            require(np.array_equal(result[~mask], classes[(base[~mask] > .5).astype(int)]), "Changed known-profile decisions")
            dst = area / (tag + ".npy")
            np.save(dst, result, allow_pickle=False)
            hashes[tag] = sha(dst)
        lock = area / "sealed.json"
        save(lock, dict(build=sha(area / "built.json"), predictions=hashes, independent_reload=True))
        locks[str(i)] = sha(lock)
    regular = []
    for i in range(5):
        fold = origin(out, i) / "anonymous"
        x, = arrays(fold / "training.zip", ["C_train.npy"])
        q, = arrays(fold / "queries.zip", ["C_test.npy"])
        regular.append(dict(fold=i, rows=len(q), eligible=int(novel_queries(x, q).sum())))
    save(root / ("score_lock_" + "-".join(map(str, folds)) + ".json"),
         dict(protocol=sha(PLAN), locks=locks, regular_coverage=regular))


def grade(out, folds):
    root = out / STAGE
    suffix = "-".join(map(str, folds))
    lock_file = root / f"score_lock_{suffix}.json"
    lock = read(lock_file)
    results, groups, hits = [], [], {}
    for i in folds:
        area, src = root / str(i), out / STUDY / str(i)
        state = read(area / "built.json")
        receipts = read(area / "sealed.json")
        require(sha(area / "sealed.json") == lock["locks"][str(i)] and receipts["build"] == sha(area / "built.json"),
                "Unverified population-policy outputs")
        answers = src / "outer_answers.joblib"
        require(sha(answers) == read(src / "inputs.json")["external_answers"], "Unverified answer packet")
        packet = joblib.load(answers)
        truth = packet["y"].astype(str)
        require(set(groups).isdisjoint(packet["groups"]), "Overlapping external folds")
        groups.extend(packet["groups"])
        row = {}
        for n, digest in receipts["predictions"].items():
            p = area / (n + ".npy")
            require(sha(p) == digest, "Predictions changed")
            pred = np.load(p, allow_pickle=False)
            require(pred.shape == truth.shape, "Wrong rows in sealed output")
            bit = pred == truth
            row[n] = int(bit.sum())
            hits.setdefault(n, []).extend(bit)
        results.append(dict(fold=i, rows=len(truth), eligible=state["eligible"], correct=row, new_fits=state["new_fits"],
                            source_fits=state["base_fits"], decision=read(area / "decision.json"),
                            route_constructions=sum(v["supervised_route_estimations"] for v in state["expert_audits"].values())))
    pairs = {k: compare(np.asarray(h), np.asarray(hits["control"]), groups) for k, h in hits.items() if k != "control"}
    ok = all(r["correct"]["chosen"] > r["correct"]["control"] for r in results) and pairs["chosen"]["gain"] >= .003
    value = dict(lock=sha(lock_file), rows=len(groups), folds=results, independent_profiles=len(set(groups)),
                 correct={k:int(np.sum(h)) for k,h in hits.items()}, comparisons=pairs, advance=bool(ok),
                 regular_coverage=lock["regular_coverage"], new_fits=sum(r["new_fits"] for r in results),
                 route_constructions=sum(r["route_constructions"] for r in results), model_api_requests=0,
                 designated_validation_opened=False, official_test_opened=False, h2_opened=False)
    save(root / ("result_" + suffix + ".json"), value)
    print(value["correct"], len(groups), pairs["chosen"], ok, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "seal", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    parser.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    args = parser.parse_args()
    if args.action == "build":
        for i in args.folds:
            build(args.out, i)
    else:
        {"seal": seal, "grade": grade}[args.action](args.out, args.folds)
