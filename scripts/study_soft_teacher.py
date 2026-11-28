"""Learn from an elected teacher only on groups which none of its models fitted."""
import argparse
from copy import copy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask, load_prediction_features
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take


PLAN = ROOT / "research/paired_encoder_search_1127/soft_teacher_protocol.json"
ARMS = ("hard", "uniform", "teacher")


def digest(matrix):
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


def fit_head(parent, features, matrix, labels, teaching, target):
    match = np.where(labels[teaching] == 1, target, 1 - target)
    weights = np.ones(len(labels))
    weights[teaching] = match
    minority = teaching[match < 1.]
    x = vstack([matrix, -matrix, matrix[minority], -matrix[minority]], format="csr")
    y = np.concatenate([labels, 1-labels, 1-labels[minority], labels[minority]])
    w = np.concatenate([weights, weights, 1-weights[minority], 1-weights[minority]])
    require(np.all(w >= 0) and np.isclose(w.sum(), 2*len(labels)), "Expected equal effective sample weight")
    head = clone(parent.named_steps["classifier"]).fit(x, y, sample_weight=w)
    model = copy(parent)
    model.steps = [("features", features), ("classifier", head)]
    return model


def predictions(models, rows):
    return {arm: np.asarray(m.predict(rows)) for arm, m in models.items()}


def compare(pred, truth):
    right = {arm: arr == truth for arm, arr in pred.items()}
    return dict(rows=len(truth), correct={arm: int(v.sum()) for arm,v in right.items()},
                teacher_vs={arm: paired(right["teacher"], right[arm]) for arm in ("hard", "uniform")})


def run(root):
    out = root / "soft-teacher"
    out.mkdir(exist_ok=True)
    old = root / "resource-audition"
    source = root / "arctic-menu/original/run/agent/refit_data.joblib"
    upstream, fitting = read(old/"inputs.json"), read(old/"fitting.json")
    recorded = read(old/"predictions.json")
    require(sha(source) == upstream["source"] and sha(old/"fitting.json") == recorded["fitting"]
            and sha(old/"worker_input.joblib") == fitting["payload"], "Upstream study changed")
    data = joblib.load(source)
    rows, y = data["X_train"], np.asarray(data["y_train"])
    metric = MetricSpec.from_payload(data)
    require(list(metric.labels) == [0,1] and metric.name == "accuracy", "Expected this group's binary metric")
    groups = sample_groups(TrainingTask(data["kind"], rows, y), "shared_text")
    splitter = StratifiedGroupKFold(n_splits=5, random_state=upstream["seed"], shuffle=True)
    folds = [v for _,v in splitter.split(groups, y, groups)]
    nomination, teaching, audit = folds[:3]
    base = np.sort(np.concatenate(folds[3:]))
    parts = dict(fitting=base, nomination=nomination, confirmation=teaching, audit=audit)
    require(all(upstream["partitions"][k]["key"] == fingerprint(v) for k,v in parts.items())
            and sum(len(set(groups[v])) for v in parts.values()) == len(set(groups)), "Roles overlap or changed")
    activate_resource_directory(root/"arctic-library")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/"arctic-menu/fit_features"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    bank, probabilities, pupils = {}, {}, []
    for name, archived in recorded["components"].items():
        file = old/name/"trial.joblib"
        receipt = read(file.parent/"receipt.json")
        require(receipt["payload_sha256"] == fitting["payload"] and sha(file) == receipt["trial_sha256"],
                "Teacher was not fitted on the agreed groups")
        trial = joblib.load(file)
        observation = planner_observation(trial, trial["metrics"], dict(sha256=fitting["payload"], labels=y[nomination]), metric)
        p = np.asarray(observation["probabilities"])
        require(digest(p) == archived["nomination"], "Nomination probabilities changed")
        bank[name], probabilities[name] = trial["model"], p
        steps = getattr(trial["model"], "named_steps", {})
        if "features" in steps and isinstance(steps.get("classifier"), LogisticRegression):
            pupils.append(name)
    require(pupils and probabilities, "No existing eligible fitted evidence")
    # Tie-break on anonymous predictions before stable names for exactly equivalent models.
    order = lambda name: (digest(probabilities[name]), name)
    student = min(pupils, key=lambda n: (-np.mean(probabilities[n].argmax(axis=1)==y[nomination]), order(n)))
    pool = sorted(probabilities, key=order)
    options = [k for size in (1,2) for k in combinations(pool, size)]
    def teacher_loss(names):
        avg = np.mean([probabilities[n][:,1] for n in names], axis=0)
        return float(np.mean((avg-y[nomination])**2))
    teachers = min(options, key=lambda k: (teacher_loss(k), len(k), tuple(map(order,k))))
    lock(out/"election.json", dict(protocol=sha(PLAN), upstream=sha(old/"predictions.json"),
         student=student, teachers=list(teachers), teacher_brier=teacher_loss(teachers), candidates=len(pool),
         partitions={k:fingerprint(v) for k,v in parts.items()}, fitting_rows=len(base)+len(teaching)))
    teaching_rows = take(rows,teaching)
    signals = []
    for name in teachers:
        model = bank[name]
        p = model.predict_proba(teaching_rows)[:,[list(model.classes_).index(c) for c in metric.labels]]
        require(digest(p) == recorded["components"][name]["confirmation"], "Teaching probabilities changed")
        signals.append(p[:,1])
    targets = dict(hard=y[teaching], uniform=.5*y[teaching]+.25,
                   teacher=.5*(y[teaching]+np.mean(signals,axis=0)))
    train = np.concatenate([base,teaching])
    raw = take(rows,train)
    parent = bank[student]
    feat = clone(parent.named_steps["features"]).fit(raw)
    matrix = feat.transform(raw)
    indices = np.arange(len(base),len(train))
    ledger = out/"fits.jsonl"
    models, seals = {}, {}
    for arm in ARMS:
        file, seal = out/(arm+".joblib"), out/(arm+".json")
        if not seal.exists():
            spent = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
            require(arm not in spent and len(spent)<len(ARMS), "Unreconciled fit or budget exhausted")
            with ledger.open("a") as stream:
                stream.write(json.dumps(arm)+"\n")
            model = fit_head(parent,feat,matrix,y[train],indices,targets[arm])
            joblib.dump(model,file)
            lock(seal, dict(arm=arm, election=sha(out/"election.json"), model=sha(file)))
        seals[arm] = read(seal)["model"]
        require(sha(file)==seals[arm], "Stored student changed")
        models[arm] = joblib.load(file)
    lock(out/"sealed.json", dict(election=sha(out/"election.json"), models=seals))
    first = compare(predictions(models,take(rows,audit)), y[audit])
    lock(out/"inner.json", first)
    second = None
    if all(d["net"]>0 for d in first["teacher_vs"].values()):
        menu = root/"arctic-menu"
        spec = read(menu/"original/run/agent/task_spec.json")
        inputs = read(menu/"inputs.json")
        require(sha(menu/"features.zip")==inputs["features"], "Outer inputs changed")
        query = load_prediction_features(menu/"features.zip",data["kind"],spec)
        require(not {s for pair in raw for s in pair} & {s for pair in query for s in pair}, "Outer training overlap")
        outer_pred = predictions(models,query)
        lock(out/"outer_lock.json",dict(models=seals, predictions={a:digest(v) for a,v in outer_pred.items()}))
        require(sha(menu/"outer_labels.joblib")==inputs["outer_labels"], "Outer truth changed")
        second = compare(outer_pred,joblib.load(menu/"outer_labels.joblib")["labels"])
    promoted = (second is not None and all(d["net"]>0 for d in second["teacher_vs"].values())
                and (first["teacher_vs"]["hard"]["net"]+second["teacher_vs"]["hard"]["net"])/
                (first["rows"]+second["rows"])>=.002)
    actual = len(ledger.read_text().splitlines())
    require(actual == len(ARMS), "Unexpected physical count")
    result = dict(seal=sha(out/"sealed.json"), student=student, teachers=list(teachers), inner=first, outer=second,
                  advance=bool(promoted), new_classifier_fits=actual, upstream_fits_reused=len(pool),
                  model_api_requests=0, designated_validation_opened=False, official_test_opened=False, H2_opened=False)
    lock(out/"results.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=lambda s: Path(s).resolve())
    with threadpool_limits(limits=2):
        run(parser.parse_args().root)
