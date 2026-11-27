"""Test a landmark preference model against the last autonomous election on training data."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.analogy_review import blend_counts, incumbent_evidence, nominate, promoted
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.preference_analogy import views, embedding_function
from superagent.splits import sample_groups


PLAN = ROOT/"research/paired_encoder_search_1127/preference_kernel_protocol.json"
FOLDS = ("primary", "confirmation_1", "confirmation_2")


def landmark_ids(rows, groups):
    # Use a canonical unordered text pair, so presentation order and the outcome cannot pick a landmark.
    first = {}
    for i, (pair, group) in enumerate(zip(rows, groups)):
        a, b = sorted(pair)
        key = hashlib.sha256(len(a.encode()).to_bytes(8, "big")+a.encode()+b.encode()).digest()
        if group not in first or key < first[group][0]:
            first[group] = (key, i)
    count = min(512, int(np.ceil(6*np.sqrt(len(first)))))
    return [i for _, i in sorted(first.values())[:count]]


def kernel(d, m, e, n):
    return (d @ e.T) * np.square((1.+m @ n.T)/2.)


def learn(train, held, y, groups, provider):
    d, m = views(train, provider)
    e, n = views(held, provider)
    ids = landmark_ids(train, groups)
    ld, lm = d[ids], m[ids]
    gram = np.asarray(kernel(ld, lm, ld, lm), dtype=float)
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    use = values > max(values[-1]*1e-7, 1e-12)
    require(use.any(), "Training features have no usable orientation")
    transform = vectors[:, use] / np.sqrt(values[use])
    def features(change, context):
        return np.column_stack((change, kernel(change, context, ld, lm) @ transform))
    model = LogisticRegression(C=1., fit_intercept=False, max_iter=1000, random_state=42)
    model.fit(features(d, m), y)
    query = features(e, n)
    p = model.predict_proba(query)
    reverse, reverse_context = views([(b,a) for a,b in held[:24]], provider)
    require(np.allclose(model.predict_proba(features(reverse, reverse_context))[:,1]+p[:len(reverse),1],
                        1., rtol=0, atol=1e-6), "Reversing a comparison must invert its probability")
    require(list(model.classes_) == [0, 1] and np.isfinite(p).all(), "Invalid probability matrix")
    return p, dict(landmarks=len(ids), rank=int(use.sum()), positive_probability_sha256=fingerprint(p[:,1]))


def run(root):
    out = root/"preference-kernel"
    out.mkdir(exist_ok=True)
    require(not (out/"results.json").exists(), "Kernel study already complete")
    previous = out/"attempt_01_failure.json"
    prior_fits = read(previous)["completed_fits_without_predictions"] if previous.exists() else 0
    base = root/"analogy-portfolio-raw/portfolio_revised"
    agent = base/"run/agent"
    review, seal = read(agent/"analogy_review.json"), read(base/"sealed.json")
    require(review["status"] == "accepted" and review["final_members"] == seal["members"]
            and review["model_sha256"] == seal["model_sha256"], "Invalid elected ensemble")
    data = joblib.load(agent/"refit_data.joblib")
    rows, labels = data["X_train"], np.asarray(data["y_train"])
    metric = MetricSpec.from_payload(data)
    require(metric.name == "accuracy" and list(metric.labels) == [0,1]
            and data["pair_semantics"] == "preference", "Expected a declared binary preference")
    groups = sample_groups(TrainingTask(data["kind"], rows, labels), "shared_text")
    history = read(agent/"history.json")
    sources = {s["member"]:s["execution_key"] for s in review["source_candidates"]}
    require(0 < len(sources) <= 2, "Source candidates exceed the registered budget")
    activate_resource_directory(root/"discovery-library")
    os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"] = str(root/"feature_bank")
    paths, panels = {}, {}
    for fold in FOLDS:
        split = read(agent/(fold+"_split.json"))
        require(sha(agent/(fold+".joblib")) == split["payload_sha256"], "Changed source payload")
        completed = {r["candidate"]:r["trial"] for r in history if r["status"] == "completed" and r["split_id"] == fold}
        paths[fold] = {m:agent/f"trial_{completed[m]}.joblib" for m in review["incumbent_members"]}
        paths[fold][review["candidate"]] = agent/f"analogy_{fold}_{review['candidate']}.joblib"
        panels[fold] = split
    lock(out/"inputs.json", dict(protocol=sha(PLAN), election=sha(base/"sealed.json"), review=sha(agent/"analogy_review.json"),
         sources=sources, panels={f:dict(payload=panels[f]["payload_sha256"],
         components={m:sha(p) for m,p in paths[f].items()}) for f in FOLDS}))
    # This label-free cache reuses pointwise frozen vectors only; all supervised features are fitted per fold.
    texts = sorted({t for pair in rows for t in pair})
    text_index = {t:i for i,t in enumerate(texts)}
    vectors = {}
    for name in sources:
        item = joblib.load(paths["primary"][name])
        encode = embedding_function(item["model"].named_steps["features"])
        vectors[name] = np.asarray(encode(texts))
    def anchors(fold):
        split = panels[fold]
        truth = labels[split["held"]]
        observations = []
        for m in seal["members"]:
            item = joblib.load(paths[fold][m])
            require(item["metrics"]["resolved_configuration"]["id"] == m, "Component identity changed")
            observations.append(planner_observation(item, item["metrics"],
                                dict(sha256=split["payload_sha256"], labels=truth), metric))
        return incumbent_evidence(observations, review["weights"], metric)
    def attempt(source, fold, anchor):
        split = panels[fold]
        fit, held = np.asarray(split["fit"]), np.asarray(split["held"])
        require(not set(groups[fit]) & set(groups[held]), "Training group also appears in evaluation")
        provider = lambda strings: vectors[source][[text_index[s] for s in strings]]
        receipt_path = out/(fold+"_"+sources[source][:12]+".json")
        retained = receipt_path.with_suffix(".npy")
        if retained.exists() and receipt_path.exists():
            positive = np.load(retained,allow_pickle=False)
            require(len(positive) == len(held) and read(receipt_path)["positive_probability_sha256"] == fingerprint(positive),
                    "Retained kernel predictions changed")
            return dict(anchor, probabilities=np.column_stack((1-positive,positive)))
        require(not retained.exists() and not receipt_path.exists(), "Incomplete supervised prediction receipt")
        probs, detail = learn([rows[i] for i in fit], [rows[i] for i in held], labels[fit], groups[fit], provider)
        np.save(retained, probs[:,1], allow_pickle=False)
        candidate = dict(anchor, probabilities=probs)
        lock(receipt_path, dict(fold=fold, source=source, fits=1, **detail))
        return candidate
    primary = anchors("primary")
    nominees = {}
    fits = 0
    for name in sources:
        candidate = attempt(name, "primary", primary)
        fits += 1
        nominees[name] = nominate(primary, candidate, primary["labels"])
    eligible = [n for n in sources if nominees[n]["weight"] > 0]
    winner = min(eligible, key=lambda n:(-nominees[n]["primary_gain"], nominees[n]["weight"], sources[n])) if eligible else None
    lock(out/"nomination.json", dict(inputs=sha(out/"inputs.json"), nominees=nominees, winner=winner))
    confirmations = []
    if winner is not None:
        decision = nominees[winner]
        for fold in FOLDS[1:]:
            anchor = anchors(fold)
            other = attempt(winner, fold, anchor)
            fits += 1
            start, after = blend_counts(anchor, other, anchor["labels"], [0., decision["weight"]])
            receipt = dict(fold=fold, rows=len(anchor["labels"]), anchor=start, candidate=after, gain=after-start)
            lock(out/(fold+"_score.json"), receipt)
            confirmations.append(receipt)
            if after < start or (after == start and decision["primary_gain"] <= 0):
                break
    passed = bool(winner is not None and promoted(nominees[winner]["primary_gain"], confirmations,
                  primary_rows=len(primary["labels"])))
    result = dict(nomination=sha(out/"nomination.json"), nominees=nominees, source=winner,
                  confirmation=confirmations, passed=passed, local_classifier_fits=fits+prior_fits,
                  fits_with_scored_predictions=fits, completed_fits_without_predictions=prior_fits,
                  supervised_share_nominations=len(sources), model_api_calls=0, designated_validation_opened=False,
                  official_test_opened=False, H2_opened=False)
    lock(out/"results.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=lambda value: Path(value).resolve(), required=True)
    with threadpool_limits(limits=2):
        run(parser.parse_args().root)
