"""Describe paired accuracy uncertainty when repeated covariate profiles are kept together."""
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_seed_completion import origin
from scripts.study_task1_threshold import get_fold
from scripts.unified_released_study import read, require, save, sha
from superagent.joint_context import tokens
from superagent.models import _clean_categories
from superagent.profile_history import discover_columns


def checked_array(path, digest):
    require(sha(path) == digest, "Modified method prediction")
    return np.load(path, allow_pickle=False)


def historical(out, idx, fold, n):
    base = fold / "regular"
    require(sha(base / "solve.csv") == read(base / "sealed.json")["prediction"], "Unbound original")
    seed = out / "seed_replication" / str(idx)
    context = out / "cross_context_outer" / str(idx)
    bag = out / "paired_weights" / str(idx)
    return dict(original=prediction_csv(base / "solve.csv", n),
                seed=checked_array(seed / "always.npy", read(seed / "sealed.json")["predictions"]["always"]),
                context=checked_array(context / "regular.npy", read(context / "sealed.json")["predictions"]["regular"]),
                seed_context=checked_array(context / "repeated.npy", read(context / "sealed.json")["predictions"]["repeated"]),
                weighted=checked_array(bag / "predictions.npy", read(bag / "sealed.json")["predictions"]))


def study(out, section, profiles, full_counts, cols):
    folds, groups, support, hits = [], [], [], {}
    if section == "historical":
        locks = [out / k / "result.json" for k in ("seed_replication","cross_context_outer","paired_weights")]
        comparisons = (("seed","original"), ("context","original"), ("seed_context","seed"), ("weighted","seed"))
    else:
        locks = [out / "seed_completion/result.json"]
        comparisons = (("seed","original"),)
    identity = {p.parent.name:sha(p) for p in locks}
    for i in range(5):
        fold = get_fold(out, 48, i) if section == "historical" else origin(out, i)
        info = read(fold / "inputs.json")
        train, query = (fold / "anonymous" / n for n in ("training.zip", "queries.zip"))
        require(sha(train) == info["entry_inputs"] and sha(query) == info["entry_queries"]
                and sha(fold / "outer_answers.joblib") == info["outer_answers"], "Unbound diagnostic rows")
        x, = arrays(train, ["C_train.npy"])
        q, = arrays(query, ["C_test.npy"])
        held_keys = tokens(_clean_categories(q)[:,cols])
        train_keys = Counter(tokens(_clean_categories(x)[:,cols]))
        require(train_keys + Counter(held_keys) == full_counts, "Wrong profile population in this fold")
        if section == "historical":
            guesses = historical(out,i,fold,len(q))
        else:
            entry = read(out / "seed_completion/score_lock.json")["folds"][str(i)]
            base = out / "seed_completion" / f"original_{i}.npy"
            p = fold / "replicated/solve.csv"
            require(sha(p) == entry["proposed"], "Modified ordinary prediction")
            guesses = dict(original=checked_array(base, entry["original"]), seed=prediction_csv(p,len(q)))
        truth = joblib.load(fold / "outer_answers.joblib")["y"].astype(str)
        require(all(v.shape == truth.shape for v in guesses.values()), "Incomplete diagnostic predictions")
        groups.extend(held_keys)
        support.extend(train_keys[k] for k in held_keys)
        for key, predictions in guesses.items():
            hits.setdefault(key, []).extend(predictions == truth)
        folds.append(sha(fold / "inputs.json"))
    require(Counter(groups) == full_counts and len(groups) == len(profiles), "Five-fold profile coverage differs")
    counts = np.asarray(support)
    groups = np.asarray(groups)
    scores = {k:np.asarray(v) for k,v in hits.items()}
    comparisons = {a + "_vs_" + b:compare(scores[a], scores[b], groups) for a,b in comparisons}
    bins = {}
    for name, mask in [("no_history",counts==0),("one_to_four",(counts>=1)&(counts<=4)),
                       ("five_to_twelve",(counts>=5)&(counts<=12)),("thirteen_or_more",counts>=13)]:
        bins[name] = dict(rows=int(mask.sum()), profiles=int(len(set(groups[mask]))),
                          correct={k:int(v[mask].sum()) for k,v in scores.items()})
    return dict(source_scores=identity, inputs=folds, profile_clustered=comparisons, by_prior_profile_rows=bins)


def run(out):
    source = out / "remaining_folds_source"
    train = source / "anonymous/training.zip"
    require(sha(train) == read(source / "inputs.json")["entry_inputs"], "Wrong feature packet")
    whole, = arrays(train, ["C_train.npy"])
    clean = _clean_categories(whole)
    cols = list(discover_columns(clean))
    require(bool(cols), "No supported recurring profile")
    profiles = tokens(clean[:,cols])
    counts = Counter(profiles)
    result = {k:study(out,k,profiles,counts,cols) for k in ("historical","prospective")}
    value = dict(evaluator_features=sha(train), profile_field_count=len(cols), distinct_profiles=len(counts), total_rows=len(profiles),
             results=result, interpret_as="Exploratory clustered uncertainty on already sealed solutions. Identical inferred profiles do not establish individual identity. No primary gate or decisions are changed.",
             task_classifier_fits=0, api_requests=0)
    save(out / "profile_uncertainty.json", value)
    print(value["distinct_profiles"], result["prospective"]["profile_clustered"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    run(parser.parse_args().out)
