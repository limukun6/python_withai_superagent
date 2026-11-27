"""Test whether disjoint reference outcomes add information in a supervised tree-leaf representation."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.field_interactions import InteractionTree
from superagent.joint_context import ContextTokenTree, blocks
from superagent.splits import sample_groups

PLAN = ROOT / "research/task1_feedback_extension_1127/honest_leaf_encoder.json"
ADAPTER = PLAN.with_name("leaf_encoder_adapter.json")
NEIGHBOURS = (32, 96)
METHODS = tuple(f"k{k}" for k in NEIGHBOURS)


def leaf_matrix(leaves, offsets, width):
    n, count = leaves.shape
    indices = (leaves + offsets).astype(np.int32).ravel()
    indptr = np.arange(0, (n + 1) * count, count, dtype=np.int32)
    return csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(n, width))


def local_residuals(ref, query, residual, leaf_counts):
    offsets = np.cumsum(np.r_[0, leaf_counts[:-1]])
    width = int(np.sum(leaf_counts))
    memory = leaf_matrix(ref, offsets, width).T.tocsr()
    representation = leaf_matrix(query, offsets, width)
    predictions = {k: np.empty(len(query)) for k in NEIGHBOURS}
    for start in range(0, len(query), 64):
        block = representation[start:start + 64]
        overlap = (block @ memory).toarray() / query.shape[1]
        for k in NEIGHBOURS:
            effective_k = min(k, len(residual))
            cutoff = np.partition(overlap, -effective_k, axis=1)[:, -effective_k, None]
            weight = np.where(overlap >= cutoff, overlap ** 8, 0.0).astype(float)
            total = weight.sum(axis=1)
            squares = np.square(weight).sum(axis=1)
            effective = np.divide(total ** 2, squares, out=np.zeros_like(total), where=squares > 0)
            correction = np.divide(weight @ residual, total, out=np.zeros_like(total), where=total > 0)
            predictions[k][start:start + len(correction)] = correction * effective / (effective + k / 4)
    return predictions


def train(out):
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    x, raw_y = fitting["x"], fitting["y"]
    classes, y = np.unique(raw_y, return_inverse=True)
    require(len(classes) == 2, "This residual experiment requires two classes")
    groups = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=x))
    panels = [bound(out, role + "_features.joblib", info) for role in ROLES]
    all_query = np.concatenate(panels)
    bounds = np.cumsum([0] + [len(p) for p in panels])
    corrections = {k: np.zeros(len(all_query)) for k in NEIGHBOURS}
    shared = out / "leaf_encoder_components_02"
    shared.mkdir(exist_ok=False)
    audit = []
    splitter = StratifiedGroupKFold(2, shuffle=True, random_state=73931)
    for i, (train_index, memory_index) in enumerate(splitter.split(x, y, groups)):
        require(not set(groups[train_index]).intersection(groups[memory_index]), "Overlapping supervised encoder and memory")
        save(shared / f"fit_{i}.json", dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"),
             encoder_rows=len(train_index), memory_rows=len(memory_index)))
        try:
            blocks(x[train_index])
        except ValueError as exc:
            if str(exc) != "No supported recurring profile and varying context":
                raise
            encoder = InteractionTree(dict(pair_limit=3, backbone_params=dict(depth=6, l2_leaf_reg=3.0)))
        else:
            encoder = ContextTokenTree(dict(pair_limit=3))
        encoder.fit(x[train_index], y[train_index])
        require(np.array_equal(encoder.classes_, [0, 1]), "Unexpected probability order")
        cat = encoder.classifier
        tree_count = cat.tree_count_
        chosen = np.linspace(0, tree_count - 1, min(128, tree_count), dtype=int)
        counts = np.asarray(cat.get_tree_leaf_counts(), dtype=int)[chosen]
        features = encoder.transform(x[memory_index])
        ref = cat.calc_leaf_indexes(features, thread_count=2)[:, chosen]
        query = cat.calc_leaf_indexes(encoder.transform(all_query), thread_count=2)[:, chosen]
        residual = y[memory_index] - cat.predict_proba(features, thread_count=2)[:, 1]
        estimates = local_residuals(ref, query, residual, counts)
        for k, values in estimates.items():
            corrections[k] += values / 2
        audit.append(dict(fold=i, encoder_rows=len(train_index), memory_rows=len(memory_index), trees=int(tree_count),
                          sampled_trees=len(chosen), encoder=type(encoder).__name__,
                          unique_reference_groups=len(set(groups[memory_index]))))
    save(shared / "sealed.json", dict(protocol=sha(PLAN), adapter=sha(ADAPTER), inputs=sha(out / "inputs.json"), folds=audit,
         classifier_fits=len(audit), indexed_reference_sets=len(audit), encoder_memory_group_overlap=0, api_requests=0))
    source = out / "trial_anchor"
    base = read(source / "sealed.json")
    require(base["classes"] == classes.tolist(), "Unmatched original labels")
    for k in NEIGHBOURS:
        destination = out / f"leaf_k{k}"
        destination.mkdir(exist_ok=False)
        signatures = {}
        for role, left, right in zip(ROLES, bounds[:-1], bounds[1:], strict=True):
            path = source / (role + ".npy")
            require(sha(path) == base["predictions"][role], "Unbound original probability")
            positive = np.clip(np.load(path, allow_pickle=False)[:, 1] + corrections[k][left:right], .001, .999)
            require(np.isfinite(positive).all(), "Non-finite corrected probability")
            target = destination / (role + ".npy")
            np.save(target, np.column_stack((1 - positive, positive)), allow_pickle=False)
            signatures[role] = sha(target)
        save(destination / "sealed.json", dict(protocol=sha(PLAN), components=sha(shared / "sealed.json"),
             classes=classes.tolist(), predictions=signatures, base=sha(source / "sealed.json"), api_requests=0))


def grade(out):
    elect(out, METHODS, "leaf", PLAN)
    info = read(out / "inputs.json")
    comparisons = {}
    for role in ROLES:
        y = bound(out, role + "_answers.joblib", info)
        hits = {}
        for name in ("trial_anchor", *("leaf_" + m for m in METHODS)):
            directory = out / name
            seal = read(directory / "sealed.json")
            p = directory / (role + ".npy")
            require(sha(p) == seal["predictions"][role], "Changed predictions")
            hits[name] = np.asarray(seal["classes"])[np.load(p, allow_pickle=False).argmax(1)] == y
        comparisons[role] = {m: paired(hits["leaf_" + m], hits["trial_anchor"]) for m in METHODS}
    save(out / "leaf_election/vs_original.json", comparisons)
    print(comparisons, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    (train if args.action == "train" else grade)(args.out)
