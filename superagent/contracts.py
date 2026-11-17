"""Task and pipeline facts checked before an agent can launch training.

This is a deliberately narrow deterministic contract adapter, not a general
natural-language interpreter. Unknown pair semantics fail closed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TaskContract:
    version: int
    kind: str
    objective: str
    labels: list[int]
    metric: str
    pair_relation: str | None
    swap_inverts_label: bool
    instruction_source: str
    instruction_sha256: str
    instruction_excerpt: str

    def to_dict(self):
        return asdict(self)


def build_contract(task):
    labels = np.asarray(task.y_train)
    if labels.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("This runtime currently supports binary labels {0, 1} only; multiclass is not implemented.")
    if len(task.X_train) != len(labels):
        raise ValueError("Training inputs and labels must have equal lengths.")
    instructions = task.spec.get("instructions", "")
    source = task.spec.get("instruction_source", "unavailable")
    relation = None
    if task.kind == "pair_text":
        # Recognise the explicit ordering contract, not the dataset name or
        # the mere existence of two text columns. Do not generalise to similarity.
        normalized = re.sub(r"[`*_]", "", instructions).lower()
        normalized = " ".join(normalized.split())
        explicit = re.search(r"labelpairwise.{0,40}\b1\b if a had the higher rate and\s+0\s+if b did", normalized)
        if not explicit:
            raise ValueError("Pair-text label semantics are unverified: need explicit 1=A higher rate, 0=B higher rate instructions. No swap augmentation will be guessed.")
        if any(not isinstance(row, (tuple, list)) or len(row) != 2 or
               not all(isinstance(v, str) for v in row) for row in task.X_train):
            raise ValueError("Pair inputs must be two strings per row.")
        relation = "ordered_comparison"
        objective = "Predict which headline has higher observed click-through rate: 1=A, 0=B; not semantic similarity."
    elif task.kind == "categorical_tabular":
        if np.asarray(task.X_train).ndim != 2:
            raise ValueError("Categorical tabular input must be a matrix.")
        objective = "Predict the supplied binary target from categorical features; see task instructions for target meaning."
    else:
        raise ValueError(f"No verified pipeline contract for task kind {task.kind!r}.")
    return TaskContract(1, task.kind, objective, [0, 1], "accuracy", relation,
                        relation == "ordered_comparison", source,
                        hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
                        instructions[:8000])


def describe_candidate(kind, candidate):
    """Metadata for the actual search_candidates implementation (not LLM prose)."""
    cid = candidate.family
    if kind == "categorical_tabular":
        if cid.startswith("logistic_"):
            representation, estimator = "clean categories + OneHotEncoder(handle_unknown=ignore)", "LogisticRegression"
        elif cid.startswith("extra_trees_"):
            representation, estimator = "clean categories + OneHotEncoder(handle_unknown=ignore)", "ExtraTreesClassifier"
        elif cid.startswith("catboost_"):
            representation, estimator = "missing sentinels + categorical strings", "CatBoostClassifier(binary Logloss)"
        else:
            raise ValueError(f"Candidate lacks pipeline metadata: {cid}")
        augmentation = "none"
    elif kind == "pair_text":
        representation = "word TF-IDF ngrams(1,2)"
        if cid.startswith("word_char_"):
            representation += " + char_wb TF-IDF ngrams(3,5)"
        elif not cid.startswith("word_"):
            raise ValueError(f"Candidate lacks pipeline metadata: {cid}")
        representation += "; features = vector(A) - vector(B)"
        estimator = "LogisticRegression"
        augmentation = "swap A/B and invert label; requires verified ordered-comparison contract"
    else:
        raise ValueError(f"Unsupported kind: {kind}")
    return {"id": cid, "params": candidate.params, "representation": representation,
            "estimator": estimator, "augmentation": augmentation,
            "preprocessing_fit_scope": "inner_fit_only", "model_seed": 42}
