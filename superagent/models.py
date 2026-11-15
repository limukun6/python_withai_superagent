"""Candidate portfolios selected from data schema, not task identity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class Candidate:
    family: str
    params: dict[str, Any]
    fit_predict: Callable[[Any, np.ndarray, Any], tuple[Any, np.ndarray]]


def _clean_categories(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=object).astype(str)
    missing = np.isin(np.char.lower(np.char.strip(values)), ["nan", "none", "null", "na", "?"])
    values[missing] = "__MISSING__"
    return values


def _dummy_tabular(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> tuple[Any, np.ndarray]:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(x_train, y_train)
    return model, model.predict(x_val)


def _logistic_tabular(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, c: float = 1.0
) -> tuple[Any, np.ndarray]:
    model = Pipeline([
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ("model", LogisticRegression(max_iter=2000, C=c, random_state=42)),
    ])
    model.fit(_clean_categories(x_train), y_train)
    return model, model.predict(_clean_categories(x_val))


def _extra_trees_tabular(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, min_samples_leaf: int = 2
) -> tuple[Any, np.ndarray]:
    model = Pipeline([
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ("model", ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=min_samples_leaf, max_features=0.8,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )),
    ])
    model.fit(_clean_categories(x_train), y_train)
    return model, model.predict(_clean_categories(x_val))


def _catboost_tabular(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    depth: int = 7,
    l2_leaf_reg: float = 3.0,
) -> tuple[Any, np.ndarray]:
    # Imported lazily so a missing optional accelerator does not prevent the
    # transparent sklearn fallback portfolio from completing.
    from catboost import CatBoostClassifier

    x_train_clean = _clean_categories(x_train)
    x_val_clean = _clean_categories(x_val)
    model = CatBoostClassifier(
        iterations=600,
        depth=depth,
        learning_rate=0.05,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="Accuracy",
        verbose=False,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )
    model.fit(x_train_clean, y_train, cat_features=list(range(x_train_clean.shape[1])))
    return model, model.predict(x_val_clean).reshape(-1)


def _dummy_text(x_train: list[tuple[str, str]], y_train: np.ndarray, x_val: list[tuple[str, str]]) -> tuple[Any, np.ndarray]:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(x_train), 1)), y_train)
    return model, model.predict(np.zeros((len(x_val), 1)))


def _difference_features(vectorizer: TfidfVectorizer, pairs: list[tuple[str, str]], fit: bool) -> Any:
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    if fit:
        vectorizer.fit(left + right)
    return vectorizer.transform(left) - vectorizer.transform(right)


def _word_difference_text(
    x_train: list[tuple[str, str]], y_train: np.ndarray, x_val: list[tuple[str, str]], c: float = 1.0
) -> tuple[Any, np.ndarray]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)
    train_matrix = _difference_features(vectorizer, x_train, fit=True)
    augmented_x = -train_matrix
    labels = np.concatenate([y_train, 1 - y_train])
    from scipy.sparse import vstack
    matrix = vstack([train_matrix, augmented_x])
    model = LogisticRegression(C=c, max_iter=2000, random_state=42)
    model.fit(matrix, labels)
    return (vectorizer, model), model.predict(_difference_features(vectorizer, x_val, fit=False))


def _word_char_difference_text(
    x_train: list[tuple[str, str]], y_train: np.ndarray, x_val: list[tuple[str, str]], c: float = 0.7
) -> tuple[Any, np.ndarray]:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40_000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=40_000)
    train_matrix = hstack([_difference_features(word, x_train, True), _difference_features(char, x_train, True)])
    from scipy.sparse import vstack
    matrix = vstack([train_matrix, -train_matrix])
    labels = np.concatenate([y_train, 1 - y_train])
    model = LogisticRegression(C=c, max_iter=2000, random_state=42)
    model.fit(matrix, labels)
    val_matrix = hstack([_difference_features(word, x_val, False), _difference_features(char, x_val, False)])
    return (word, char, model), model.predict(val_matrix)


def candidates_for(kind: str) -> list[Candidate]:
    if kind == "categorical_tabular":
        return [
            Candidate("majority_baseline", {}, _dummy_tabular),
            Candidate("one_hot_logistic_c0_25", {"C": 0.25}, partial(_logistic_tabular, c=0.25)),
            Candidate("one_hot_logistic_c1", {"C": 1.0}, partial(_logistic_tabular, c=1.0)),
            Candidate("one_hot_extra_trees_leaf1", {"n_estimators": 400, "min_samples_leaf": 1}, partial(_extra_trees_tabular, min_samples_leaf=1)),
            Candidate("one_hot_extra_trees_leaf3", {"n_estimators": 400, "min_samples_leaf": 3}, partial(_extra_trees_tabular, min_samples_leaf=3)),
            Candidate("catboost_depth6_reg5", {"iterations": 600, "depth": 6, "l2_leaf_reg": 5.0}, partial(_catboost_tabular, depth=6, l2_leaf_reg=5.0)),
            Candidate("catboost_depth8_reg3", {"iterations": 600, "depth": 8, "l2_leaf_reg": 3.0}, partial(_catboost_tabular, depth=8, l2_leaf_reg=3.0)),
        ]
    if kind == "pair_text":
        return [
            Candidate("majority_baseline", {}, _dummy_text),
            Candidate("word_tfidf_pairwise_c0_3", {"ngram_range": [1, 2], "C": 0.3, "swap_augmentation": True}, partial(_word_difference_text, c=0.3)),
            Candidate("word_tfidf_pairwise_c1", {"ngram_range": [1, 2], "C": 1.0, "swap_augmentation": True}, partial(_word_difference_text, c=1.0)),
            Candidate("word_tfidf_pairwise_c3", {"ngram_range": [1, 2], "C": 3.0, "swap_augmentation": True}, partial(_word_difference_text, c=3.0)),
            Candidate("word_char_pairwise_c0_3", {"word_ngrams": [1, 2], "char_ngrams": [3, 5], "C": 0.3, "swap_augmentation": True}, partial(_word_char_difference_text, c=0.3)),
            Candidate("word_char_pairwise_c0_7", {"word_ngrams": [1, 2], "char_ngrams": [3, 5], "C": 0.7, "swap_augmentation": True}, partial(_word_char_difference_text, c=0.7)),
            Candidate("word_char_pairwise_c1_5", {"word_ngrams": [1, 2], "char_ngrams": [3, 5], "C": 1.5, "swap_augmentation": True}, partial(_word_char_difference_text, c=1.5)),
        ]
    raise ValueError(f"Unsupported task kind: {kind}")


def predict_with_model(kind: str, model: Any, inputs: Any) -> np.ndarray:
    """Apply an already-fitted portfolio model without accessing labels."""
    if kind == "categorical_tabular":
        return np.asarray(model.predict(_clean_categories(inputs))).reshape(-1)
    if kind == "pair_text":
        if isinstance(model, DummyClassifier):
            return np.asarray(model.predict(np.zeros((len(inputs), 1)))).reshape(-1)
        if len(model) == 2:
            vectorizer, classifier = model
            matrix = _difference_features(vectorizer, inputs, fit=False)
        elif len(model) == 3:
            word, char, classifier = model
            matrix = hstack([
                _difference_features(word, inputs, fit=False),
                _difference_features(char, inputs, fit=False),
            ])
        else:
            raise ValueError("Unrecognised saved pair-text model structure.")
        return np.asarray(classifier.predict(matrix)).reshape(-1)
    raise ValueError(f"Unsupported task kind: {kind}")


def search_candidates(kind: str) -> list[Candidate]:
    """Common search space, deliberately larger than a benchmark trial budget."""
    if kind == "categorical_tabular":
        return [
            Candidate(f"logistic_C{c}", {"C": c}, partial(_logistic_tabular, c=c))
            for c in (0.05, 0.25, 1.0, 4.0)
        ] + [
            Candidate(f"extra_trees_leaf{leaf}", {"min_samples_leaf": leaf, "n_estimators": 400},
                      partial(_extra_trees_tabular, min_samples_leaf=leaf))
            for leaf in (1, 2, 4, 8)
        ] + [
            Candidate(f"catboost_depth{depth}_reg{reg}",
                      {"depth": depth, "l2_leaf_reg": reg, "iterations": 600, "learning_rate": 0.05},
                      partial(_catboost_tabular, depth=depth, l2_leaf_reg=reg))
            for depth in (4, 6, 8) for reg in (3.0, 10.0)
        ]
    if kind == "pair_text":
        return [
            Candidate(f"{name}_C{c}", {"C": c, "representation": name, "swap_augmentation": True},
                      partial(function, c=c))
            for name, function in (("word", _word_difference_text), ("word_char", _word_char_difference_text))
            for c in (0.05, 0.1, 0.3, 0.7, 1.5, 3.0, 10.0)
        ]
    raise ValueError(f"Unsupported task kind: {kind}")

