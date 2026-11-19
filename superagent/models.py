"""Candidate portfolios selected from data schema, not task identity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        ("model", _tabular_logistic(c)),
    ])
    model.fit(_clean_categories(x_train), y_train)
    return model, model.predict(_clean_categories(x_val))


def _extra_trees_tabular(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, min_samples_leaf: int = 2
) -> tuple[Any, np.ndarray]:
    model = Pipeline([
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ("model", _tabular_trees(min_samples_leaf)),
    ])
    model.fit(_clean_categories(x_train), y_train)
    return model, model.predict(_clean_categories(x_val))


def _tabular_logistic(c):
    return LogisticRegression(max_iter=2000, C=c, random_state=42)


def _tabular_trees(min_samples_leaf):
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=min_samples_leaf, max_features=.8,
                                n_jobs=-1, random_state=42, class_weight="balanced")


def _tabular_catboost(y_train, depth, l2_leaf_reg):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=600,
        depth=depth,
        learning_rate=0.05,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss" if len(np.unique(y_train))==2 else "MultiClass",
        eval_metric="Accuracy",
        verbose=False,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )


def _catboost_tabular(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
    depth: int = 7, l2_leaf_reg: float = 3.0,
) -> tuple[Any, np.ndarray]:
    x_train_clean = _clean_categories(x_train)
    x_val_clean = _clean_categories(x_val)
    model = _tabular_catboost(y_train, depth, l2_leaf_reg)
    model.fit(x_train_clean, y_train, cat_features=list(range(x_train_clean.shape[1])))
    return model, model.predict(x_val_clean).reshape(-1)


def _dummy_text(x_train: list[tuple[str, str]], y_train: np.ndarray, x_val: list[tuple[str, str]]) -> tuple[Any, np.ndarray]:
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(x_train), 1)), y_train)
    return model, model.predict(np.zeros((len(x_val), 1)))


def _difference_features(vectorizer: TfidfVectorizer, pairs: list[tuple[str, str]], fit: bool) -> Any:
    from .feature_cache import paired_features
    def compute():
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        if fit:vectorizer.fit(left + right)
        return vectorizer.transform(left) - vectorizer.transform(right)
    return paired_features(vectorizer,pairs,fit,compute)


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


def _char_difference_text(x_train, y_train, x_val, c=1.0):
    from scipy.sparse import vstack
    vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=60_000)
    matrix=_difference_features(vectorizer,x_train,True)
    classifier=LogisticRegression(C=c,max_iter=2000,random_state=42)
    classifier.fit(vstack([matrix,-matrix]),np.concatenate([y_train,1-y_train]))
    return (vectorizer,classifier),classifier.predict(_difference_features(vectorizer,x_val,False))


def text_statistics(text):
    """Language-agnostic shape/count features; no label or task-specific word list."""
    words=re.findall(r"\b\w+\b",text,flags=re.UNICODE)
    n_chars,n_words=max(1,len(text)),max(1,len(words))
    counts=[len(text),len(words),sum(map(len,words))/n_words,
            sum(any(c.isdigit() for c in word) for word in words),
            text.count('?'),text.count('!'),text.count(','),text.count(':'),text.count(';'),
            text.count('"')+text.count("'"),sum(text.count(c) for c in '()[]{}')]
    ratios=[sum(c.isdigit() for c in text)/n_chars,
            sum(c.isupper() for c in text)/n_chars,
            sum(c.isspace() for c in text)/n_chars,
            len({word.casefold() for word in words})/n_words,
            sum(word[0].isupper() for word in words)/n_words]
    return np.asarray([*np.log1p(counts),*ratios],dtype=float)


def style_differences(pairs):
    return np.asarray([text_statistics(a)-text_statistics(b) for a,b in pairs])


@dataclass
class StylePairModel:
    word: Any
    char: Any
    scaler: Any
    classifier: Any

    @property
    def classes_(self): return self.classifier.classes_

    def transform(self,pairs):
        from scipy.sparse import csr_matrix
        styles=self.scaler.transform(style_differences(pairs))
        styles=styles/np.sqrt(styles.shape[1])
        return hstack([_difference_features(self.word,pairs,False),
                       _difference_features(self.char,pairs,False),csr_matrix(styles)]).tocsr()

    def predict(self,pairs): return self.classifier.predict(self.transform(pairs))

    def predict_proba(self,pairs): return self.classifier.predict_proba(self.transform(pairs))


def _word_char_stats_text(x_train,y_train,x_val,c=1.0,edit=False):
    from scipy.sparse import csr_matrix,vstack
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=40_000)
    char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=40_000)
    word_matrix=_difference_features(word,x_train,True)
    char_matrix=_difference_features(char,x_train,True)
    raw_styles=style_differences(x_train)
    scaler=StandardScaler(with_mean=False).fit(np.vstack([raw_styles,-raw_styles]))
    styles=scaler.transform(raw_styles)/np.sqrt(raw_styles.shape[1])
    matrix=hstack([word_matrix,char_matrix,csr_matrix(styles)]).tocsr()
    if edit:
        from .challengers import EditPairModel, edit_features
        matrix=hstack([matrix,edit_features(x_train)]).tocsr()
    classifier=LogisticRegression(C=c,max_iter=2000,random_state=42)
    classifier.fit(vstack([matrix,-matrix]),np.concatenate([y_train,1-y_train]))
    model=(EditPairModel if edit else StylePairModel)(word,char,scaler,classifier)
    return model,model.predict(x_val)


def candidates_for(kind: str) -> list[Candidate]:
    if kind in ("numeric_tabular","mixed_tabular","text_tabular","single_text","image"):
        from .general_models import generic_candidates
        return generic_candidates(kind)
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
    from .generated_pipeline import GeneratedModel
    from .pretrained_choice import ContextClassifier
    if isinstance(model,(GeneratedModel,ContextClassifier)):return np.asarray(model.predict(inputs)).reshape(-1)
    from .trainable_pair import NeuralPairModel, LearnedPairStack
    if isinstance(model, (NeuralPairModel, LearnedPairStack)):
        return np.asarray(model.predict(inputs)).reshape(-1)
    from .recipe_models import RecipeTextModel
    if isinstance(model,RecipeTextModel):return np.asarray(model.predict(inputs)).reshape(-1)
    from .supervised_text import NBPairModel
    from .nonlinear_text import StyleTreePairModel,SymmetricPairModel,FixedStyleBlend
    if isinstance(model,(StyleTreePairModel,SymmetricPairModel,FixedStyleBlend)):return model.predict(inputs)
    if isinstance(model,NBPairModel):return model.predict(inputs)
    if isinstance(model, (SoftVoteModel,StylePairModel)):
        return model.predict(inputs)
    if kind in ("numeric_tabular","mixed_tabular","text_tabular","single_text","image"):
        return np.asarray(model.predict(inputs)).reshape(-1)
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


def probability_with_model(kind, model, inputs):
    """Return fitted class labels with probability columns in that same order."""
    from .generated_pipeline import GeneratedModel
    from .pretrained_choice import ContextClassifier
    if isinstance(model,(GeneratedModel,ContextClassifier)):return model.classes_,model.predict_proba(inputs)
    from .trainable_pair import NeuralPairModel, LearnedPairStack
    if isinstance(model, (NeuralPairModel, LearnedPairStack)):
        return model.classes_, model.predict_proba(inputs)
    from .recipe_models import RecipeTextModel
    if isinstance(model,RecipeTextModel):return model.classes_,model.predict_proba(inputs)
    from .supervised_text import NBPairModel
    from .nonlinear_text import StyleTreePairModel,SymmetricPairModel,FixedStyleBlend
    if isinstance(model,(StyleTreePairModel,SymmetricPairModel,FixedStyleBlend)):return model.classes_,model.predict_proba(inputs)
    if isinstance(model,NBPairModel):return model.classes_,model.predict_proba(inputs)
    if isinstance(model, (SoftVoteModel,StylePairModel)):
        return model.classes_, model.predict_proba(inputs)
    if kind in ("numeric_tabular","mixed_tabular","text_tabular","single_text","image"):
        if not hasattr(model,'predict_proba'):raise ValueError('Selected classifier has no probability output')
        return model.classes_,model.predict_proba(inputs)
    if kind == "categorical_tabular":
        classifier, matrix = model, _clean_categories(inputs)
    elif kind == "pair_text":
        if isinstance(model, DummyClassifier):
            classifier, matrix = model, np.zeros((len(inputs), 1))
        elif len(model) == 2:
            vectorizer, classifier = model
            matrix = _difference_features(vectorizer, inputs, False)
        elif len(model) == 3:
            word, char, classifier = model
            matrix = hstack([_difference_features(word, inputs, False), _difference_features(char, inputs, False)])
        else:
            raise ValueError("Unrecognised saved pair-text model structure")
    else:
        raise ValueError(f"Unsupported task kind: {kind}")
    classes = np.asarray(classifier.classes_)
    probabilities = np.asarray(classifier.predict_proba(matrix), dtype=float)
    if (probabilities.shape != (len(inputs), len(classes)) or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.)):
        raise ValueError("Invalid class probabilities")
    return classes, probabilities


@dataclass
class SoftVoteModel:
    """A fixed equal-weight vote, including explicit class-column alignment."""
    kind: str
    models: list[Any]
    classes_: np.ndarray

    def predict_proba(self, inputs):
        if len(self.models) < 2:
            raise ValueError("Soft voting requires at least two models")
        values = []
        for model in self.models:
            classes, probabilities = probability_with_model(self.kind, model, inputs)
            if set(classes.tolist()) != set(self.classes_.tolist()):
                raise ValueError("Ensemble components have different fitted classes")
            order = [classes.tolist().index(label) for label in self.classes_.tolist()]
            values.append(probabilities[:, order])
        return np.mean(values, axis=0)

    def predict(self, inputs):
        return self.classes_[np.argmax(self.predict_proba(inputs), axis=1)]


def search_candidates(kind: str, portfolio: str = "original") -> list[Candidate]:
    """Common search space, deliberately larger than a benchmark trial budget."""
    if portfolio not in ("original","expanded","research","nonlinear","adaptive","semantic","learned","challenger","all"):
        raise ValueError("Unknown portfolio")
    if kind in ("numeric_tabular","mixed_tabular","text_tabular","single_text","image"):
        from .general_models import generic_candidates
        from .recipe_models import semantic_candidates
        return generic_candidates(kind) + (semantic_candidates(kind) if portfolio in ("semantic","all") else [])
    if kind == "categorical_tabular":
        base = [
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
        if portfolio in ('challenger','all'):
            from .challengers import cross_candidate
            base=base+[cross_candidate()]
        if portfolio in ('adaptive','all'):
            from .adaptive_tabular import adaptive_candidates
            return base+adaptive_candidates()
        return base
    if kind == "pair_text":
        original = [
            Candidate(f"{name}_C{c}", {"C": c, "representation": name, "swap_augmentation": True},
                      partial(function, c=c))
            for name, function in (("word", _word_difference_text), ("word_char", _word_char_difference_text))
            for c in (0.05, 0.1, 0.3, 0.7, 1.5, 3.0, 10.0)
        ]
        if portfolio=="original": return original
        expanded = original + [
            Candidate(f"{name}_C{c}", {"C":c,"representation":name,"swap_augmentation":True},partial(function,c=c))
            for name,function in (("char",_char_difference_text),("word_char_stats",_word_char_stats_text))
            for c in (0.1,0.3,1.0,3.0)]
        if portfolio in ('research','all'):
            from .supervised_text import research_candidates
            expanded=expanded+research_candidates()
        if portfolio in ('nonlinear','all'):
            from .nonlinear_text import nonlinear_candidates
            expanded=expanded+nonlinear_candidates()
        if portfolio in ('semantic','all'):
            from .recipe_models import semantic_candidates
            expanded=expanded+semantic_candidates(kind)
        if portfolio in ('learned','all'):
            from .trainable_pair import trainable_candidates
            expanded=expanded+trainable_candidates(kind)
        if portfolio in ('challenger','all'):
            expanded=expanded+[Candidate('edit_pair_C0.3',{'C':0.3,'representation':'word_char_stats_plus_aligned_edits',
                'swap_augmentation':True,'hash_bins':65536},partial(_word_char_stats_text,c=0.3,edit=True))]
        return expanded
    raise ValueError(f"Unsupported task kind: {kind}")

