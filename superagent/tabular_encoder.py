"""Feedback-selected, frozen embeddings of reusable categorical level names."""
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.decomposition import PCA

from .confirmation import shortlist
from .models import Candidate, _clean_categories, _tabular_catboost, _tabular_logistic, _tabular_trees
from .pretrained_choice import Encoder, budget
from .task_budget import BudgetStateError

MODEL = 'qwen3-embedding-0.6b'
SEMANTIC = 'pretrained_qwen3_levels'
PERMUTED = 'ablation_permuted_qwen3_levels'
FILES = {
    'config.json': ('b5afe39ac0d47717f5104238e7df93635f36adec', 'b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9', 727),
    'tokenizer.json': ('c75dc100bea0a07420353465595ecfef791e7e83', 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a', 11423705),
    'model.safetensors': ('5092237580d1545d466a2d454c09f18181c341ec', '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd', 1191586416),
}
MISSING = {'', 'nan', 'none', 'null', 'na', '?', '__missing__'}
WORD = re.compile(r'[^\W\d_]{3,}|[\u3400-\u9fff]{2,}', re.UNICODE)
BACKBONES = ('catboost_depth', 'extra_trees_leaf', 'logistic_C')


class LevelEncoder(Encoder):
    model_id = MODEL
    repository = 'Qwen/Qwen3-Embedding-0.6B'
    files = FILES
    weight_key = 'tabular_weights'
    cache_name = 'category_vectors.sqlite'
    last_token = True


def level_names(column):
    return Counter(s for value in column if (s := str(value).strip()) and s.casefold() not in MISSING)


def eligible_columns(rows):
    """Use repeats, word-like names and bounded cardinality; no column names or labels."""
    eligible = []
    for index, column in enumerate(np.asarray(rows).T):
        counts = level_names(column)
        if not 3 <= len(counts) <= 128 or sum(counts.values()) < 2 * len(counts):
            continue
        readable = sum(bool(WORD.search(name)) for name in counts)
        if 2 * readable >= len(counts):
            eligible.append((index, counts))
    # Prefer more distinct readable levels when the safety bound is reached.
    return sorted(eligible, key=lambda item: (-len(item[1]), item[0]))[:8]


def decide_level_transfer(report, history, common_errors, train, search, config):
    if train is None or search is None:
        return dict(report, reason='no_categorical_features')
    columns = eligible_columns(train)
    descriptions = []
    rare_rows = np.zeros(len(search), dtype=bool)
    for index, counts in columns:
        median = np.median(list(counts.values()))
        low = {s for s, count in counts.items() if count <= median and count < max(counts.values())}
        held = [str(v).strip() for v in np.asarray(search)[:, index]]
        rare_rows |= np.array([s.casefold() not in MISSING and (s in low or s not in counts) for s in held])
        descriptions.append(dict(column=index, levels=len(counts), lower_support_levels=len(low)))
    required = max(4, math.ceil(.10 * int(common_errors.sum())))
    report = dict(report, eligible_columns=descriptions, shared_errors_with_low_support_level=int((rare_rows & common_errors).sum()),
                  required_low_support_errors=required)
    if not descriptions:
        return dict(report, reason='no_repeated_word_like_levels')
    if report['shared_errors_with_low_support_level'] < required:
        return dict(report, reason='insufficient_shared_errors_on_lower_support_levels')
    incumbent = shortlist(history, 1)[0]
    if not incumbent['candidate'].startswith(BACKBONES):
        return dict(report, reason='incumbent_has_no_compatible_feature_adapter')
    mode = config.get('level_assignment', 'semantic')
    if mode not in ('semantic', 'permuted'):
        raise ValueError('Unknown category level assignment')
    names = {name for _, counts in columns for name in counts}
    allowance = sum(min(64, len(s.encode('utf-8')) + 2) for s in names)
    try:
        budget(config, allowance, model_id=MODEL)
    except BudgetStateError as exc:
        return dict(report, reason=str(exc), estimated_max_input_tokens=allowance)
    params = dict(model=MODEL, backbone=incumbent['candidate'], backbone_params=incumbent['params'],
                  columns=[d['column'] for d in descriptions], dimensions_per_column=3, assignment=mode)
    return dict(report, decision='try', reason='Original trials share errors on rows with lower-support named levels; test added category features on the incumbent.',
                unique_names=len(names), estimated_max_input_tokens=allowance,
                candidate=SEMANTIC if mode == 'semantic' else PERMUTED, params=params)


class LevelClassifier:
    """Fold-local lookup tables and a fitted classifier; prediction never invokes the encoder."""
    def __init__(self, params):
        self.params = params

    def fit(self, rows, labels, encoder):
        raw = np.asarray(rows)
        self.lookups = {}
        counts = {i: sorted(level_names(raw[:, i])) for i in self.params['columns']}
        names = sorted({s for levels in counts.values() for s in levels})
        vectors = dict(zip(names, encoder.encode(names))) if names else {}
        for i, levels in counts.items():
            if len(levels) < 2:
                continue
            matrix = np.stack([vectors[s] for s in levels])
            dims = min(self.params['dimensions_per_column'], len(levels)-1, matrix.shape[1])
            coords = PCA(n_components=dims, svd_solver='full').fit_transform(matrix)
            if self.params['assignment'] == 'permuted':
                cycle = np.random.default_rng(45317+i).permutation(len(levels))
                swapped = coords.copy()
                swapped[cycle] = coords[np.roll(cycle, 1)]
                coords = swapped
            self.lookups[i] = dict(zip(levels, coords))
        p, backbone = self.params['backbone_params'], self.params['backbone']
        if backbone.startswith('catboost_depth'):
            self.one_hot = None
            self.classifier = _tabular_catboost(labels, p['depth'], p['l2_leaf_reg'])
            if 'one_hot_max_size' in self.params:
                self.classifier.set_params(one_hot_max_size=self.params['one_hot_max_size'])
            self.classifier.fit(self.features(rows), labels, cat_features=list(range(raw.shape[1])))
        else:
            from sklearn.preprocessing import OneHotEncoder
            self.one_hot = OneHotEncoder(handle_unknown='ignore').fit(_clean_categories(rows))
            if backbone.startswith('logistic_C'):
                self.classifier = _tabular_logistic(p['C'])
            elif backbone.startswith('extra_trees_leaf'):
                self.classifier = _tabular_trees(p['min_samples_leaf'])
            else:
                raise ValueError('Unknown categorical backbone')
            self.classifier.fit(self.features(rows), labels)
        self.classes_ = self.classifier.classes_
        return self

    def features(self, raw):
        from scipy.sparse import csr_matrix, hstack
        raw = np.asarray(raw)
        parts = []
        for index, lookup in self.lookups.items():
            zero = np.zeros_like(next(iter(lookup.values())))
            part = np.asarray([lookup.get(str(s).strip(), zero) for s in raw[:, index]], dtype=np.float32)
            parts.append(part.reshape(len(raw), len(zero)))
        dense = np.hstack(parts) if parts else np.empty((len(raw), 0))
        clean = _clean_categories(raw)
        if self.one_hot is not None:
            return hstack([self.one_hot.transform(clean), csr_matrix(dense)], format='csr')
        return np.hstack([clean.astype(object), dense])

    def predict(self, rows):
        return self.classifier.predict(self.features(rows)).reshape(-1)

    def predict_proba(self, rows):
        return self.classifier.predict_proba(self.features(rows))


def candidate(kind, candidate_id, runtime):
    decision = json.loads(Path(runtime).with_name('pretrained_decision.json').read_text())
    if (kind != 'categorical_tabular' or decision['decision'] != 'try' or decision['candidate'] != candidate_id
            or decision['params']['model'] != MODEL or decision['params']['backbone'] != decision['protected_incumbent']):
        raise ValueError('Category encoder does not match the frozen decision')
    params = decision['params']
    def fit(rows, labels, search):
        model = LevelClassifier(params).fit(rows, labels, LevelEncoder(runtime))
        return model, model.predict(search)
    return Candidate(candidate_id, params, fit)
