"""Random-initialized pair networks and grouped, training-only learned fusion.

No pretrained weights, language-model transport, dataset identifiers or labels
outside the current training capability are used. Torch is loaded only on fit.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
import hashlib
import json
import re

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from .data import TrainingTask
from .splits import inner_split, overlap_audit, sample_groups, take


MAX_WORDS = 20000
MAX_TOKENS = 80
MAX_EPOCHS = 24
PATIENCE = 4
MODEL_SEED = 4317
SELECTOR_SEED = 4311
OOF_SEED = 4321
COMPONENTS = ('word_char_stats_C1.0', 'char_C1.0', 'neural_pair_dim32')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def checked_training(pairs, labels):
    if not isinstance(pairs, (list, tuple)) or not pairs or any(
        not isinstance(pair, (list, tuple)) or len(pair) != 2 or
        any(not isinstance(s, str) for s in pair) for pair in pairs):
        raise ValueError('Expected pairs of text strings')
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(pairs) or set(y.tolist()) != {0, 1}:
        raise ValueError('Training requires both binary classes, 1 meaning left preferred')
    return y.astype(np.int64)


def tokens(text):
    return re.findall(r'\b\w+\b', text.casefold(), flags=re.UNICODE)[:MAX_TOKENS]


def vocabulary(pairs):
    counts = Counter(token for pair in pairs for text in pair for token in tokens(text))
    words = sorted((w for w, n in counts.items() if n >= 2), key=lambda w: (-counts[w], w))[:MAX_WORDS]
    return {w: i + 2 for i, w in enumerate(words)}


@contextmanager
def torch_context(seed):
    import torch
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            yield torch
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(threads)


def make_network(words, dimension):
    import torch
    from torch import nn

    class PairNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.EmbeddingBag(words + 2, dimension, mode='mean',
                                             include_last_offset=True, padding_idx=0)
            nn.init.normal_(self.embedding.weight, std=.1)
            with torch.no_grad(): self.embedding.weight[0].zero_()
            self.encoder = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh())
            self.head = nn.Sequential(nn.Linear(dimension * 4, dimension), nn.Tanh(),
                                      nn.Linear(dimension, 1, bias=False))

        def forward(self, flat, offsets):
            z = self.encoder(self.embedding(flat, offsets))
            left, right = z[0::2], z[1::2]
            delta = left - right
            context = torch.cat([left + right, left * right, torch.abs(delta)], dim=1)
            a = self.head(torch.cat([delta, context], dim=1))
            b = self.head(torch.cat([-delta, context], dim=1))
            return ((a - b) / 2).reshape(-1)
    return PairNetwork()


def encoded_documents(pairs, words):
    return [[words.get(w, 1) for w in tokens(s)] or [1] for pair in pairs for s in pair]


def packed(documents, rows):
    import torch
    selected = [documents[2 * int(i) + side] for i in rows for side in (0, 1)]
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in selected])])
    flat = np.asarray([w for s in selected for w in s], dtype=np.int64)
    return torch.from_numpy(flat), torch.from_numpy(offsets.astype(np.int64))


def logits_for(network, documents):
    import torch
    network.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(documents) // 2, 256):
            rows = np.arange(start, min(len(documents) // 2, start + 256))
            out.append(network(*packed(documents, rows)).cpu().numpy().astype(float))
    return np.concatenate(out) if out else np.empty(0)


def optimize(pairs, y, words, dimension, epochs, validation=None):
    import torch
    network = make_network(len(words), dimension)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.003, weight_decay=.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    documents = encoded_documents(pairs, words)
    target = torch.from_numpy(np.asarray(y, dtype=np.float32))
    held = encoded_documents(validation[0], words) if validation else None
    curve = []; best = float('inf'); best_epoch = 0
    for epoch in range(1, epochs + 1):
        network.train(); total = 0.
        order = torch.randperm(len(y)).numpy()
        for start in range(0, len(y), 128):
            rows = order[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(network(*packed(documents, rows)), target[rows])
            if not torch.isfinite(loss): raise ValueError('Nonfinite neural training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5.)
            optimizer.step(); total += float(loss.detach()) * len(rows)
        row = {'epoch': epoch, 'train_loss': total / len(y)}
        if validation:
            values = logits_for(network, held)
            measured = float(np.mean(np.logaddexp(0., values) - validation[1] * values))
            if not np.isfinite(measured): raise ValueError('Nonfinite neural selector loss')
            row['holdout_loss'] = measured
            if measured < best:
                best, best_epoch = measured, epoch
        curve.append(row)
        if validation and epoch - best_epoch >= PATIENCE: break
    return network, curve, best_epoch


@dataclass
class NeuralPairModel:
    words: dict
    dimension: int
    weights: dict
    training_metadata: dict
    classifier_fits: int = 2
    _network: object = field(default=None, repr=False)

    @property
    def classes_(self): return np.asarray([0, 1])

    def __getstate__(self):
        return {**self.__dict__, '_network': None}

    def decision_function(self, pairs):
        with torch_context(MODEL_SEED) as torch:
            if self._network is None:
                self._network = make_network(len(self.words), self.dimension)
                self._network.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in self.weights.items()})
            return logits_for(self._network, encoded_documents(pairs, self.words))

    def predict_proba(self, pairs):
        p = expit(self.decision_function(pairs))
        return np.column_stack([1 - p, p])

    def predict(self, pairs): return np.argmax(self.predict_proba(pairs), axis=1)


def fit_neural_pair(x_train, y_train, x_search, dimension=32):
    if type(dimension) is not int or dimension not in (32, 64):
        raise ValueError('Neural dimension must be32 or64')
    y = checked_training(x_train, y_train)
    task = TrainingTask('pair_text', x_train, y)
    fit, held = inner_split(task, SELECTOR_SEED, 'shared_text')
    audit = overlap_audit(task, fit, held)
    if audit['search_rows_sharing_text'] or any(set(y[ids].tolist()) != {0, 1} for ids in (fit, held)):
        raise ValueError('Both classes and disjoint text groups required for epoch selection')
    selection_words = vocabulary(take(x_train, fit))
    with torch_context(MODEL_SEED):
        selector, curve, chosen_epoch = optimize(take(x_train, fit), y[fit], selection_words, dimension,
            MAX_EPOCHS, (take(x_train, held), y[held]))
    del selector
    words = vocabulary(x_train)
    with torch_context(MODEL_SEED):
        final, refit_curve, _ = optimize(x_train, y, words, dimension, chosen_epoch)
    weights = {k: v.detach().cpu().numpy().copy() for k, v in final.state_dict().items()}
    metadata = {'method': 'random_initialized_neural_pair', 'pretrained_weights': False,
        'selection_data': 'current_X_train_y_train_only', 'training_rows': len(y),
        'training_features_sha256': fingerprint(x_train), 'training_labels_sha256': fingerprint(y.tolist()),
        'dimension': dimension, 'max_vocabulary': MAX_WORDS, 'min_token_count': 2, 'max_tokens': MAX_TOKENS,
        'selector_fit_indices': fit.tolist(), 'selector_holdout_indices': held.tolist(),
        'selector_overlap_audit': audit, 'selector_seed': SELECTOR_SEED, 'model_seed': MODEL_SEED,
        'selector_vocabulary': selection_words, 'full_vocabulary_sha256': fingerprint(words),
        'selector_curve': curve, 'selected_iterations': chosen_epoch, 'stopping_metric': 'logloss',
        'maximum_iterations': MAX_EPOCHS, 'patience': PATIENCE, 'refit_curve': refit_curve,
        'parameter_count': sum(int(v.size) for v in weights.values()), 'classifier_fits': 2,
        'refit_eval_set_used': False, 'outer_search_labels_accessed': False, 'cpu_threads': 2}
    model = NeuralPairModel(words, dimension, weights, metadata)
    return model, model.predict(x_search)


def fit_component(name, x, y):
    from .models import _word_char_stats_text, _char_difference_text
    if name == COMPONENTS[0]: return _word_char_stats_text(x, y, x[:1], c=1.)[0]
    if name == COMPONENTS[1]: return _char_difference_text(x, y, x[:1], c=1.)[0]
    if name == COMPONENTS[2]: return fit_neural_pair(x, y, x[:1], 32)[0]
    raise ValueError('Unknown declared fusion component')


def component_margin(model, pairs):
    from .models import probability_with_model
    _, forward = probability_with_model('pair_text', model, pairs)
    _, reverse = probability_with_model('pair_text', model, [(b, a) for a, b in pairs])
    p = (forward[:, 1] + 1 - reverse[:, 1]) / 2
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def complementary_gate(margins, labels):
    m = np.asarray(margins); y = np.asarray(labels)
    if m.ndim != 2 or m.shape[0] != len(y) or m.shape[1] != 3 or not np.isfinite(m).all():
        raise ValueError('Expected finite three-component OOF margins')
    predictions = m > 0
    correct = predictions == y[:, None]
    scores = correct.mean(axis=0)
    oracle = float(np.any(correct, axis=1).mean())
    disagreement = float(np.any(predictions != predictions[:, :1], axis=1).mean())
    best = min(range(3), key=lambda i: (-scores[i], COMPONENTS[i]))
    gain = oracle - float(scores[best])
    return {'requested': gain >= .02 and disagreement >= .05, 'best_component': COMPONENTS[best],
        'component_accuracies': {name: float(scores[i]) for i, name in enumerate(COMPONENTS)},
        'oracle_accuracy': oracle, 'oracle_gain': gain, 'disagreement_fraction': disagreement,
        'minimum_oracle_gain': .02, 'minimum_disagreement_fraction': .05,
        'oracle_is_achievable_score': False, 'source': 'training_grouped_out_of_fold_predictions_only'}


@dataclass
class LearnedPairStack:
    models: list
    component_names: list
    head: object
    training_metadata: dict
    audit_folds: list
    classifier_fits: int

    @property
    def classes_(self): return np.asarray([0, 1])

    def decision_function(self, pairs):
        matrix = np.column_stack([component_margin(m, pairs) for m in self.models])
        return matrix[:, 0] if self.head is None else self.head.decision_function(matrix)

    def predict_proba(self, pairs):
        p = expit(self.decision_function(pairs))
        return np.column_stack([1 - p, p])

    def predict(self, pairs): return np.argmax(self.predict_proba(pairs), axis=1)


def fit_learned_stack(x_train, y_train, x_search):
    y = checked_training(x_train, y_train)
    task = TrainingTask('pair_text', x_train, y)
    groups = sample_groups(task, 'shared_text')
    if len(set(groups)) < 8: raise ValueError('Too few independent groups for nested learned fusion')
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=OOF_SEED)
    margins = np.full((len(y), 3), np.nan); covered = np.zeros(len(y), dtype=int)
    folds = []; fits = 0; records = []
    for index, (fit, held) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        audit = overlap_audit(task, fit, held)
        if audit['search_rows_sharing_text'] or any(set(y[ids].tolist()) != {0, 1} for ids in (fit, held)):
            raise ValueError('Fusion folds need text isolation and both classes')
        models = []
        for column, name in enumerate(COMPONENTS):
            model = fit_component(name, take(x_train, fit), y[fit])
            margins[held, column] = component_margin(model, take(x_train, held))
            models.append(model); fits += int(getattr(model, 'classifier_fits', 1))
        covered[held] += 1
        folds.append({'fit': fit.tolist(), 'heldout': held.tolist(), 'models': models})
        records.append({'fold': index, 'fit_indices': fit.tolist(), 'heldout_indices': held.tolist(),
                        'overlap_audit': audit, 'classifier_fits': sum(int(getattr(m, 'classifier_fits', 1)) for m in models)})
    if not np.all(covered == 1) or not np.isfinite(margins).all():
        raise ValueError('Every row must have exactly one genuinely out-of-fold prediction')
    gate = complementary_gate(margins, y)
    head = None
    if gate['requested']:
        matrix = np.vstack([margins, -margins]); target = np.concatenate([y, 1 - y])
        head = Pipeline([('scale', StandardScaler(with_mean=False)),
                         ('classifier', LogisticRegression(C=.1, fit_intercept=False, max_iter=2000, random_state=42))])
        head.fit(matrix, target); fits += 1
        names = list(COMPONENTS)
    else:
        names = [gate['best_component']]
    models = []
    for name in names:
        model = fit_component(name, x_train, y)
        models.append(model); fits += int(getattr(model, 'classifier_fits', 1))
    metadata = {'method': 'training_only_grouped_conditional_stacking', 'training_rows': len(y),
        'training_features_sha256': fingerprint(x_train), 'training_labels_sha256': fingerprint(y.tolist()),
        'oof_seed': OOF_SEED, 'oof_folds': records, 'oof_margins': margins.tolist(),
        'coverage_counts': covered.tolist(), 'component_names': list(COMPONENTS),
        'complementarity_gate': gate, 'fusion_head_fitted': head is not None,
        'head_C': .1, 'head_fit_intercept': False, 'meta_feature_scaler_fit_on': 'training_OOF_margins_and_reversed_margins',
        'final_components': names, 'classifier_fits': fits, 'outer_search_labels_accessed': False,
        'oof_models_retained_for_verification': True, 'exact_tie_break_class': 0}
    model = LearnedPairStack(models, names, head, metadata, folds, fits)
    return model, model.predict(x_search)


def trainable_candidates(kind):
    if kind != 'pair_text': return []
    from .models import Candidate
    neural = [Candidate(f'neural_pair_dim{d}', {'dimension': d, 'max_iterations': MAX_EPOCHS,
        'representation': 'trainable_word_bag_pair_interactions', 'classifier': 'random_initialized_neural_network',
        'pretrained_weights': False, 'classifier_fits_per_evaluation': 2, 'symmetric_probability': True},
        partial(fit_neural_pair, dimension=d)) for d in (32, 64)]
    fusion = Candidate('learned_stack_C0.1', {'C': .1, 'representation': 'grouped_OOF_component_margins',
        'classifier': 'conditional_logistic_fusion', 'maximum_classifier_fits_per_evaluation': 17,
        'oof_folds': 3, 'symmetric_probability': True}, fit_learned_stack)
    return [*neural, fusion]
