"""Train-only calibration and portable prediction with two frozen text encoders."""
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import brentq

from .method_choice import TEXT_DEEP, TEXT_MINILM
from .minilm_features import MiniLMTextClassifier
from .splits import sample_groups


MIXTURE = 'minilm_confirmed_probability_mixture'


def fit_weight(inner, ids, truth, groups):
    """Overlapping inner splits give equal total weight to each distinct row."""
    anchor, other = np.concatenate(inner).T
    ids = np.concatenate(ids)
    rows, inverse, visits = np.unique(ids, return_inverse=True, return_counts=True)
    labels = truth[ids]
    anchor_true = np.where(labels == 1, anchor, 1 - anchor)
    other_true = np.where(labels == 1, other, 1 - other)
    difference = anchor_true - other_true
    weights = 1 / (len(rows) * visits[inverse])

    def slope(weight):
        probability = np.maximum(other_true + weight * difference, 1e-15)
        return -float(np.dot(weights, difference / probability))

    if slope(1) <= 0:
        raw = 1.
    elif slope(0) >= 0:
        raw = 0.
    else:
        raw = float(brentq(slope, 0., 1.))

    gain = np.log(np.maximum(other_true + raw * difference, 1e-15)) - np.log(np.maximum(anchor_true, 1e-15))
    per_row = np.bincount(inverse, weights=gain) / visits
    mean = float(per_row.mean())
    _, cluster = np.unique(groups[rows], return_inverse=True)
    sizes = np.bincount(cluster)
    sums = np.bincount(cluster, weights=per_row) - mean * sizes
    se = None
    if len(sizes) > 1:
        se = float(np.sqrt(len(sizes) / (len(sizes) - 1) * np.sum(sums**2)) / len(rows))

    correct = (raw * anchor + (1 - raw) * other > .5) == labels
    anchor_correct = (anchor > .5) == labels
    accuracy_gain = float(np.dot(weights, correct.astype(int) - anchor_correct))
    supported = se is not None and mean >= max(.002, se) and accuracy_gain >= -.002
    return dict(raw=raw, guarded=raw if supported else 1., unique_rows=len(rows), visits=len(ids),
                groups=len(sizes), loss_gain=mean, group_standard_error=se, accuracy_gain=accuracy_gain,
                inner_accuracy=float(np.dot(weights, correct)))


def learn_from_confirmations(task, history, confirmation, directory):
    winner = confirmation.get('selected_candidate')
    names = {TEXT_MINILM, TEXT_DEEP}
    classes = np.unique(task.y_train)
    if (not confirmation.get('valid_confirmation') or winner not in names or len(classes) != 2
            or confirmation['primary_metric'] != 'accuracy' or not names <= set(confirmation['shortlist'])):
        return dict(eligible=False, reason='Both encoders must finish all three binary accuracy checks and one must win.')

    other = next(iter(names - {winner}))
    numbers = {row['candidate']: row['trial'] for row in history}
    indices = []
    probabilities = []
    for index in range(3):
        split_name = 'split.json' if index == 0 else f'confirmation_split_{index}.json'
        ids = np.asarray(json.loads((directory / split_name).read_text())['search_indices'])
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(task.y_train)):
            raise ValueError('Invalid calibration row indices')
        observed = []
        for name in (winner, other):
            number = numbers[name]
            file = f'trial_{number}.joblib' if index == 0 else f'confirmation_{index}_{number}.joblib'
            trial = joblib.load(directory / file)
            probs = trial['search_scores']
            if (not np.array_equal(trial['score_classes'], classes) or probs.shape != (len(ids), 2)
                    or not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1)
                    or not np.array_equal(classes[probs.argmax(axis=1)], trial['search_predictions'])):
                raise ValueError('Encoder probabilities differ from confirmed decisions')
            observed.append(probs[:, 1])
        indices.append(ids)
        probabilities.append(np.column_stack(observed))

    groups = sample_groups(task, 'shared_text')
    binary_truth = (np.asarray(task.y_train) == classes[1]).astype(int)
    calibrated = fit_weight(probabilities, indices, binary_truth, groups)
    return dict(eligible=True, anchor=winner, other=other, other_trial=numbers[other], **calibrated)


class EncoderMixture:
    def __init__(self, anchor, other, weight):
        if (not 0 <= weight <= 1 or not np.array_equal(anchor.classes_, other.classes_)
                or anchor.weight_setting == other.weight_setting):
            raise ValueError('Incompatible frozen classifiers or invalid mixture weight')
        self.anchor = anchor
        self.other = other
        self.weight = weight
        self.classes_ = anchor.classes_

    def predict_proba(self, rows):
        if self.weight == 0:
            return self.other.predict_proba(rows)
        if self.weight == 1:
            return self.anchor.predict_proba(rows)
        return self.weight * self.anchor.predict_proba(rows) + (1 - self.weight) * self.other.predict_proba(rows)

    def predict(self, rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]


def bind_text_prediction(model, settings, directory):
    if isinstance(model, MiniLMTextClassifier):
        if not settings.get(model.weight_setting):
            raise ValueError('Supply the pinned MiniLM weights for this frozen text classifier')
        model.bind_prediction_weights(settings[model.weight_setting], directory)
    elif isinstance(model, EncoderMixture):
        parts = (model.anchor, model.other)
        if any(not settings.get(part.weight_setting) for part in parts):
            raise ValueError('Supply the pinned L6 and L12 directories for this encoder mixture')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        runtime = directory / 'pretrained_runtime.json'
        config = {part.weight_setting: str(Path(settings[part.weight_setting]).resolve()) for part in parts}
        if runtime.exists() and json.loads(runtime.read_text()) != config:
            raise ValueError('Prediction cache belongs to different encoder weights')
        if not runtime.exists():
            runtime.write_text(json.dumps(config))
        for part in parts:
            part.runtime = str(runtime)
            part.encoder = None
