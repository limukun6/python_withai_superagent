"""Use grouped search probabilities to choose one regularization experiment."""
import numpy as np

from .data import TrainingTask
from .splits import sample_groups


def regularization_evidence(probabilities, classes, labels, groups, initial_c):
    probabilities = np.asarray(probabilities, dtype=float)
    matches = np.asarray(labels)[:, None] == np.asarray(classes)[None, :]
    if (probabilities.shape != matches.shape or not np.all(matches.sum(axis=1) == 1)
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Classifier probabilities and search labels must align")
    log_probs = np.log(np.maximum(probabilities, np.finfo(float).tiny))
    # Derivative of cross entropy when every logit is multiplied by the same scale.
    gradients = (probabilities * log_probs).sum(axis=1) - log_probs[matches]
    mean = float(np.mean(gradients))
    _, codes = np.unique(groups, return_inverse=True)
    sizes = np.bincount(codes)
    if len(sizes) < 2:
        return dict(derivative=mean, group_count=len(sizes), standard_error=None, next_c=None)
    centered = np.bincount(codes, weights=gradients) - mean * sizes
    variance = len(sizes) / (len(sizes) - 1) * np.sum(centered**2) / len(gradients)**2
    error = float(np.sqrt(variance))
    next_c = None
    if abs(mean) > max(error, 1e-10):
        next_c = initial_c * 4 if mean < 0 else initial_c / 4
    return dict(derivative=mean, group_count=len(sizes), standard_error=error, next_c=next_c)


def observe_model(model, kind, rows, labels):
    groups = sample_groups(TrainingTask(kind, rows, labels), "shared_text")
    return regularization_evidence(model.predict_proba(rows), model.classes_, labels, groups, model.classifier.C)
