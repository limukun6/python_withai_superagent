"""Choose an optional inference shortcut from two confirmed binary preference encoders."""
from dataclasses import dataclass

import numpy as np

from .ab_complement import PreferenceProbabilityVote, canonical_probabilities, split_columns
from .metrics import MetricSpec


THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 1.0)


def _hard(probability, tie_left):
    return ((probability > 0.5) | ((probability == 0.5) & tie_left)).astype(int)


def _measure(fold, first, second, threshold):
    scores, labels, ties = fold
    base = scores[first]
    dual = (base + scores[second]) / 2
    skip = np.abs(base - 0.5) >= threshold
    routed = _hard(np.where(skip, base, dual), ties)
    return dict(threshold=threshold, rows=len(labels), skips=int(skip.sum()),
                correct=int((routed == labels).sum()), full_correct=int((_hard(dual, ties) == labels).sum()))


def select_threshold(primary, fresh):
    scores, labels, ties = primary
    names = sorted(scores)
    first_correct = {name: int((_hard(scores[name], ties) == labels).sum()) for name in names}
    first = max(names, key=first_correct.get)
    second = next(name for name in names if name != first)
    grid = [_measure(primary, first, second, threshold) for threshold in THRESHOLDS]
    eligible = [record for record in grid if record['correct'] >= record['full_correct']]
    choice = max(eligible, key=lambda record: (record['skips'], record['threshold']))
    check = _measure(fresh, first, second, choice['threshold'])
    accepted = check['correct'] >= check['full_correct'] and check['skips'] >= 0.1 * check['rows']
    return dict(first=first, second=second, first_correct=first_correct, primary=grid, proposed=choice['threshold'],
                confirmation=check, adopted=accepted, threshold=choice['threshold'] if accepted else 1.0)


def assess(matched, state, trusted, capabilities):
    if not matched['promoted']:
        return dict(status='single_selected', adopted=False)
    members = matched['members']
    if len(members) != 2 or any(capabilities[name].get('resource') != 'local_frozen_encoder_cpu' for name in members):
        return dict(status='pair_without_two_frozen_encoders', adopted=False)
    main, confirm = 'primary', 'confirmation_1'
    seen = set(trusted[main]['groups'])
    fresh = np.array([group not in seen for group in trusted[confirm]['groups']], dtype=bool)
    if not fresh.any():
        return dict(status='no_fresh_confirmation_groups', adopted=False)
    metric = MetricSpec('accuracy', (0, 1))

    def fold(split, mask):
        labels, ties, records = split_columns(state, trusted, split, metric)
        return {name: records[name][1][mask] for name in members}, labels[mask], ties[mask]

    result = select_threshold(fold(main, slice(None)), fold(confirm, fresh))
    return dict(status='accepted' if result['adopted'] else 'retained_full_vote',
                confirmation_rows_before_filter=len(fresh), **result)


@dataclass
class ConfidenceRoutedPreferenceVote(PreferenceProbabilityVote):
    first_index: int
    threshold: float

    def predict_proba(self, inputs):
        rows = list(inputs)
        if not rows:
            return np.empty((0, 2), dtype=float)
        ordered = [tuple(sorted(pair)) for pair in rows]
        first = canonical_probabilities(self.models[self.first_index], ordered)[1][:, 1]
        needed = np.abs(first - 0.5) < self.threshold
        probability = first.copy()
        if needed.any():
            subset = [pair for pair, use in zip(ordered, needed) if use]
            second = canonical_probabilities(self.models[1 - self.first_index], subset)[1][:, 1]
            probability[needed] = (first[needed] + second) / 2
        original_order = np.array([a < b for a, b in rows], dtype=bool)
        identical = np.array([a == b for a, b in rows], dtype=bool)
        positive = np.where(identical, 0.5, np.where(original_order, probability, 1 - probability))
        return np.column_stack((1 - positive, positive))


def routed_model(vote, members, decision):
    return ConfidenceRoutedPreferenceVote(vote.kind, vote.models, vote.classes_,
                                         members.index(decision['first']), decision['threshold'])
