"""Choose whether to explore one more preference method using paired group uncertainty."""
from copy import deepcopy
from itertools import combinations
import math

import numpy as np

from . import ab_complement
from .ab_scheduler import _action, CONFIRMATIONS, PRIMARY, next_action, selection_evidence
from .metrics import MetricSpec


def opening_uncertainty(records, y, groups):
    order = sorted(records, key=lambda n: -int((records[n][0] == y).sum()))
    first, second = order[:2]
    d = (records[first][0] == y).astype(float) - (records[second][0] == y)
    _, ids = np.unique(groups, return_inverse=True)
    sizes = np.bincount(ids)
    sums = np.bincount(ids, weights=d)
    se = None
    if len(sizes) > 1:
        residuals = sums - d.mean() * sizes
        se = math.sqrt(len(sizes) / (len(sizes) - 1) * float(residuals @ residuals)) / len(d)
    return dict(leader=first, runner_up=second, gap=float(d.mean()), cluster_standard_error=se,
                groups=len(sizes), disagreements=int(np.count_nonzero(d)),
                ambiguous=bool(se is not None and np.any(d) and d.mean() <= 1.96 * se))


def review(state, trusted):
    selection_evidence(state, trusted_partitions=trusted)
    if len(state['prediction_evidence']) != len(state['history']):
        raise ValueError('Incomplete primary or confirmation records')
    metric = MetricSpec.from_payload({'metric_spec':state['task']['metric']})
    y, tie, opening = ab_complement.split_columns(state, trusted, PRIMARY, metric)
    if len(opening) not in (4, 5):
        raise ValueError('Expected four or five opening methods')
    counts = {n:int((r[0] == y).sum()) for n, r in opening.items()}
    anchor = max(counts, key=counts.get)
    pairs = {pair:ab_complement.average_hits(opening[pair[0]], opening[pair[1]], tie, y)
             for pair in combinations(opening, 2)}
    winning = max(pairs, key=pairs.get)
    selected = sorted(winning, key=lambda n: -counts[n])
    uncertainty = opening_uncertainty(dict(list(opening.items())[:4]), y, trusted[PRIMARY]['groups'])
    result = dict(anchor=anchor, members=[anchor], partner=None, primary_rows=len(y), opening_correct=counts,
                  opening_combinations=[dict(members=list(k), correct=v) for k, v in pairs.items()],
                  primary_single_accuracy=counts[anchor]/len(y), portfolio_score=counts[anchor]/len(y),
                  proposed_pair=selected, comparison_members=[], evidence_trials=[r[2] for r in opening.values()],
                  opening_uncertainty=uncertainty, confirmation_rows=0, promoted=False)
    if (pairs[winning] - counts[anchor]) / len(y) < ab_complement.MIN_GAIN:
        return result
    comparators = list(dict.fromkeys([*selected, anchor]))
    second_y, second_tie, second = ab_complement.split_columns(state, trusted, CONFIRMATIONS[0], metric)
    result.update(comparison_members=comparators, partner=selected[1], pending=[n for n in comparators if n not in second])
    result['evidence_trials'].extend(r[2] for r in second.values())
    if result['pending']:
        return result
    other_counts = {n:int((second[n][0] == second_y).sum()) for n in comparators}
    combination = ab_complement.average_hits(second[selected[0]], second[selected[1]], second_tie, second_y)
    pooled = pairs[winning] + combination - max(counts[n] + other_counts[n] for n in comparators)
    total = len(y) + len(second_y)
    promoted = combination > max(other_counts.values()) and pooled / total >= ab_complement.MIN_GAIN
    result.update(confirmation_rows=len(second_y), confirmation_correct=other_counts,
                  confirmation_combination=combination, pooled_gain=pooled, promoted=promoted)
    if promoted:
        result.update(members=selected, portfolio_score=(pairs[winning] + combination)/total)
    return result


def unseen_encoder_candidates(state, capabilities):
    def encoder(name):
        record = capabilities[name]
        params = record.get('params', {})
        if (record.get('resource') == 'local_frozen_encoder_cpu' and params.get('revision')
                and record.get('supports_probabilities') and record.get('classifier_fits') == record.get('refit_classifier_fits') == 1):
            return params.get('encoder')
        return None

    used = {encoder(row['candidate']) for row in state['history'] if row['split_id'] == PRIMARY}
    return [entry for entry in state['available_candidates']
            if (identity := encoder(entry['id'])) and identity not in used]


def action(state, capabilities, trusted, *, cover_untried_encoder=False):
    if len(state['history']) < 4:
        return ab_complement.action(state, capabilities, trusted)
    result = review(state, trusted)
    budget = state['budget']['remaining_fit_allowance']
    if len(state['history']) == 4 and budget >= 6:
        unseen = unseen_encoder_candidates(state, capabilities) if cover_untried_encoder else []
        if unseen or result['opening_uncertainty']['ambiguous']:
            projected = deepcopy(state)
            projected['registered_confirmation_splits'] = []
            if unseen:
                projected['available_candidates'] = unseen
            proposed = next_action(projected, capabilities=capabilities, trusted_partitions={PRIMARY:trusted[PRIMARY]})
            if proposed['action'] == 'explore':
                proposed['reason'] = ('An eligible frozen encoder has no observed trial; the remaining budget covers exploration and matched comparison.'
                    if unseen else 'Leading methods disagree and their grouped paired gap is uncertain; explore one additional eligible route.')
                proposed['evidence_trials'] = result['evidence_trials']
                return proposed
    pending = result.get('pending', [])
    if pending:
        if len(pending) + 2 > budget:
            raise ValueError('Insufficient fits for matched comparators and both potential refits')
        return _action('confirm', 'Check the proposed pair and the strongest single on the same confirmation rows.',
                       result['evidence_trials'], pending[0], CONFIRMATIONS[0], 1)
    return _action('stop', 'Finish the paired comparison and deliver the selected model or pair.', result['evidence_trials'])
