"""Declared pair semantics, the augmentation rule they imply, and a swap probe.

Two text columns can mean three different tasks, and only one of them permits
training on swapped copies with flipped labels:

    preference  left is preferred over right; swapping flips the 0/1 label
    relation    a multi-class relation between two texts; swap behaviour is not
                implied by the label coding, so it stays unverified
    symmetric   the two texts are interchangeable and the label is preserved

The word/char preference candidates train on (left-right) and on its negation
with flipped labels, so they encode the preference assumption. Running them on
a symmetric task silently trains on wrong labels. This module states the
assumption, checks it against the loaded package, and measures what a fitted
model actually does under swap.
"""
import numpy as np
from functools import partial

SEMANTICS = ('preference', 'relation', 'symmetric')
SWAP_POLICY = {'preference': 'flips', 'relation': 'unverified', 'symmetric': 'preserves'}
PAIR_KINDS = ('pair_text', 'text_tabular')
# A pipeline is called complementary (flips) or symmetric (preserves) only when
# the mean absolute deviation is small; labels are 0/1 so the range is [0, 1].
SWAP_TOLERANCE = 0.10


def declared_semantics(task):
    spec = getattr(task, 'spec', None) or {}
    value = (spec.get('input_schema') or {}).get('pair_semantics')
    return value if isinstance(value, str) else None


def swap_augmented(candidates):
    return sorted(c['id'] for c in candidates if (c.get('params') or {}).get('swap_augmentation'))


def candidate_swap_rule(candidate):
    """Resolve known execution rules; absence of a flag is not evidence of safety."""
    from .recipe_models import fit_recipe
    from .ab_symmetric import fit_symmetric_pair
    function = candidate.fit_predict
    while isinstance(function, partial):
        function = function.func
    if function is fit_symmetric_pair:
        return 'preserves'
    if function is fit_recipe and candidate.params.get('kind') == 'pair_text':
        return 'flips'
    if candidate.params.get('swap_augmentation'):
        return 'flips'
    return 'unknown'


def validate_candidate_semantics(kind, declared, candidate):
    """Final execution guard, applied after resolving static or registered candidates."""
    if declared is not None and declared not in SEMANTICS:
        raise ValueError(f'Unknown pair semantics {declared!r}')
    rule = candidate_swap_rule(candidate) if kind == 'pair_text' else 'unknown'
    if kind == 'pair_text' and declared == 'symmetric' and rule != 'preserves':
        raise ValueError(f'Candidate {candidate.family} cannot serve symmetric semantics: swap rule {rule}')
    if kind == 'pair_text' and rule == 'preserves' and declared != 'symmetric':
        raise ValueError('An unordered pair candidate requires declared symmetric semantics')
    if kind == 'pair_text' and declared == 'relation':
        raise ValueError('Relation semantics requires the relation input adapter')
    return {'candidate': candidate.family, 'swap_label_rule': rule}


def validate_task_semantics(task, portfolio='expanded'):
    """Check the declared semantics against the loaded package, before training."""
    from .models import search_candidates
    kind = getattr(task, 'kind', None)
    declared = declared_semantics(task)
    candidates = [{'id': c.family, 'params': c.params,
                   'swap_label_rule': candidate_swap_rule(c) if kind == 'pair_text' else 'unknown'}
                  for c in search_candidates(kind, portfolio)]
    checks, reasons, excluded = [], [], []

    if declared is None:
        checks.append({'check': 'semantics_declared', 'status': 'undeclared',
                       'detail': 'No pair_semantics in the package; swap behaviour is unverified'})
    elif declared not in SEMANTICS:
        checks.append({'check': 'semantics_declared', 'status': 'invalid',
                       'detail': f'{declared!r} is not one of {list(SEMANTICS)}'})
        reasons.append(f'unknown pair semantics {declared!r}')
    else:
        checks.append({'check': 'semantics_declared', 'status': 'ok', 'detail': declared})

    expected_kind = {'preference': 'pair_text', 'symmetric': 'pair_text', 'relation': 'text_tabular'}
    if declared in expected_kind:
        want = expected_kind[declared]
        status = 'ok' if kind == want else 'contradicted'
        checks.append({'check': 'semantics_matches_package', 'status': status,
                       'detail': f'{declared} implies {want}, package loaded as {kind}'})
        if status == 'contradicted':
            reasons.append(f'{declared} semantics implies {want} but the package loaded as {kind}')

    policy = SWAP_POLICY.get(declared, 'unverified')
    if policy == 'preserves':
        dropped = sorted(c['id'] for c in candidates if c['swap_label_rule'] != 'preserves')
        if dropped:
            excluded = dropped
            checks.append({'check': 'swap_augmentation', 'status': 'filtered',
                           'detail': 'flipping or unknown execution rules are incompatible with '
                                     f'preserved labels; excluded {dropped}'})
        else:
            checks.append({'check': 'swap_augmentation', 'status': 'ok',
                           'detail': 'no candidate assumes flipped labels'})
    elif policy == 'flips':
        excluded = [c['id'] for c in candidates if c['swap_label_rule'] == 'preserves']
        checks.append({'check': 'swap_augmentation', 'status': 'ok',
                       'detail': 'label-flipping augmentation matches the declared semantics'})
    else:
        excluded = [c['id'] for c in candidates if c['swap_label_rule'] == 'preserves']
        checks.append({'check': 'swap_augmentation', 'status': 'unverified',
                       'detail': 'semantics does not imply a swap policy'})

    remaining = [c for c in candidates if c['id'] not in excluded]
    if kind in PAIR_KINDS and not remaining:
        reasons.append('every candidate assumes the other swap policy; nothing can be run safely')

    if reasons:
        status = 'rejected'
    elif declared is None:
        status = 'undeclared'
    else:
        status = 'accepted'
    return {'task_kind': kind, 'portfolio': portfolio, 'declared_semantics': declared,
            'swap_policy': policy, 'swap_tolerance': SWAP_TOLERANCE, 'status': status,
            'checks': checks, 'excluded_candidates': excluded, 'reasons': reasons,
            'remaining_candidates': len(remaining),
            'candidate_swap_rules': {c['id']:c['swap_label_rule'] for c in candidates},
            'scope': 'Declaration and static portfolio checks only; swap_behavior measures the '
                     'fitted pipeline separately'}


def swap_behavior(probabilities, swapped_probabilities):
    """Describe what a fitted binary pair model does when the two texts swap."""
    first = np.asarray(probabilities, dtype=float).reshape(-1)
    second = np.asarray(swapped_probabilities, dtype=float).reshape(-1)
    if first.shape != second.shape or first.size == 0:
        raise ValueError('Swap probe needs two equally sized nonempty probability vectors')
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError('Swap probe probabilities must be finite')
    if not (((first >= 0) & (first <= 1)).all() and ((second >= 0) & (second <= 1)).all()):
        raise ValueError('Swap probe probabilities must be in [0, 1]')
    complementarity = float(np.mean(np.abs(first + second - 1.0)))
    symmetry = float(np.mean(np.abs(first - second)))
    if complementarity <= SWAP_TOLERANCE and symmetry <= SWAP_TOLERANCE:
        pattern = 'ambiguous'
    elif complementarity <= SWAP_TOLERANCE:
        pattern = 'flips'
    elif symmetry <= SWAP_TOLERANCE:
        pattern = 'preserves'
    else:
        pattern = 'neither'
    return {'rows': int(first.size), 'pattern': pattern, 'tolerance': SWAP_TOLERANCE,
            'mean_complementarity_error': complementarity, 'mean_symmetry_error': symmetry}


def declaration_consistency(probe):
    if (not probe.get('applicable') or probe.get('expected_pattern') is None
            or probe.get('pattern') == 'ambiguous'):
        return None
    return probe.get('pattern') == probe['expected_pattern']


def probe_model_swap(kind, model, inputs, limit=400):
    """Measure swap behaviour on the first `limit` rows of an already fitted model."""
    if kind != 'pair_text':
        return {'applicable': False, 'reason': 'swap behaviour is defined for two-text-column packages'}
    from .models import probability_with_model
    rows = list(inputs)[:limit]
    if not rows:
        return {'applicable': False, 'reason': 'no inputs supplied'}
    swapped = [(right, left) for left, right in rows]
    classes, forward = probability_with_model(kind, model, rows)
    _, backward = probability_with_model(kind, model, swapped)
    classes = list(np.asarray(classes).reshape(-1))
    if 1 not in classes:
        return {'applicable': False, 'reason': f'positive class 1 absent from fitted classes {classes}'}
    column = classes.index(1)
    report = swap_behavior(forward[:, column], backward[:, column])
    return {'applicable': True, 'scope': 'inner search rows of the selected candidate before full refit',
            **report}


def expectations_for(declared_semantics_value):
    """The measured pattern a declared semantics predicts, when it predicts one."""
    policy = SWAP_POLICY.get(declared_semantics_value)
    if policy == 'flips':
        return {'expected_pattern': 'flips', 'policy': policy}
    if policy == 'preserves':
        return {'expected_pattern': 'preserves', 'policy': policy}
    return {'expected_pattern': None, 'policy': 'unverified'}
