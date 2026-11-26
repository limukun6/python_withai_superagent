"""A-owned equal-evidence finalists; the B scheduler and default remain unchanged."""
from copy import deepcopy

from .ab_scheduler import (PRIMARY, CONFIRMATIONS, _action, _allowance,
                           next_action, selection_evidence)
from .metrics import score_of


def _validated(state, capabilities, trusted_partitions):
    budget = state['budget']
    if not 5 <= budget['fit_limit'] <= 11 or budget['reserved_final_refit'] != 1:
        raise ValueError('Matched finalists require 5..11 fits including one reserved refit')
    if not capabilities or any(c['classifier_fits'] != 1 or c['refit_classifier_fits'] != 1
                               for c in capabilities.values()):
        raise ValueError('Matched finalists require known single-fit executable capabilities')
    if state.get('registered_confirmation_splits') != list(CONFIRMATIONS):
        raise ValueError('Matched finalists require the registered confirmation partitions')
    if any(r.get('status') != 'completed' for r in state['history']):
        raise ValueError('Matched finalists cannot continue after incomplete execution')
    # Reuse full identity/metric/row/payload verification, not its incumbent policy.
    selection_evidence(state, trusted_partitions=trusted_partitions)
    if {r['trial'] for r in state['history']} != {p['trial'] for p in state['prediction_evidence']}:
        raise ValueError('Every finalist observation needs bound prediction evidence')
    for row in state['history']:
        cap = capabilities[row['candidate']]
        if row['params'] != cap['params'] or row.get('classifier_fits') != 1:
            raise ValueError('Finalist history differs from registered configuration or fit cost')
    primary = sorted((r for r in state['history'] if r['split_id'] == PRIMARY),
                     key=lambda r: (-score_of(r), r['candidate']))
    return primary, min(len(capabilities), budget['fit_limit'] - 3)


def finalist_action(state, *, capabilities, trusted_partitions, on_explore=None):
    primary, target = _validated(state, capabilities, trusted_partitions)
    available, reason = _allowance(state['budget'])
    if reason:
        return _action('stop', reason)
    if len(primary) < target:
        projected = deepcopy(state)
        projected['registered_confirmation_splits'] = []
        # This preserves common eligibility/ordering, while reserving both checks and refit.
        action = next_action(projected, capabilities=capabilities,
                             trusted_partitions={PRIMARY: trusted_partitions[PRIMARY]}, on_explore=on_explore)
        if action['action'] != 'explore':
            raise ValueError('Matched finalist primary allocation cannot finish: ' + action['reason'])
        action['reason'] = 'Reserve two matched finalist checks and a full refit. ' + action['reason']
        return action
    shortlist = primary[:2]
    existing = {(r['candidate'], r['split_id']) for r in state['history']}
    pending = [r for r in shortlist if (r['candidate'], CONFIRMATIONS[0]) not in existing]
    if pending:
        if len(pending) + 1 > available:
            raise ValueError('Not enough allowance to finish both finalist checks and refit')
        row = pending[0]
        return _action('confirm', 'Primary top-two shortlist; give every finalist the same confirmation split.',
                       [r['trial'] for r in shortlist], row['candidate'], CONFIRMATIONS[0], 1)
    return _action('stop', 'All finalists have matched primary and confirmation evidence; preserve final refit.',
                   [r['trial'] for r in state['history']])


def finalist_review(state, *, capabilities, trusted_partitions):
    primary, target = _validated(state, capabilities, trusted_partitions)
    if not primary or len(primary) != target:
        raise ValueError('Matched finalist primary allocation is incomplete')
    shortlist = primary[:2]
    lookup = {(r['candidate'], r['split_id']): r for r in state['history']}
    basis = [PRIMARY, CONFIRMATIONS[0]]
    means = {}
    for row in shortlist:
        name = row['candidate']
        if any((name, split) not in lookup for split in basis):
            raise ValueError('Incomplete matched finalist confirmation; cannot certify final selection')
        means[name] = sum(score_of(lookup[name, split]) for split in basis) / len(basis)
    chosen = min(means, key=lambda name: (-means[name], name))
    return dict(initial=primary[0]['candidate'], incumbent=chosen, promotions=[], basis=basis,
                finalists=[r['candidate'] for r in shortlist], mean_scores=means,
                primary_metric=state['task']['metric']['name'],
                selection_rule='Primary top two, equal primary/confirmation mean; ties by candidate id')
