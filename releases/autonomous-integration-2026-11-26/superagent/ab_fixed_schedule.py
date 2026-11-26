"""A-owned fixed allocation control; shares B's exploration and final selection."""
from copy import deepcopy

from .ab_scheduler import next_action, selection_evidence, PRIMARY, CONFIRMATIONS
from .metrics import score_of


def fixed_action(state, *, capabilities, trusted_partitions, on_explore=None):
    budget = state['budget']
    if budget['fit_limit'] != 11 or budget['reserved_final_refit'] != 1:
        raise ValueError('AB fixed control requires total 11-fit budget and one reserved refit')
    if any(c['classifier_fits'] != 1 or c['refit_classifier_fits'] != 1 for c in capabilities.values()):
        raise ValueError('AB fixed control requires single-fit registered candidates')
    # Validate the full state/evidence before choosing any follow-up training.
    if state['history']:
        if any(r.get('status') != 'completed' for r in state['history']):
            raise ValueError('Fixed control cannot continue after incomplete execution')
        selection_evidence(state, trusted_partitions=trusted_partitions)
    primary = [r for r in state['history'] if r.get('split_id', PRIMARY) == PRIMARY]
    target = min(4, len(capabilities))
    if len(primary) < target:
        projected = deepcopy(state)
        projected['registered_confirmation_splits'] = []
        action = next_action(projected, capabilities=capabilities,
                             trusted_partitions={PRIMARY: trusted_partitions[PRIMARY]}, on_explore=on_explore)
        if action['action'] != 'explore':
            raise ValueError('Fixed primary allocation cannot be completed: '+action['reason'])
        action['reason'] = 'Fixed allocation primary stage. ' + action['reason']
        return action
    ranked = sorted(primary, key=lambda r: (-score_of(r), r['candidate']))[:3]
    intended = [(r, split) for split in CONFIRMATIONS for r in ranked]
    existing = {(r['candidate'], r['split_id']) for r in state['history']}
    pending = [(row, split) for row, split in intended if (row['candidate'], split) not in existing]
    if pending:
        # Call B's common budget guard; it checks unknown costs, counters and wall time.
        from .ab_scheduler import _allowance
        available, reason = _allowance(budget)
        if reason or len(pending)+1 > available:
            raise ValueError('Fixed confirmation block cannot complete with refit: '+str(reason))
        row, split = pending[0]
        return dict(action='confirm', candidate_id=row['candidate'], split_id=split,
                    reason='Fixed primary top-three shortlist; complete both registered replication rounds.',
                    evidence_trials=[r['trial'] for r in ranked], estimated_classifier_fits=1)
    return dict(action='stop', candidate_id=None, split_id=None, estimated_classifier_fits=0,
                evidence_trials=[r['trial'] for r in ranked],
                reason='Fixed primary and top-three replication allocation completed; preserve final refit.')
