"""Resolve discordant single-model evidence after an adaptive preference vote is declined."""
from collections import defaultdict

import numpy as np

from . import ab_adaptive_opening, ab_complement
from .ab_scheduler import CONFIRMATIONS, PRIMARY, _action
from .metrics import MetricSpec


def reconcile(panels):
    """Give each original row one vote and report the last panel's unseen groups separately."""
    totals = defaultdict(lambda: np.zeros(3, dtype=float))
    identities = {}
    known_groups = set()
    fresh_groups = set()
    fresh_delta = fresh_rows = 0
    for index, (ids, groups, y, anchor, challenger) in enumerate(panels):
        if not len(ids) == len(groups) == len(y) == len(anchor) == len(challenger) or len(set(ids)) != len(ids):
            raise ValueError('Single-model comparison needs aligned, unique row IDs per split')
        for rid, group, label, pred_a, pred_b in zip(ids, groups, y, anchor, challenger, strict=True):
            if identities.setdefault(rid, (group, label)) != (group, label):
                raise ValueError('A shared evaluation row has an inconsistent group or label')
            a, b = int(pred_a == label), int(pred_b == label)
            totals[rid] += (a, b, 1)
            if index == 2 and group not in known_groups:
                fresh_groups.add(group)
                fresh_delta += b-a
                fresh_rows += 1
        if index < 2:
            known_groups.update(groups)
    values = np.asarray(list(totals.values()))
    acc = (values[:, :2]/values[:, 2, None]).mean(axis=0)
    return dict(unique_rows=len(totals), row_observations=int(values[:,2].sum()),
                fresh_groups=len(fresh_groups), fresh_rows=fresh_rows, fresh_net_correct=fresh_delta,
                anchor_score=float(acc[0]), challenger_score=float(acc[1]), unique_gain=float(acc[1]-acc[0]))


def review(state, trusted):
    result = ab_adaptive_opening.review(state, trusted)
    counts = result.get('confirmation_correct')
    if result['promoted'] or not counts or CONFIRMATIONS[1] not in state['registered_confirmation_splits']:
        return result
    anchor = result['anchor']
    opening = result['opening_correct']
    ahead = [name for name in opening if name in counts and counts[name] > counts[anchor]]
    if not ahead:
        return result
    rival = max(ahead, key=lambda name: (counts[name], opening[name]))
    metric = MetricSpec.from_payload({'metric_spec':state['task']['metric']})
    third_y, _, third = ab_complement.split_columns(state, trusted, CONFIRMATIONS[1], metric)
    missing = [name for name in (anchor,rival) if name not in third]
    result['evidence_trials'].extend(third[n][2] for n in (anchor,rival) if n in third)
    check = dict(anchor=anchor, challenger=rival, missing=missing)
    result['single_conflict'] = check
    if missing:
        left = state['budget']['remaining_fit_allowance']
        reserve = state['budget']['reserved_final_refit']
        check['status'] = 'collect' if left >= len(missing) + reserve else 'retain_budget'
        if check['status'] == 'collect':
            result['pending'] = missing
        return result
    panels = []
    for split in (PRIMARY, *CONFIRMATIONS):
        if split == CONFIRMATIONS[1]:
            y, cols = third_y, third
        else:
            y, _, cols = ab_complement.split_columns(state, trusted, split, metric)
        panels.append((trusted[split]['row_ids'], trusted[split]['groups'], y, cols[anchor][0], cols[rival][0]))
    stats = reconcile(panels)
    switch = stats['fresh_net_correct'] > 0 and stats['unique_gain'] >= ab_complement.MIN_GAIN
    check.update(**stats, status='switch' if switch else 'retain')
    result['members'] = [rival if switch else anchor]
    result['portfolio_score'] = stats['challenger_score'] if switch else stats['anchor_score']
    return result


def action(state, capabilities, trusted, *, cover_untried_encoder=False):
    original = ab_adaptive_opening.action(state, capabilities, trusted, cover_untried_encoder=cover_untried_encoder)
    if original['action'] != 'stop':
        return original
    evidence = review(state, trusted)
    check = evidence.get('single_conflict')
    if not check:
        return original
    if check['status'] == 'collect':
        return _action('confirm', 'The opening and first confirmation prefer different single models. Compare both on the remaining split.',
                       evidence['evidence_trials'], check['missing'][0], CONFIRMATIONS[1], 1)
    reason = {'switch':'The challenger improved on new groups and on distinct rows across the three splits.',
              'retain':'The third comparison did not support replacing the opening leader.',
              'retain_budget':'Retain the opening leader with the remaining fit and refit allowance.'}[check['status']]
    return _action('stop', reason, evidence['evidence_trials'])
