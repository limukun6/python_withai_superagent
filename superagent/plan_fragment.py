"""Short plan fragments that the scheduler can still overrule.

A fragment is at most two structured steps that a planner submits alongside its
immediate choice. It never replaces the scheduler: before any step runs, the
fragment is checked against the current state, and a `confirm` or `stop`
decision from the scheduler always wins.

Every step carries only machine-checkable preconditions drawn from recorded
facts (a completed trial, which trial is currently best, a minimum best score).
Free text is rejected on purpose: an unverifiable condition cannot invalidate a
plan, and a plan that cannot be invalidated would silently pre-commit budget.

Nothing here adds fits, changes eligibility, or edits the candidate catalogue.
"""
from copy import deepcopy

MAX_STEPS = 2
CONDITIONS = ('reference_trial', 'best_completed_trial', 'minimum_best_score')


def _completed(history):
    return [row for row in history if row.get('status') == 'completed']


def _score(row):
    for key in ('primary_score', 'search_score', 'search_accuracy'):
        value = row.get(key)
        if type(value) in (int, float):
            return float(value)
    return None


def _best(history):
    scored = [row for row in _completed(history) if _score(row) is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: (_score(row), -row['trial']))


def validate_fragment(fragment, available_ids):
    """Parse a proposed fragment; return (normalised, errors)."""
    errors = []
    if not isinstance(fragment, dict) or set(fragment) != {'steps'}:
        return None, ['A fragment must be an object with exactly a steps list']
    steps = fragment['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        return None, [f'A fragment needs 1..{MAX_STEPS} steps']
    seen, normalised = set(), []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or set(step) != {'candidate', 'requires'}:
            errors.append(f'Step {index} needs exactly candidate and requires')
            continue
        candidate, requires = step['candidate'], step['requires']
        if not isinstance(candidate, str) or candidate not in available_ids:
            errors.append(f'Step {index} candidate is not in the eligible menu')
        elif candidate in seen:
            errors.append(f'Step {index} repeats an earlier candidate')
        seen.add(candidate)
        if not isinstance(requires, dict) or not requires:
            errors.append(f'Step {index} requires a nonempty structured condition set')
            continue
        unknown = sorted(set(requires) - set(CONDITIONS))
        if unknown:
            errors.append(f'Step {index} uses unsupported conditions {unknown}')
            continue
        for key, value in requires.items():
            if key in ('reference_trial', 'best_completed_trial') and type(value) is not int:
                errors.append(f'Step {index} {key} must be an integer trial id')
            if key == 'minimum_best_score' and not isinstance(value, (int, float)):
                errors.append(f'Step {index} minimum_best_score must be a number')
        normalised.append({'step': index, 'candidate': candidate, 'requires': deepcopy(requires)})
    if errors:
        return None, errors
    return {'steps': normalised, 'maximum_steps': MAX_STEPS}, []


def check_conditions(requires, history):
    """Return per-condition results for one step against the recorded history."""
    results = []
    completed = {row['trial']: row for row in _completed(history)}
    best = _best(history)
    for key in CONDITIONS:
        if key not in requires:
            continue
        wanted = requires[key]
        if key == 'reference_trial':
            status = 'holds' if wanted in completed else 'failed'
            observed = sorted(completed)[-1] if completed else None
            detail = 'referenced trial must already be completed'
        elif key == 'best_completed_trial':
            observed = best['trial'] if best else None
            status = 'holds' if observed == wanted else 'failed'
            detail = 'the currently best completed trial must still be this one'
        else:
            observed = round(_score(best), 6) if best and _score(best) is not None else None
            status = 'holds' if observed is not None and observed >= wanted else 'failed'
            detail = 'the best completed score must stay at or above this floor'
        results.append({'condition': key, 'required': wanted, 'observed': observed,
                        'status': status, 'detail': detail})
    return results


def fragment_status(fragment, state, scheduler_action, step_index=0):
    """Decide whether the fragment's next step may run.

    `scheduler_action` is the action dict the B scheduler returned for this
    decision; anything that is not an explore supersedes the fragment.
    """
    action = (scheduler_action or {}).get('action')
    history = state.get('history') or []
    available = {c['id'] for c in state.get('available_candidates') or []}
    steps = (fragment or {}).get('steps') or []
    record = {'step_index': step_index, 'scheduler_action': action,
              'fragment_steps': len(steps), 'checks': []}
    if action != 'explore':
        return {**record, 'status': 'superseded_by_scheduler',
                'reason': f'scheduler action is {action!r}; a fragment may not override it',
                'candidate': None}
    if step_index < 0 or step_index >= len(steps):
        return {**record, 'status': 'consumed', 'reason': 'no planned step remains', 'candidate': None}
    step = steps[step_index]
    if step['candidate'] not in available:
        return {**record, 'status': 'invalidated',
                'reason': 'planned candidate is no longer eligible',
                'candidate': None, 'checks': []}
    checks = check_conditions(step['requires'], history)
    failed = [c for c in checks if c['status'] != 'holds']
    record['checks'] = checks
    if failed:
        return {**record, 'status': 'invalidated',
                'reason': '; '.join(f"{c['condition']} expected {c['required']} observed {c['observed']}"
                                    for c in failed),
                'candidate': None}
    return {**record, 'status': 'valid', 'reason': 'all recorded conditions still hold',
            'candidate': step['candidate']}


def next_step(fragment, state, scheduler_action, step_index=0):
    """Convenience wrapper: the candidate to explore, or None."""
    return fragment_status(fragment, state, scheduler_action, step_index)['candidate']
