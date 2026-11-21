"""Bound a planner's early menu to unseen method routes without inventing actions."""


def project_routes(state, capabilities, eligible_ids):
    eligible = list(eligible_ids)
    if not eligible or len(eligible) != len(set(eligible)):
        raise ValueError('Coverage needs nonempty distinct eligible candidate IDs')
    available = {row['id'] for row in state['available_candidates']}
    if not set(eligible) <= available or not set(eligible) <= set(capabilities):
        raise ValueError('Coverage cannot expand the trusted eligible menu')
    routes = {name: cap.get('route') for name, cap in capabilities.items()}
    if any(not isinstance(route, str) or not route for route in routes.values()):
        raise ValueError('Coverage requires explicit trusted method routes')
    seen = set()
    for row in state['history']:
        if row.get('status') == 'completed' and row.get('split_id') == 'primary':
            name = row['candidate']
            if name not in capabilities:
                raise ValueError('Observed candidate is not in the trusted catalogue')
            seen.add(routes[name])
    target = min(4, len(set(routes.values())))
    selected = eligible
    reason = 'coverage_target_met'
    if not seen:
        reason = 'shared_anchor_unchanged'
    elif len(seen) < target:
        unseen = [name for name in eligible if routes[name] not in seen]
        if unseen:
            selected, reason = unseen, 'prefer_unseen_feasible_routes'
        else:
            reason = 'no_unseen_route_in_feasible_menu'
    return tuple(selected), dict(target_routes=target, observed_primary_routes=sorted(seen),
        original_eligible_ids=eligible, projected_eligible_ids=selected,
        projected_routes=sorted({routes[name] for name in selected}), reason=reason)
