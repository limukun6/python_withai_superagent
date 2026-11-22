"""Stateless v1 action adapter. The executor owns training, accounting and final selection."""
import argparse
import json
import math
from pathlib import Path

from .ab_methods import ROUTES
from .ab_comparison import MIN_GAIN, MIN_PAIRED_WINS, incumbent_review, macro_f1_pair_spread
from .evidence_agent import route_of
from .metrics import MetricSpec, score_of
from .prediction_evidence import fingerprint


PRIMARY = 'primary'
CONFIRMATIONS = ('confirmation_1', 'confirmation_2')
ROW_SCOPE = 'positions within this run training input'


def _action(action, reason, evidence=(), candidate=None, split=None, cost=0):
    return dict(action=action, candidate_id=candidate, split_id=split, reason=reason,
                evidence_trials=list(dict.fromkeys(evidence)), estimated_classifier_fits=cost)


def _allowance(budget):
    fields = ('fit_limit', 'attempted_classifier_fits', 'completed_classifier_fits',
              'remaining_fit_allowance', 'reserved_final_refit', 'llm_request_limit', 'used_llm_requests')
    if budget.get('unknown_failed_attempts') != 0 or any(type(budget.get(k)) is not int for k in fields):
        return 0, 'Unknown usage; reconcile the existing task ledger before any new fit.'
    limit, attempted, completed, left, reserve, llm_limit, used = [budget[k] for k in fields]
    if not (0 <= completed <= attempted <= limit <= 11 and left == limit - attempted
            and 0 <= reserve <= limit and 0 <= used <= llm_limit <= 3):
        return 0, 'Budget counters or caps disagree; check the task ledger.'
    elapsed = budget.get('elapsed_seconds')
    wall = budget.get('time_limit_seconds')
    if type(elapsed) not in (float,int) or not math.isfinite(elapsed) or elapsed < 0:
        return 0, 'Elapsed time is unknown.'
    if wall is not None and (type(wall) not in (float,int) or not math.isfinite(wall) or elapsed >= wall):
        return 0, 'The shared wall-clock limit has been reached or is invalid.'
    if not reserve or left <= reserve:
        return 0, 'Search is closed; preserve the recorded final refit allowance.'
    return left, None


def _digest(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _observations(history, evidence, metric, splits, trusted_partitions):
    by_trial, attempted, measured = {}, {}, {}
    for row in history:
        trial, name, split = row['trial'], row['candidate'], row.get('split_id', PRIMARY)
        if (type(trial) is not int or trial < 1 or trial in by_trial or not isinstance(name,str)
                or split not in (PRIMARY,*splits) or (name,split) in attempted):
            raise ValueError('Invalid or duplicated trial, candidate or partition')
        by_trial[trial] = row
        attempted[name,split] = row
        if row.get('status') == 'completed':
            if row.get('primary_metric','accuracy') != metric.name or score_of(row) is None:
                raise ValueError('Completed trial lacks a comparable primary metric')
            metric.check_identity(row.get('metric_spec'))
            measured[name,split] = row
    signatures, records, partitions = {}, {}, {}
    for item in evidence:
        trial = item['trial']
        row = by_trial.get(trial)
        name, split = item.get('candidate_id', item.get('candidate')), item['split_id']
        if not isinstance(name, str) or item.get('candidate', name) != name:
            raise ValueError('Missing or conflicting candidate identity in prediction evidence')
        if (type(trial) is not int or row is None or measured.get((name,split)) is not row
                or trial in records or item.get('row_id_scope') != ROW_SCOPE):
            raise ValueError('Prediction evidence does not uniquely identify a completed trial')
        ids, predictions = item['row_ids'], item['predictions']
        if (not isinstance(ids,list) or not ids or any(type(i) is not int or i < 0 for i in ids)
                or len(set(ids)) != len(ids) or not isinstance(predictions,list) or len(ids) != len(predictions)
                or not all(p in metric.labels for p in predictions)):
            raise ValueError('Invalid prediction domain or ordered row identities')
        payload, labels = item['evaluation_payload_sha256'], item['label_sha256']
        if not _digest(payload) or not _digest(labels) or item['prediction_sha256'] != fingerprint(predictions):
            raise ValueError('Prediction or payload fingerprint is invalid')
        score_values=score_classes=None
        if metric.requires_scores:
            if trusted_partitions is None:
                raise ValueError('AUC comparisons require trusted local labels, not scalar scores alone')
            from .score_evidence import checked_score_record
            score_values,score_classes=checked_score_record(item.get('score_record'),item,metric,name,
                                                           trusted_partitions[split]['labels'])
        if trusted_partitions is not None:
            trusted = trusted_partitions[split]
            if (list(trusted['row_ids']) != ids or trusted['evaluation_payload_sha256'] != payload
                    or fingerprint(trusted['labels']) != labels):
                raise ValueError('Evidence differs from the trusted local training partition')
            actual = metric.evaluate(trusted['labels'],predictions,score_values,score_classes)['score']
            if not math.isclose(actual,score_of(row),abs_tol=1.e-10):
                raise ValueError('Stored score differs from the declared metric on the local rows')
        signature = (tuple(ids), payload, labels)
        if split in signatures and signatures[split] != signature:
            raise ValueError('Ordered rows, payload or labels differ inside one partition')
        identity = frozenset(ids)
        if any(k != split and (v[0] == identity or v[1] == payload) for k,v in partitions.items()):
            raise ValueError('A confirmation reuses an existing evaluation partition')
        signatures[split], partitions[split] = signature, (identity,payload)
        records[trial] = predictions
    return attempted, measured, records


def _cost(record, known_single):
    params = record.get('params') or {}
    declared = [v for v in (record.get('expected_classifier_fits'),
                record.get('classifier_fits') if 'id' in record else None,
                params.get('maximum_classifier_fits_per_evaluation'),
                params.get('classifier_fits_per_evaluation')) if v is not None]
    if not declared and record.get('id',record.get('candidate')) in known_single:
        declared = [1]
    if not declared or any(type(v) is not int or v < 1 for v in declared):
        return None
    observed = record.get('classifier_fits',0)
    ceiling = max(declared)
    return ceiling if type(observed) is int and 0 <= observed <= ceiling else None


def _refit_cost(record, cost):
    value = record.get('refit_classifier_fits', cost)
    return value if type(value) is int and value > 0 else None


def _with_capability(row, name, capabilities, kind):
    """Use locally registered bounds and metadata; retain the worker's actual fit count."""
    cap = capabilities[name]
    if cap['kind'] != kind or row.get('params') != cap['params']:
        raise ValueError('Recorded configuration differs from the trusted executable capability: ' + name)
    fits, refit = cap['classifier_fits'], cap['refit_classifier_fits']
    if any(type(n) is not int or n < 1 for n in (fits, refit)):
        raise ValueError('Invalid trusted fit cost for ' + name)
    bound = dict(row, expected_classifier_fits=fits, refit_classifier_fits=refit,
                 supports_probabilities=cap['supports_probabilities'])
    if 'id' in row:
        bound['classifier_fits'] = fits
    if 'route' in cap:
        bound['route'] = cap['route']
    return bound


def _contenders(lead, front, records, metric, trusted):
    pred = records.get(lead['trial'])
    if pred is None:
        return
    for other in front:
        if other['trial'] == lead['trial']:
            continue
        second = records.get(other['trial'])
        if second is None:
            continue
        discordant = sum(a != b for a,b in zip(pred,second,strict=True))
        if not discordant and not metric.requires_scores:
            continue
        # A descriptive triage threshold, not a confidence bound or an efficacy test.
        if metric.name == 'accuracy':
            margin = max(MIN_GAIN, math.sqrt(discordant)/len(pred))
            screen = 'paired hard-label disagreement'
        elif metric.name == 'macro_f1' and trusted is not None:
            margin = max(MIN_GAIN,macro_f1_pair_spread(trusted['labels'],pred,second,metric.labels,
                                                       trusted.get('groups')))
            screen = 'paired local macro-F1 resampling'
        else:
            margin = MIN_GAIN
            screen = 'practical primary-metric threshold'
        if score_of(lead) - score_of(other) <= margin:
            yield lead, other, discordant, len(pred), screen


def _promotion_block(incumbent, challenger, splits, attempted, records):
    """An optimistic bound for this pair on the current registered partitions."""
    possible_wins = 0
    refs = []
    for split in (PRIMARY, *splits):
        leader = attempted.get((incumbent, split))
        rival = attempted.get((challenger, split))
        observed = [row for row in (leader, rival) if row is not None]
        refs.extend(row['trial'] for row in observed)
        # An attempted side without a verified prediction cannot produce a new matched win.
        if any(row['trial'] not in records for row in observed):
            continue
        # The contract normalizes all primary metrics to [0, 1]. Give each missing
        # side the score most favorable to the challenger, including half-finished pairs.
        leader_floor = score_of(leader) if leader is not None else 0.
        rival_ceiling = score_of(rival) if rival is not None else 1.
        if rival_ceiling > leader_floor:
            possible_wins += 1
    if possible_wins >= MIN_PAIRED_WINS:
        return None
    reason = (f"Skipped replication of {challenger} against {incumbent}. At most "
              f"{possible_wins} of the registered partitions could be strict wins. "
              f"Promotion requires {MIN_PAIRED_WINS}.")
    return reason, refs


def _explain_skips(action, skipped):
    for reason, refs in skipped:
        action['reason'] += ' ' + reason
        action['evidence_trials'] = list(dict.fromkeys([*action['evidence_trials'], *refs]))
    return action


def _confirmation(pair, splits, attempted, measured, records, available, reserve, known_single, metric):
    if pair is None:
        return None
    lead, rival, discordant, rows, screen = pair
    names = [r['candidate'] for r in (lead,rival)]
    refs = [lead['trial'], rival['trial']]
    costs = {r['candidate']:_cost(r,known_single) for r in (lead,rival)}
    if None in costs.values():
        return None
    reason = (f"Comparable primary {metric.name} {score_of(lead):.4f} and {score_of(rival):.4f}; "
              f"hard predictions differ on {discordant}/{rows} aligned rows ({screen}).")
    for index, split in enumerate(splits):
        if index:
            previous = [measured.get((name,splits[index-1])) for name in names]
            if any(r is None or r['trial'] not in records for r in previous):
                return None
            primary_gap = score_of(lead) - score_of(rival)
            previous_gap = score_of(previous[0]) - score_of(previous[1])
            if (primary_gap*previous_gap > 0 and abs((primary_gap+previous_gap)/2) >= MIN_GAIN
                    or primary_gap == previous_gap == 0):
                return None
            refs.extend(r['trial'] for r in previous)
            reason += f" The first replication did not resolve the primary ordering (gap {previous_gap:+.4f})."
        tried = [attempted.get((name,split)) for name in names]
        if any(r is not None and (r.get('status') != 'completed' or r['trial'] not in records) for r in tried):
            return None
        missing = [name for name,r in zip(names,tried,strict=True) if r is None]
        refs.extend(r['trial'] for r in tried if r is not None)
        if missing:
            if sum(costs[name] for name in missing) + reserve > available:
                return None
            name = missing[0]
            return _action('confirm',reason + ' Complete a matched pair on ' + split + '.',
                           refs,name,split,costs[name])
    return None


def selection_evidence(state, *, trusted_partitions=None):
    """Local finalization input for A. The six-field stop action never designates a winner."""
    metric = MetricSpec.from_payload({'metric_spec':state['task']['metric']})
    splits = tuple(state.get('registered_confirmation_splits',CONFIRMATIONS))
    if state.get('ab_schema_version') != 1 or splits != CONFIRMATIONS[:len(splits)]:
        raise ValueError('Unsupported v1 partition declaration')
    _, measured, records = _observations(state['history'],state['prediction_evidence'],metric,splits,trusted_partitions)
    review = incumbent_review(measured,records,splits)
    review['primary_metric'] = metric.name
    return review


def _fixed_confirm(front, attempted, records, splits, available, reserve, known_single):
    finalists = front[:3]
    refs = [row['trial'] for row in finalists]
    if len(finalists) < 2 or any(trial not in records for trial in refs):
        return _action('stop', 'Insufficient verified primary contenders for fixed confirmation.', refs)
    for split in splits:
        missing = []
        for row in finalists:
            prior = attempted.get((row['candidate'], split))
            if prior is None:
                missing.append(row)
            elif prior['trial'] in records:
                refs.append(prior['trial'])
            else:
                return _action('stop', 'Fixed confirmation has an unverified attempt.', refs)
        if missing:
            costs = [_cost(row, known_single) for row in missing]
            if None in costs or sum(costs) + reserve > available:
                return _action('stop', 'Remaining budget cannot complete this fixed confirmation round.', refs)
            return _action('confirm', 'Fixed allocation replicates the best primary configurations on both registered splits.',
                           refs, missing[0]['candidate'], split, costs[0])
    return _action('stop', 'Completed the registered fixed exploration and confirmation allocation.', refs)


def next_action(state, *, on_explore=None, trusted_partitions=None, capabilities=None, allocation='adaptive'):
    """on_explore receives eligible IDs only; A owns the Flash request and cumulative ledger."""
    if allocation not in ('adaptive', 'fixed'):
        raise ValueError('Unknown AB budget allocation')
    if state.get('ab_schema_version') != 1:
        raise ValueError('Expected ab_schema_version=1')
    task, budget = state['task'], state['budget']
    if task.get('contract_version') != 1 or task.get('objective') != 'classification':
        raise ValueError('Expected an A classification contract')
    metric = MetricSpec.from_payload({'metric_spec':task['metric']})
    available, reason = _allowance(budget)
    if reason:
        return _action('stop',reason)
    splits = tuple(state.get('registered_confirmation_splits',CONFIRMATIONS))
    if splits != CONFIRMATIONS[:len(splits)]:
        raise ValueError('Expected an ordered prefix of the v1 confirmation split IDs')
    history, entries = state['history'], state['available_candidates']
    try:
        if capabilities is not None:
            history = [_with_capability(r, r['candidate'], capabilities, task['kind']) for r in history]
            entries = [_with_capability(e, e['id'], capabilities, task['kind']) for e in entries]
        attempted, measured, records = _observations(history,state['prediction_evidence'],metric,splits,trusted_partitions)
    except (KeyError,TypeError,ValueError) as error:
        return _action('stop','Execution evidence requires reconciliation: ' + str(error))
    # The published A fixture supplies just id/params. Only recognized single-fit methods
    # may use the v1 default; extension methods must explicitly declare their fit ceiling.
    from .models import search_candidates
    known_single = {c.family for c in search_candidates(task['kind'],'ab-v1')}
    front = sorted((r for (name,split),r in measured.items() if split == PRIMARY),
                   key=lambda r:(-score_of(r),r['candidate']))
    costs = [_cost(r,known_single) for r in front]
    refits = [_refit_cost(row, cost) for row, cost in zip(front, costs, strict=True)]
    reserved = budget['reserved_final_refit']
    if None in costs or None in refits or any('refit_classifier_fits' in row and refit > reserved
                                              for row, refit in zip(front, refits, strict=True)):
        return _action('stop','A completed candidate has unknown cost or exceeds the registered refit reserve.')
    physical = sum(row.get('classifier_fits',0) for row in measured.values() if type(row.get('classifier_fits')) is int)
    if physical > budget['completed_classifier_fits']:
        return _action('stop','Recorded worker fits exceed the ledger count.')
    reserve = max([reserved,*refits])
    if len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate entries in shared catalogue')
    feasible = []
    for e in entries:
        name, cost = e['id'], _cost(e,known_single)
        refit = _refit_cost(e,cost)
        if refit is None or ('refit_classifier_fits' in e and refit > reserved):
            continue
        minimum = 2*cost if cost and len(front)>=3 else cost
        if (name,PRIMARY) not in attempted and cost and minimum + max(reserve,refit) <= available:
            if not metric.requires_scores or e.get('supports_probabilities', 'rbf_svm' not in name):
                feasible.append((e,cost))
    skipped = []
    if allocation == 'fixed':
        if len(front) >= 4 or not feasible:
            return _fixed_confirm(front, attempted, records, splits, available, reserve, known_single)
    elif any(r['trial'] in records for r in front) and (len(front) >= 3 or not feasible):
        lead_id = incumbent_review(measured,records,splits)['incumbent']
        lead = next(r for r in front if r['candidate']==lead_id)
        local_primary = trusted_partitions.get(PRIMARY) if trusted_partitions is not None else None
        for pair in _contenders(lead,front,records,metric,local_primary):
            confirm = _confirmation(pair,splits,attempted,measured,records,available,reserve,known_single,metric)
            if confirm:
                block = _promotion_block(lead_id, pair[1]['candidate'], splits, attempted, records)
                if block is None:
                    return _explain_skips(confirm, skipped)
                skipped.append(block)
    if not feasible:
        reason = ('No remaining eligible search or replication can advance selection.' if skipped else
                  'No eligible new fit or complete matched replication fits the shared budget.')
        action = _action('stop',reason,[r['trial'] for r in front[:2]])
        return _explain_skips(action, skipped)
    def route(record):
        name = record.get('id',record.get('candidate'))
        if record.get('route'):
            return record['route']
        try:
            return route_of(name)
        except ValueError:
            return name
    tried_routes = [route(r) for (name,split),r in attempted.items() if split == PRIMARY]
    # A single shared first candidate anchors both planning modes. Afterwards explore new
    # routes, then prioritize the best observed route when revisiting a configuration.
    if not attempted:
        entry, cost = feasible[0]
        return _action('explore','Shared first eligible candidate.',(),entry['id'],PRIMARY,cost)
    scores = {}
    for row in front:
        scores.setdefault(route(row),score_of(row))
    ordered = sorted(feasible,key=lambda ec:(tried_routes.count(route(ec[0])),
                     0 if ec[0]['id'] in ROUTES else 1, -scores.get(route(ec[0]),0)))
    entry, cost = ordered[0]
    why = 'Explore a feasible route using prior route coverage and observed primary scores.'
    if on_explore is not None and budget['used_llm_requests'] < budget['llm_request_limit']:
        selected = on_explore(tuple(e['id'] for e,c in ordered))
        if selected not in {e['id'] for e,c in ordered}:
            raise ValueError('Planner returned an ineligible candidate')
        entry,cost = next((e,c) for e,c in ordered if e['id'] == selected)
        why = 'Planner selected an eligible untried configuration within the shared allowance.'
    action = _action('explore',why,[r['trial'] for r in front[:2]],entry['id'],PRIMARY,cost)
    return _explain_skips(action, skipped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Return one local A/B v1 action as JSON')
    parser.add_argument('input',type=Path)
    args = parser.parse_args()
    print(json.dumps(next_action(json.loads(args.input.read_text())),ensure_ascii=False,indent=2))
