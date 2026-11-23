"""Experimental screening with bounded sample rungs and full-resource selection.

This is a small staged policy inspired by successive halving, not Hyperband.
Every final candidate is measured again at full training size before confirmation.
"""
import json
from pathlib import Path
import time
import joblib
import numpy as np
from .models import search_candidates
from .evidence_agent import route_of
from .local_search import LocalSearchPolicy
from .splits import inner_split, take
from .metrics import MetricSpec, score_of


def candidate_pool(kind, portfolio, size, seed):
    catalog = [{'id': c.family, 'params': c.params, 'route': route_of(c.family)} for c in search_candidates(kind, portfolio)]
    policy = LocalSearchPolicy('balanced_blind', catalog, seed)
    history = []
    for _ in range(min(size, len(catalog))):
        selected, _ = policy.choose(history)
        item = next(c for c in catalog if c['id'] == selected)
        history.append({'candidate': selected, 'route': item['route']})
    return [r['candidate'] for r in history]


def nested_subsample(labels, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('Sample fraction must lie in (0,1]')
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        order = rng.permutation(rows)
        selected.extend(order[:max(1, int(np.ceil(len(rows) * fraction)))].tolist())
    return np.array(sorted(selected), dtype=int)


def promote(records, top=2, protect_route=True):
    successful = sorted((r for r in records if r['status'] == 'completed'),
                        key=lambda r: (-score_of(r), r['candidate']))
    chosen = successful[:top]
    reason = 'highest_observed_score'
    if protect_route and len(successful) > top:
        routes = {route_of(r['candidate']) for r in chosen}
        remaining = successful[top:]
        rescue = next((r for r in remaining if route_of(r['candidate']) not in routes), remaining[0])
        chosen.append(rescue)
        reason += '_plus_one_protected_alternative'
    return [r['candidate'] for r in chosen], reason


def run_allocated(task, strategy, seed, directory, client, mode, **options):
    from .benchmark import run_one, train_candidate
    if client is not None or not strategy.startswith('local_') or options.get('refinement', 'off') != 'off':
        raise ValueError('Resource study requires local planning and no composition/refinement')
    if mode not in ('uniform', 'successive'):
        raise ValueError('Unknown resource allocation')
    started = time.monotonic()
    duration = options['time_limit']
    pool = candidate_pool(task.kind, options.get('portfolio', 'expanded'), options['max_trials'], seed)
    metadata = {'mode': mode, 'initial_candidate_pool': pool, 'screening_fractions': [.25, .5] if mode == 'successive' else [],
                'selection_after_screening': 'full-resource primary and confirmation only',
                'same_declared_time_ceiling_seconds': duration, 'equal_actual_compute_guaranteed': False,
                'screening_records': [], 'promotions': [], 'screening_classifier_fits': 0,
                'failed_screening_component_count_unknown': 0, 'all_planned_screening_complete': True}
    final_pool = pool
    if mode == 'successive':
        screen = Path(str(directory) + '_allocation')
        screen.mkdir(parents=True, exist_ok=False)
        train, search = inner_split(task, seed, options.get('search_grouping', 'shared_text'))
        metric = MetricSpec.for_task(task)
        choices = pool
        for stage, fraction in enumerate((.25, .5), 1):
            relative = nested_subsample(task.y_train[train], fraction, seed)
            selected = train[relative]
            payload = screen / f'rung_{stage}.joblib'
            joblib.dump({'kind': task.kind, 'X_train': take(task.X_train, selected), 'y_train': task.y_train[selected],
                         'X_search': take(task.X_train, search), 'y_search': task.y_train[search],
                         'metric_spec': metric.record()}, payload)
            (screen / f'rung_{stage}_indices.json').write_text(json.dumps(
                {'training': selected.tolist(), 'search': search.tolist(), 'fraction': fraction}) + '\n')
            rung = []
            for i, candidate in enumerate(choices):
                remaining = .45 * duration - (time.monotonic() - started)
                if remaining <= 0:
                    metadata['all_planned_screening_complete'] = False
                    metrics = {'status': 'timeout', 'error': 'Screening share exhausted before transport/training'}
                else:
                    _, metrics = train_candidate(payload, candidate, screen / f'rung_{stage}_{i}.joblib',
                                                 min(options['trial_timeout'], remaining))
                row = {'stage': stage, 'candidate': candidate, 'sample_fraction': fraction,
                       'training_rows': len(selected), 'artifact': f'rung_{stage}_{i}.joblib', **metrics}
                rung.append(row); metadata['screening_records'].append(row)
                if row['status'] == 'completed':
                    metadata['screening_classifier_fits'] += row.get('classifier_fits', 1)
                elif remaining > 0:
                    metadata['failed_screening_component_count_unknown'] += 1
                    metadata['all_planned_screening_complete'] = False
            choices, reason = promote(rung, top=2, protect_route=stage == 1)
            metadata['promotions'].append({'after_stage': stage, 'candidate_ids': choices, 'reason': reason})
            (screen / 'screening.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
            if not choices:
                raise RuntimeError('No screening candidate succeeded; see preserved screening records')
        final_pool = choices
    remaining = duration - (time.monotonic() - started)
    if remaining <= 0:
        raise RuntimeError('No budget remains for full-resource evaluation')
    metadata['final_primary_trials_planned'] = len(final_pool)
    metadata['final_candidate_pool'] = final_pool
    final_options = {**options, 'max_trials': len(final_pool), 'time_limit': remaining,
                     'confirmation_top_k': min(3, len(final_pool)), 'catalog_ids': final_pool}
    if final_options.get('run_metadata') is not None:
        final_options['run_metadata'] = {**final_options['run_metadata'],
                                        'resource_allocation': metadata,
                                        'primary_trials_including_anchor': len(final_pool)}
    summary = run_one(task, strategy, seed, directory, None, **final_options)
    metadata['total_elapsed_seconds'] = time.monotonic() - started
    summary['resource_allocation'] = metadata
    summary['elapsed_seconds'] = metadata['total_elapsed_seconds']
    accounting = summary['classifier_fit_accounting']
    accounting['completed_classifier_fits'] += metadata['screening_classifier_fits']
    accounting['failed_candidate_attempts_with_unknown_component_count'] += metadata['failed_screening_component_count_unknown']
    accounting['complete_count_known'] = accounting['failed_candidate_attempts_with_unknown_component_count'] == 0
    accounting['scope'] += '; includes all preceding resource-screening fits'
    summary['valid_policy_run'] = summary['valid_policy_run'] and metadata['all_planned_screening_complete']
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (directory / 'resource_allocation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return summary
