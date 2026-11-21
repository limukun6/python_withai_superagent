"""B's shared, initially visible menu. Both planners receive the same returned records."""
from copy import deepcopy
from importlib.util import find_spec

from .ab_methods import FIELDS, RELATION, ROUTES, TEXT
from .ab_symmetric import ENCODER
from .ab_vision import NAME as VISION, supports_decoder
from .evidence_agent import route_of
from .metrics import MetricSpec
from .models import search_candidates
from .task_semantics import candidate_swap_rule, validate_candidate_semantics


ADAPTER_VERSION = '1.4'


def common_catalog(task, *, train_rows=None, kernel_memory_bytes=512 * 2**20):
    if task.get('contract_version') != 1 or task.get('objective') != 'classification':
        raise ValueError('Expected an A/B v1 classification task')
    metric = MetricSpec.from_payload({'metric_spec': task['metric']})
    kind = task['kind']
    schema = task['input_schema']
    pair = schema.get('pair_semantics')
    if train_rows is not None and (type(train_rows) is not int or train_rows < 1):
        raise ValueError('Full refit row count must be positive')
    if type(kernel_memory_bytes) is not int or kernel_memory_bytes < 0:
        raise ValueError('Invalid kernel memory allowance')
    menu = []
    for candidate in search_candidates(kind, 'ab-v1'):
        name, params = candidate.family, candidate.params
        if name == VISION and not supports_decoder(schema):
            continue
        if name == RELATION:
            columns = schema.get('columns')
            if (pair != 'relation' or schema.get('modality') != 'pair_text' or not isinstance(columns, list)
                    or len(columns) != 2 or any(not isinstance(c,dict) or c.get('type') != 'text' for c in columns)):
                continue
        if kind == 'pair_text':
            rule = candidate_swap_rule(candidate)
            preference = pair == 'preference' or (pair is None and schema.get('pair_label_one') in ('left','right'))
            if pair == 'symmetric':
                meaning = schema.get('pair_label_one')
                if rule != 'preserves' or len(metric.labels) != 2 or meaning not in (None, 'equivalent'):
                    continue
                if meaning == 'equivalent' and set(metric.labels) != {0, 1}:
                    continue
            elif not preference or rule != 'flips' or set(metric.labels) != {0, 1}:
                continue
            validate_candidate_semantics(kind, pair, candidate)
        route = route_of(name)
        if name == FIELDS or 'catboost' in route:
            estimator = 'catboost'
            if find_spec('catboost') is None:
                continue
        elif 'extra_trees' in route:
            estimator = 'extra_trees'
        elif 'rbf_svm' in route:
            estimator = 'rbf_svm'
            if metric.requires_scores or train_rows is None or 8 * train_rows**2 > kernel_memory_bytes:
                continue
        else:
            estimator = 'logistic'
        representation = params.get('representation') or {
            'numeric_tabular':'imputed scaled numeric columns',
            'mixed_tabular':'scaled numeric and encoded categories',
            'text_tabular':'typed table and text features',
            'image':'flattened standardized pixels',
            'categorical_tabular':'native categories' if estimator == 'catboost' else 'one-hot categories',
        }.get(kind, route)
        menu.append(dict(id=name, params=dict(params), route=route, input_kind=kind,
                         representation=representation, estimator=estimator,
                         swap_label_rule=candidate_swap_rule(candidate) if kind == 'pair_text' else 'not_applicable',
                         supports_probabilities=estimator != 'rbf_svm', expected_classifier_fits=1,
                         resource='local_frozen_encoder_cpu' if name in (TEXT, ENCODER, VISION) else 'cpu',
                         origin='b_port' if name in ROUTES else 'shared_baseline'))
    return menu


def capabilities_for(task, *, train_rows=None, kernel_memory_bytes=512 * 2**20):
    """Build the executor's registry before exploration. Use only locally runnable methods."""
    entries = common_catalog(task, train_rows=train_rows, kernel_memory_bytes=kernel_memory_bytes)
    return {
        entry['id']: dict(entry, kind=task['kind'], classifier_fits=entry['expected_classifier_fits'],
                          refit_classifier_fits=1)
        for entry in entries
    }


def untried_candidates(capabilities, history):
    """Detach the planners' common view from the trusted registry, including nested params."""
    tried = {row['candidate'] for row in history}
    return [deepcopy(entry) for name, entry in capabilities.items() if name not in tried]
