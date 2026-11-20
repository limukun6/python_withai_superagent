"""Check declared configuration facts and numerical slice expectations."""
from functools import partial
import math


def candidate_facts(kind, configuration):
    from .models import search_candidates
    candidate = next((c for c in search_candidates(kind, 'all')
                      if c.family == configuration['id']), None)
    if candidate is None:
        classifier = configuration['params'].get('classifier')
        if classifier in ('logistic', 'linear_svm'):
            return {'classifier': classifier, 'source': 'validated_recipe_parameters'}
        return _unverified('candidate is not in the static catalogue and declares no classifier')
    registered = candidate.fit_predict
    function = registered.func if isinstance(registered, partial) else registered
    name = getattr(function, '__name__', None)
    classifiers = {'_word_difference_text': 'logistic', '_char_difference_text': 'logistic',
        '_word_char_difference_text': 'logistic', '_word_char_stats_text': 'logistic',
        '_logistic_tabular': 'logistic', '_extra_trees_tabular': 'extra_trees',
        '_catboost_tabular': 'catboost', '_dummy_tabular': 'dummy', '_dummy_text': 'dummy',
        '_text': 'logistic'}
    classifier = classifiers.get(name)
    if name == '_fit' and function.__module__.endswith('.general_models') and isinstance(registered, partial):
        classifier = registered.args[1]
    if classifier is not None:
        return {'classifier': classifier, 'source': 'registered_training_function', 'training_function': name}
    return _unverified(f'unrecognised training function {name!r}')


def _unverified(reason):
    """An unknown fact is stated, not omitted: the fitted estimator decides later."""
    return {'classifier': None, 'source': 'unverified', 'reason': reason}


def validate_slice_expectation(expectation, history):
    if not isinstance(expectation, dict) or set(expectation) != {'slice_id', 'reference_trial', 'min_delta'}:
        raise ValueError('Slice expectation needs slice_id, reference_trial and min_delta')
    if type(expectation['reference_trial']) is not int:
        raise ValueError('Slice reference must be a prior integer trial')
    reference = next((r for r in history if r['trial'] == expectation['reference_trial']), None)
    slices = (reference.get('error_slices') or {}).get('slices', []) if reference else []
    if expectation['slice_id'] not in {s['slice_id'] for s in slices}:
        raise ValueError('Slice expectation must cite an observed prior slice')
    delta = expectation['min_delta']
    if type(delta) not in (int, float) or not math.isfinite(delta) or not 0 < delta <= 1:
        raise ValueError('Expected slice improvement must be finite and positive')


def check_slice_expectation(expectation, result, history):
    reference = next((r for r in history if r['trial'] == expectation['reference_trial']), None)
    rows = []
    for trial in (reference, result):
        slices = (trial.get('error_slices') or {}).get('slices', []) if trial else []
        rows.append(next((s for s in slices if s['slice_id'] == expectation['slice_id']), None))
    before, after = rows
    identity = ('feature', 'threshold_from_training', 'relation', 'rows')
    if (not before or not after or reference.get('status') != 'completed'
            or result.get('status') != 'completed'
            or any(before[k] != after[k] for k in identity)):
        return {'status': 'not_evaluable', 'slice_id': expectation['slice_id']}
    delta = after['accuracy'] - before['accuracy']
    return {'status': 'supported' if delta + 1e-12 >= expectation['min_delta'] else 'not_supported',
            'slice_id': expectation['slice_id'], 'reference_trial': reference['trial'],
            'rows': before['rows'], 'reference_accuracy': before['accuracy'],
            'observed_accuracy': after['accuracy'], 'observed_delta': delta,
            'required_delta': expectation['min_delta'],
            'scope': 'same internal search partition and train-fitted slice definition'}
