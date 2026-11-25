"""A portable, bounded calibration of the expected score of the next experiment."""
import json
from pathlib import Path

import numpy as np


RULE_FORMAT = 'exploration-forecast-v1'


def snapshot(weights):
    values = list(weights)
    if len(values) != 3 or any(type(x) not in (float, int, np.float64) or not -.12 <= x <= .12
                               for x in values):
        raise ValueError('Expected three finite exploration weights between -0.12 and 0.12')
    return {'format': RULE_FORMAT, 'weights': [float(x) for x in values]}


def load_rule(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(record, dict) or set(record) != {'format', 'weights'} or record['format'] != RULE_FORMAT:
        raise ValueError('Unrecognized exploration rule format')
    if not isinstance(record['weights'], list):
        raise ValueError('Expected a list of exploration weights')
    return snapshot(record['weights'])


def features(prior, evidence):
    return np.array([1.0, evidence['novelty'] - 0.5, 1 / np.sqrt(prior)])


def forecast(weights, prior, evidence):
    correction = np.clip(weights @ features(prior, evidence), -0.10, 0.10)
    return float(np.clip(evidence['expected'] + correction, 0, 1))


def apply_rule(menu, weights, prior):
    if prior == 0 or not np.any(weights):
        return menu
    revised = []
    for action in menu:
        if action['action'] == 'explore' and action['split'] == 'primary':
            evidence = action['evidence']
            delta = forecast(weights, prior, evidence) - evidence['expected']
            action = dict(action, utility=action['utility'] + delta,
                          evidence=dict(evidence, inherited_forecast_adjustment=delta))
        revised.append(action)
    return sorted(revised, key=lambda action: -action['utility'])
