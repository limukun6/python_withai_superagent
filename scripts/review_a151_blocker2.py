"""Offline review only: no model calls, no ledger writes, no classifier fits."""
import argparse
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import requests
import urllib3
from urllib3.exceptions import MaxRetryError, ProtocolError, ReadTimeoutError


def encode_tree(value):
    # Tagged tree, with repeated dictionary keys shared across a list of records.
    # Preserve order and missing keys distinctly from explicit null values.
    if isinstance(value, dict):
        return ['d', [[k, encode_tree(v)] for k, v in value.items()]]
    if isinstance(value, list):
        if value and all(isinstance(v, dict) for v in value):
            keys = list(dict.fromkeys(k for row in value for k in row))
            return ['t', keys, [[[keys.index(k), encode_tree(v)] for k, v in row.items()] for row in value]]
        return ['l', [encode_tree(v) for v in value]]
    return ['v', value]


def decode_tree(value):
    tag = value[0]
    if tag == 'd': return {k: decode_tree(v) for k, v in value[1]}
    if tag == 't': return [{value[1][i]: decode_tree(v) for i, v in row} for row in value[2]]
    if tag == 'l': return [decode_tree(v) for v in value[1]]
    if tag == 'v': return value[1]
    raise ValueError('Unknown encoding tag')


def compact_menu(payload):
    value = deepcopy(payload)
    def table(records):
        groups = []
        for record in records:
            columns = list(record)
            if not groups or groups[-1]['columns'] != columns:
                groups.append({'columns': columns, 'rows': []})
            groups[-1]['rows'].append(list(record.values()))
        return groups
    if 'available_candidates' in value:
        value['available_candidates'] = table(value['available_candidates'])
    if 'candidate_facts' in value:
        facts = value['candidate_facts']
        value['candidate_facts'] = {'ids': list(facts), 'tables': table(facts.values())}
    counts = Counter()
    originals = {}
    def collect(node):
        key = dumps(node)
        if len(key) > 40:
            counts[key] += 1
            originals[key] = node
        if isinstance(node, dict):
            for child in node.values(): collect(child)
        if isinstance(node, list):
            for child in node: collect(child)
    for field in ('available_candidates', 'candidate_facts'):
        if field in value: collect(value[field])
    keys = [k for k, n in counts.items() if (n-1)*len(k) > 15*n+10]
    indices = {k: i for i, k in enumerate(keys)}
    def encode(node, own=None):
        key = dumps(node)
        if key in indices and key != own: return {'@': indices[key]}
        if isinstance(node, dict):
            if '@' in node or '@@' in node:
                return {'@@': [[k, encode(v)] for k, v in node.items()]}
            return {k: encode(v) for k, v in node.items()}
        if isinstance(node, list): return [encode(v) for v in node]
        return node
    for key in ('available_candidates', 'candidate_facts'):
        if key in value:
            value[key] = encode(value[key])
    value['menu_encoding'] = {
        'version': 'research-menu-tables-v3',
        'shared_values': [encode(originals[k], own=k) for k in keys],
        'instruction': 'Only available_candidates and candidate_facts are encoded. '
        'Recursively replace an object with sole key @ by shared_values[its integer value]. '
        'An object with sole key @@ is escaped original key/value pairs. '
        'Then decode available_candidates as consecutive tables: columns name row values. '
        'candidate_facts has ids and tables of corresponding value records. '
        'All candidates, exact parameters, omitted/null distinctions and proposal requirements remain in force.'}
    return value


def expand_menu(payload):
    value = deepcopy(payload)
    metadata = value.pop('menu_encoding')
    if metadata['version'] != 'research-menu-tables-v3':
        raise ValueError('Unknown menu encoding')
    def decode(node):
        if isinstance(node, dict):
            if set(node) == {'@'}: return decode(metadata['shared_values'][node['@']])
            if set(node) == {'@@'}: return {k: decode(v) for k, v in node['@@']}
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list): return [decode(v) for v in node]
        return node
    for key in ('available_candidates', 'candidate_facts'):
        if key in value: value[key] = decode(value[key])
    def untab(groups):
        return [dict(zip(group['columns'], row)) for group in groups for row in group['rows']]
    if 'available_candidates' in value:
        value['available_candidates'] = untab(value['available_candidates'])
    if 'candidate_facts' in value:
        facts = value['candidate_facts']
        value['candidate_facts'] = dict(zip(facts['ids'], untab(facts['tables'])))
    return value


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def transport_review():
    with requests.Session() as session:
        retry = session.get_adapter('https://example.invalid').max_retries
        checks = {}
        for name, error in [('connect', urllib3.exceptions.ConnectTimeoutError('offline fixture')),
                            ('read', ReadTimeoutError(None, '/', 'offline fixture')),
                            ('protocol', ProtocolError('offline fixture'))]:
            try:
                retry.increment(method='POST', url='/', error=error)
            except (MaxRetryError, ReadTimeoutError, ProtocolError): checks[name] = 'raises_without_retry'
            else: checks[name] = 'would_allow_retry'
        return dict(requests_version=requests.__version__, urllib3_version=urllib3.__version__,
                    retry_repr=repr(retry), total_retries=retry.total, error_checks=checks,
                    campus_or_upstream_retry_observable=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def save(name, value):
        (a.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('protocol.json', dict(batch='A151-blocker2-offline-review',
         input_sha256=hashlib.sha256(a.events.read_bytes()).hexdigest(),
         runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         maximum_fits=0, maximum_model_API_requests=0, modifies_production=False))
    try:
        events = [json.loads(line) for line in a.events.read_text(encoding='utf-8').splitlines() if line.strip()]
        requests_log = [e for e in events if e.get('kind') == 'llm_request']
        if len(requests_log) != 1: raise ValueError('Expected exactly one recorded request')
        event = requests_log[0]
        payload = json.loads(next(m['content'] for m in event['messages'] if m['role'] == 'user'))
        packed = compact_menu(payload)
        if expand_menu(packed) != payload: raise ValueError('Lossless reconstruction failed')
        original = dumps(payload).encode('utf-8'); encoded = dumps(packed).encode('utf-8')
        result = dict(transport=transport_review(), roundtrip_equal=True,
            all_candidate_ids_preserved=True, candidate_count=len(payload['available_candidates']),
            original_minified_bytes=len(original), encoded_minified_bytes=len(encoded),
            reduction_percent=100 * (1 - len(encoded) / len(original)),
            recorded_max_tokens=event.get('max_tokens'), recorded_streaming=event.get('streaming'),
            recorded_timeout=event.get('request_timeout_seconds'),
            tokenizer_measurement=False, live_model_acceptance_tested=False,
            production_switch_implemented=False,
            encoding_scope='menu keys only; all remaining input/output requirements preserved')
        save('results.json', result)
        save('completion.json', dict(status='passed', elapsed_seconds=time.monotonic()-start,
             actual_classifier_fits=0, real_API_requests=0, course_cost_rmb=0))
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        save('completion.json', dict(status='failed', error=str(error), actual_classifier_fits=0, real_API_requests=0))
        raise


if __name__ == '__main__': main()
