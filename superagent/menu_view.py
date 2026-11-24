"""Lossless compact view of the candidate menu sent to the planner.

The menu is the largest block of the planning payload: twenty-nine candidates
with full parameter dictionaries take about sixty percent of the bytes. The
same structure repeated per candidate is the redundancy, not the information.

Only the representation changes. `decode_menu(compact_menu(payload))` must
equal the original payload, so the planner sees every candidate, every exact
parameter, and the same distinctions between an omitted key and a null value.
Proposal validation keeps using the original capability table; nothing here
invents or drops a candidate.

Version `menu-tables-v3` comes from the offline prototype in
`scripts/review_a151_blocker2.py`; this module is the production copy and
`tests/test_menu_view.py` checks the two agree.
"""
from collections import Counter
from copy import deepcopy
import json

ENCODING_VERSION = 'menu-tables-v3'
ENCODED_FIELDS = ('available_candidates', 'candidate_facts')
INSTRUCTION = (
    'Only available_candidates and candidate_facts are encoded. Recursively replace an object with '
    'sole key @ by shared_values[its integer value]. An object with sole key @@ is escaped original '
    'key/value pairs. Then decode available_candidates as consecutive tables: columns name row values. '
    'candidate_facts has ids and tables of corresponding value records. All candidates, exact '
    'parameters, omitted/null distinctions and proposal requirements remain in force.')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _table(records):
    groups = []
    for record in records:
        columns = list(record)
        if not groups or groups[-1]['columns'] != columns:
            groups.append({'columns': columns, 'rows': []})
        groups[-1]['rows'].append(list(record.values()))
    return groups


def _untable(groups):
    return [dict(zip(group['columns'], row)) for group in groups for row in group['rows']]


def compact_menu(payload):
    """Return a compact view; `decode_menu` restores the input exactly."""
    value = deepcopy(payload)
    if 'available_candidates' in value:
        value['available_candidates'] = _table(value['available_candidates'])
    if 'candidate_facts' in value:
        facts = value['candidate_facts']
        value['candidate_facts'] = {'ids': list(facts), 'tables': _table(list(facts.values()))}
    counts, originals = Counter(), {}
    def collect(node):
        key = dumps(node)
        if len(key) > 40:
            counts[key] += 1
            originals[key] = node
        if isinstance(node, dict):
            for child in node.values():
                collect(child)
        if isinstance(node, list):
            for child in node:
                collect(child)
    for field in ENCODED_FIELDS:
        if field in value:
            collect(value[field])
    shared = [k for k, n in counts.items() if (n - 1) * len(k) > 15 * n + 10]
    indices = {k: i for i, k in enumerate(shared)}
    def encode(node, own=None):
        key = dumps(node)
        if key in indices and key != own:
            return {'@': indices[key]}
        if isinstance(node, dict):
            if '@' in node or '@@' in node:
                return {'@@': [[k, encode(v)] for k, v in node.items()]}
            return {k: encode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [encode(v) for v in node]
        return node
    for field in ENCODED_FIELDS:
        if field in value:
            value[field] = encode(value[field])
    value['menu_encoding'] = {'version': ENCODING_VERSION,
                              'shared_values': [encode(originals[k], own=k) for k in shared],
                              'instruction': INSTRUCTION}
    return value


def decode_menu(payload):
    """Inverse of `compact_menu`; raises on an unknown or malformed encoding."""
    value = deepcopy(payload)
    if 'menu_encoding' not in value:
        raise ValueError('Payload has no menu encoding to decode')
    metadata = value.pop('menu_encoding')
    if not isinstance(metadata, dict) or metadata.get('version') != ENCODING_VERSION:
        raise ValueError('Unknown menu encoding version')
    shared = metadata.get('shared_values')
    if not isinstance(shared, list):
        raise ValueError('Menu encoding needs a shared value list')
    def decode(node):
        if isinstance(node, dict):
            if set(node) == {'@'}:
                index = node['@']
                if type(index) is not int or not 0 <= index < len(shared):
                    raise ValueError('Shared value reference is out of range')
                return decode(shared[index])
            if set(node) == {'@@'}:
                return {k: decode(v) for k, v in node['@@']}
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [decode(v) for v in node]
        return node
    for field in ENCODED_FIELDS:
        if field in value:
            value[field] = decode(value[field])
    if 'available_candidates' in value:
        value['available_candidates'] = _untable(value['available_candidates'])
    if 'candidate_facts' in value:
        facts = value['candidate_facts']
        if not isinstance(facts, dict) or set(facts) != {'ids', 'tables'}:
            raise ValueError('Encoded candidate_facts must carry ids and tables')
        value['candidate_facts'] = dict(zip(facts['ids'], _untable(facts['tables'])))
    return value


def recorded_size(payload):
    """UTF-8 size of the payload as the client serialises it."""
    return len(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
