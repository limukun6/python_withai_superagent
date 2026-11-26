"""Checked export of model outputs using a frozen, evidence-proposed CSV recipe."""
import csv
from copy import deepcopy
import hashlib
import io
import json

import numpy as np

from .events import atomic_text
from .format_sources import checked_name, decode_text, package_files, roles


VERSION = 'output-csv-v1'


def _require(ok, message):
    if not ok: raise ValueError('Output contract: '+message)


def _scalar(value):
    return type(value) in (str, int, float, bool) and not (
        isinstance(value, float) and not np.isfinite(value))


def validate_recipe(recipe, labels):
    _require(isinstance(recipe, dict) and set(recipe) ==
             {'version','delimiter','encoding','header','columns','order'}, 'invalid recipe fields')
    _require(recipe['version'] == VERSION, 'unknown version')
    delimiter = recipe['delimiter']
    _require(isinstance(delimiter, str) and len(delimiter) == 1 and delimiter not in '\r\n\x00"', 'invalid delimiter')
    _require(recipe['encoding'] in ('utf-8','utf-8-sig'), 'unsupported encoding')
    _require(type(recipe['header']) is bool, 'header must be boolean')
    columns = recipe['columns']
    _require(isinstance(columns, list) and 1 <= len(columns) <= 128, 'invalid column count')
    names, ids, predicted = [], 0, 0
    for column in columns:
        _require(isinstance(column, dict), 'invalid column')
        name, kind = column.get('name'), column.get('kind')
        _require(isinstance(name, str) and name.strip() and len(name) <= 128
                 and all(ord(c) >= 32 for c in name), 'invalid column name')
        names.append(name)
        if kind == 'input_id':
            _require(set(column) in ({'name','kind','field'}, {'name','kind','field','unique'}) and isinstance(column['field'], str)
                     and column['field'], 'invalid ID source')
            _require('unique' not in column or type(column['unique']) is bool, 'invalid ID uniqueness declaration')
            ids += 1
        elif kind == 'row_index':
            _require(set(column) == {'name','kind','start'} and type(column['start']) is int
                     and column['start'] in (0,1), 'invalid sequential index')
            ids += 1
        elif kind == 'label':
            _require(set(column) in ({'name','kind'}, {'name','kind','mapping'}), 'invalid label output')
            if 'mapping' in column:
                mapping = column['mapping']
                _require(isinstance(mapping, dict) and set(mapping) == {str(v) for v in labels}
                         and all(_scalar(v) for v in mapping.values()), 'incomplete or invalid label mapping')
                _require(len({json.dumps(v, ensure_ascii=False) for v in mapping.values()}) == len(labels),
                         'label mapping collapses classes')
            predicted += 1
        elif kind == 'probability':
            _require(set(column) == {'name','kind','class_label'} and _scalar(column['class_label'])
                     and column['class_label'] in labels, 'probability class outside training domain')
            predicted += 1
        else:
            raise ValueError('Output contract: unsupported column source')
    _require(len(set(names)) == len(names) and ids <= 1 and predicted > 0, 'duplicate columns or invalid ID/prediction layout')
    order = recipe['order']
    _require(isinstance(order, dict), 'invalid row order')
    if order.get('mode') == 'input':
        _require(set(order) == {'mode'}, 'invalid input order')
    else:
        _require(order.get('mode') == 'template' and set(order) == {'mode','member','id_column'}, 'invalid template order')
        checked_name(order['member'])
        _require(isinstance(order['id_column'], str) and order['id_column'], 'invalid template ID column')
        _require(any(c['kind'] == 'input_id' for c in columns), 'template order requires original input IDs')
        _require(all(c.get('unique',True) for c in columns if c['kind']=='input_id'),
                 'template ordering requires unambiguous unique row IDs')
        _require(not roles(order['member']) & {'train','validation'}, 'template cannot be a training/validation source')
    return deepcopy(recipe)


def needs_scores(recipe):
    return recipe is not None and any(c.get('kind') == 'probability' for c in recipe.get('columns', []))


def _identity_context(package, specification, recipe, count):
    ids, template_ids, template_hash = None, None, None
    column = next((c for c in recipe['columns'] if c['kind'] == 'input_id'), None)
    if column:
        _require('input_recipe' in specification, 'original ID extraction needs a frozen record reader')
        from .format_intake import prediction_rows
        records = prediction_rows(package, specification)
        _require(len(records) == count, 'ID rows differ from feature rows')
        _require(all(column['field'] in row for row in records), 'ID field missing from prediction input')
        from .record_streams import scalar
        ids = [scalar(row[column['field']]) for row in records]
        _require(all(v.strip() for v in ids), 'empty input ID')
        if column.get('unique',True):
            _require(len(set(ids)) == len(ids), 'duplicate input IDs')
    order = recipe['order']
    if order['mode'] == 'template':
        with package_files(package) as files:
            _require(order['member'] in files.namelist(), 'template member missing')
            raw = files.read(order['member'])
        template_hash = hashlib.sha256(raw).hexdigest()
        table = csv.reader(io.StringIO(decode_text(raw), newline=''), delimiter=recipe['delimiter'], strict=True)
        header = next(table, [])
        _require(header == [c['name'] for c in recipe['columns']], 'template headers differ from contract')
        _require(order['id_column'] in header, 'template ID column missing')
        index = header.index(order['id_column'])
        id_column = next(c for c in recipe['columns'] if c['kind'] == 'input_id')
        _require(order['id_column'] == id_column['name'], 'template ID column differs from exported ID')
        rows = list(table)
        _require(all(len(row) == len(header) for row in rows), 'template row width mismatch')
        template_ids = [row[index] for row in rows]
        _require(len(template_ids) == count and len(set(template_ids)) == count and set(template_ids) == set(ids),
                 'template and prediction ID sets differ')
    positions = {v: i for i, v in enumerate(ids)} if ids is not None else {}
    order_indices = [positions[v] for v in template_ids] if template_ids is not None else list(range(count))
    return ids, order_indices, template_hash


def _checked_arrays(predictions, labels, recipe, classes, scores):
    array = np.asarray(predictions)
    _require(array.ndim == 1 and len(array) > 0, 'expected a nonempty one-dimensional prediction vector')
    predicted = array.tolist()
    _require(all(v in labels for v in predicted), 'predicted class outside training domain')
    if needs_scores(recipe):
        classes = np.asarray(classes).tolist()
        values = np.asarray(scores, dtype=float)
        _require(isinstance(classes, list) and len(set(classes)) == len(labels) and set(classes) == set(labels),
                 'probability class order mismatch')
        _require(values.shape == (len(predicted), len(labels)) and np.isfinite(values).all()
                 and (values >= 0).all() and (values <= 1).all()
                 and np.allclose(values.sum(axis=1), 1., atol=1e-6), 'invalid probability matrix')
        return predicted, classes, values
    return predicted, None, None


def _cell(column, index, predicted, ids, classes, scores):
    kind = column['kind']
    if kind == 'input_id': return ids[index]
    if kind == 'row_index': return index + column['start']
    if kind == 'label':
        label = predicted[index]
        return column['mapping'][str(label)] if 'mapping' in column else label
    return float(scores[index, classes.index(column['class_label'])])


def verify_export(path, recipe, labels, predictions, *, ids, order_indices, classes=None, scores=None):
    """Read the actual serialized file and check it against frozen model outputs and original IDs."""
    validate_recipe(recipe, labels)
    predicted, classes, scores = _checked_arrays(predictions, labels, recipe, classes, scores)
    with open(path, encoding=recipe['encoding'], newline='') as stream:
        rows = list(csv.reader(stream, delimiter=recipe['delimiter'], strict=True))
    if recipe['header']:
        _require(bool(rows) and rows.pop(0) == [c['name'] for c in recipe['columns']], 'written header mismatch')
    _require(len(rows) == len(predicted), 'written row count mismatch')
    _require(sorted(order_indices) == list(range(len(predicted))), 'row order is not a permutation')
    for record, index in zip(rows, order_indices):
        _require(len(record) == len(recipe['columns']), 'written column count mismatch')
        for value, column in zip(record, recipe['columns']):
            expected = _cell(column, index, predicted, ids, classes, scores)
            if column['kind'] == 'probability':
                try: actual = float(value)
                except ValueError: raise ValueError('Output contract: nonnumeric probability') from None
                _require(np.isfinite(actual) and abs(actual-expected) <= 1e-12, 'written probability changed')
            else:
                _require(value == str(expected), 'written label or original ID changed')
    return dict(header_checked=True, row_count_checked=True, original_id_checked=ids is not None,
                model_outputs_checked=True, probability_columns_checked=needs_scores(recipe))


def write_export(package, specification, output, predictions, classes=None, scores=None):
    labels = specification['metric_spec']['labels']
    recipe = validate_recipe(specification['output_recipe'], labels)
    predicted, classes, scores = _checked_arrays(predictions, labels, recipe, classes, scores)
    ids, indices, template_hash = _identity_context(package, specification, recipe, len(predicted))
    # Validate before atomic replacement, so a failed check cannot publish malformed output.
    checks = {}
    def check_before_publish(temporary):
        checks.update(verify_export(temporary, recipe, labels, predicted, ids=ids, order_indices=indices,
                                   classes=classes, scores=scores))
    with atomic_text(output, encoding=recipe['encoding'], newline='', validate=check_before_publish) as stream:
        writer = csv.writer(stream, delimiter=recipe['delimiter'])
        if recipe['header']: writer.writerow([c['name'] for c in recipe['columns']])
        writer.writerows([_cell(c, i, predicted, ids, classes, scores) for c in recipe['columns']] for i in indices)
    return dict(version=VERSION, columns=recipe['columns'], order=recipe['order'], encoding=recipe['encoding'],
                delimiter=recipe['delimiter'], template_sha256=template_hash, checks=checks)
