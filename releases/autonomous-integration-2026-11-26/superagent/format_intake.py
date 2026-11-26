"""Interpret unfamiliar record packages, freeze the schema, and replay it at prediction."""
import csv
import hashlib
import io
from pathlib import PurePosixPath
import re
import sqlite3
import tarfile
import zipfile
from xml.etree.ElementTree import ParseError

import numpy as np

from .data import TrainingTask
from .format_guidance import fixed_records, fixed_width_plan, resolve_target
from .format_sources import NativeContract, UnsupportedFormat, checked_name, package_files, roles, select
from .native_schema import record_schema
from .record_streams import MAX_ROWS, decode_records, fields, flat, scalar
from .schema import load_manifest_split, read_manifest
from .shared_table import ROLES, grouping_column
from .table_input import declared_metric


DECODE_ERRORS = (ValueError, KeyError, IndexError, TypeError, OSError, EOFError, RecursionError, NotImplementedError,
                 ImportError, csv.Error, tarfile.TarError, zipfile.BadZipFile, sqlite3.DatabaseError, ParseError)


def declared_source(prose, role, unit):
    split = {'train': r'(?:training|train|训练)', 'test': r'(?:prediction|test(?:ing)?|预测|测试)'}[role]
    term = {'member': r'(?:file|文件)', 'collection': r'(?:collection|sheet|table|工作表|数据表)'}[unit]
    pattern = split + r'\s*' + term + r'\s*(?:is\s+|[:=：]\s*|为|是)["\']?([\w./-]+)'
    declarations = {s.rstrip('.') for s in re.findall(pattern, prose, re.I)}
    if len(declarations) > 1:
        raise ValueError('Conflicting ' + role + ' ' + unit + ' declarations')
    if declarations:
        return checked_name(declarations.pop())
    return None


def declared_delimiter(prose):
    found = set(re.findall(r'(?:delimiter|separator|分隔符)\s*(?:is\s+|为|是|[:=：]\s*)?["“\x27]([^\w\r\n"“”\x27])["”\x27]', prose, re.I))
    if len(found) > 1:
        raise ValueError('Conflicting documented delimiters')
    return next(iter(found), None)


def manifest_descriptor(files, role):
    from .task_files import TrainingMembers
    manifest = read_manifest(TrainingMembers(files) if role == 'train' else files, read_metric=role == 'train')
    if manifest is None:
        return {}, {}, None
    descriptor = manifest.get('splits', {}).get(role, {})
    if not isinstance(descriptor, dict):
        raise ValueError('Invalid declared split descriptor')
    if descriptor.get('format') not in {'json', 'jsonl', 'ndjson', 'json-records', 'json-lines',
                                       'delimited', 'xlsx', 'sqlite', 'arrow-ipc', 'xml-records', 'arff', 'records'}:
        raise NativeContract('The native manifest reader owns this declared split')
    path = checked_name(descriptor.get('path', ''))
    member = (PurePosixPath(manifest['_manifest_parent']) / path).as_posix()
    return manifest, descriptor, member


def source(files, role, declaration):
    names = files.data_names()
    if declaration is None and any(PurePosixPath(n).suffix.casefold() in {'.npy', '.npz'} for n in names):
        raise NativeContract('An explicit array layout owns this package')
    name = select(names, role, explicit=declaration, single=True)
    tokens = set(re.split(r'[\W_]+', PurePosixPath(name).stem.casefold()))
    if tokens & {'labels', 'truth', 'targets'} or 'y' in tokens and roles(name):
        raise ValueError('A label vector cannot serve as a feature table')
    return name, files.read(name)


def split_markers(mapping):
    if not isinstance(mapping, dict) or not {'train','test'} <= mapping.keys() or mapping.keys() - ROLES.keys():
        raise ValueError('Invalid declared split roles')
    result = {}
    for role, words in mapping.items():
        words = [words] if isinstance(words, str) else words
        if not isinstance(words, (list, set)) or not words:
            raise ValueError('Empty split vocabulary')
        for word in words:
            if not isinstance(word, str) or not word.strip() or word.strip().casefold() in result:
                raise ValueError('A split marker is missing or repeated')
            result[word.strip().casefold()] = role
    return result


def records_for_role(records, role, prose, *, contract=None, separate_query=False):
    """Use only the split marker until a row is assigned. Profile selected records only."""
    contract = contract or {}
    group = contract.get('split_column')
    markers = split_markers(contract.get('split_values') or ROLES)
    selected = []
    for count, record in enumerate(records, 1):
        if count > MAX_ROWS or not isinstance(record, dict):
            raise ValueError('Expected an object per row within the size limit')
        record = flat(record)
        nested_markers = [n for n in record if n.split('.')[-1].casefold() in {'split','partition','subset'}]
        if len(nested_markers) > 1 or count > 1 and nested_markers and nested_markers[0] != group:
            raise ValueError('Inconsistent or ambiguous row split fields')
        if count == 1:
            if group and group not in record:
                if not separate_query:
                    raise ValueError('Declared split field is missing')
                group = None
            group = grouping_column(list(record), prose, group) or next(iter(nested_markers),None)
            if group and group not in record or not group and re.search(r'(?:split|partition)\s+(?:column|field)\s*(?:is|[:=])|(?:划分列|分组列[：为是])', prose,re.I):
                raise ValueError('Declared split column is absent')
        if group:
            marker = scalar(record.get(group)).strip().casefold()
            if marker not in markers:
                raise ValueError('Unknown or missing row split marker')
            if markers[marker] != role:
                continue
        selected.append({k:v for k,v in record.items() if k != group})
    if not selected:
        raise ValueError('No rows for the requested split')
    return selected, group


class CanonicalTable:
    """Expose a single role-specific table through the ordinary validated model input interface."""
    def __init__(self, role, header, records):
        self.name = role + '.csv'
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(records)
        self.raw = stream.getvalue().encode('utf-8')

    def namelist(self):
        return [self.name]

    def read(self, name):
        if name != self.name:
            raise ValueError('No such canonical member')
        return self.raw


def canonical(role, schema, header, rows, metric, declared):
    view = CanonicalTable(role, header, rows)
    manifest = dict(schema_version=1, _manifest_parent='.', _schema_origin='native_v1', metric=metric,
                    splits={role:dict(format='csv', path=view.name)}, **schema)
    if declared:
        manifest['_metric_source'] = 'declared_task_readme'
    return load_manifest_split(view, manifest, role, include_labels=role == 'train')


def check_codec(descriptor, codec):
    expected = descriptor.get('format', 'records')
    expected = {'json':'json-records', 'jsonl':'json-lines', 'ndjson':'json-lines'}.get(expected, expected)
    if expected not in ('records', codec):
        raise ValueError('Declared record format conflicts with the file contents')


def declared_schema(manifest):
    return {key:manifest[key] for key in ('modality','columns','target','pair_label_one','pair_semantics',
            'positive_label','evaluation_protocol','image_preprocessing') if key in manifest}


def adapt_training(path, client=None, log=None):
    with package_files(path) as files:
        manifest, descriptor, declared_train = manifest_descriptor(files, 'train')
        prose = files.documentation()
        name, raw = source(files, 'train', declared_train or declared_source(prose, 'train', 'member'))
        selector = descriptor.get('collection') or declared_source(prose, 'train', 'collection')
        fixed, target, evidence = None, None, None
        try:
            codec, location, records = decode_records(raw, 'train', selector=selector,
                delimiter=descriptor.get('delimiter') or declared_delimiter(prose))
            check_codec(descriptor, codec)
        except UnsupportedFormat:
            if selector or not re.search(r'fixed[ -]width|定宽', prose, re.I):
                raise
            fixed, target, evidence = fixed_width_plan(prose, raw, client, log)
            codec, location, records = 'fixed-width', '', fixed_records(raw, fixed, target, 'train')
        selected, group = records_for_role(records, 'train', prose, contract=manifest)
        header = fields(list(selected[0]))
        if any(row.keys() != selected[0].keys() for row in selected):
            raise ValueError('Training record fields are inconsistent')
        if manifest:
            target, evidence = manifest.get('target'), 'declared manifest'
            columns = manifest.get('columns')
            if not isinstance(columns, list) or not columns or target not in header:
                raise ValueError('Declared feature columns or target are missing')
            header = fields([c['name'] for c in columns] + [target])
        elif target is None:
            target, evidence = resolve_target(prose, header, client, log)
        metric = manifest.get('metric') or declared_metric(prose)
        recipe = dict(version=1, codec=codec, train_member=name, train_location=location,
                      train_member_sha256=hashlib.sha256(raw).hexdigest(), split_column=group,
                      prediction_member=declared_source(prose, 'test', 'member'),
                      prediction_collection=declared_source(prose, 'test', 'collection'),
                      documented_delimiter=descriptor.get('delimiter') or declared_delimiter(prose),
                      target=target, target_evidence=evidence, fixed_fields=fixed)
        if manifest:
            query = manifest.get('splits', {}).get('test', {})
            if query.get('path'):
                recipe['prediction_member'] = (PurePosixPath(manifest['_manifest_parent']) / checked_name(query['path'])).as_posix()
            recipe['prediction_collection'] = query.get('collection')
            recipe['prediction_delimiter'] = query.get('delimiter')
            recipe['split_values'] = manifest.get('split_values')
        if (recipe['prediction_member'] == name and not group and not roles(location)
                and recipe['prediction_collection'] in (None, location)):
            raise ValueError('Shared source needs distinct rows or distinct collections')
        try:
            rows = [[scalar(row[n]) for n in header] for row in selected]
            schema = declared_schema(manifest) if manifest else record_schema(header, rows, target, prose)
            kind, x, y, spec = canonical('train', schema, header, rows, metric or 'accuracy', bool(metric))
            if len(np.unique(y)) < 2:
                raise ValueError('Training requires at least two distinct observed classes')
        except (ValueError, KeyError) as error:
            from .partial_classification import class_prior
            labels = np.asarray([scalar(record[target]) for record in selected])
            stats = class_prior(labels, metric or 'accuracy', manifest.get('positive_label'))
            frozen = declared_schema(manifest) if manifest else dict(target=target, columns=[{'name':n} for n in header if n != target])
            prior = dict(stats, format='adaptive_records', target=target, frozen_spec=dict(input_recipe=recipe,input_schema=frozen))
            issue = UnsupportedFormat(str(error))
            issue.verified_prior = prior
            raise issue from error
        if manifest and metric:
            spec['goal_sources']['primary_metric'] = manifest.get('_metric_source','declared_manifest')
        spec.update(input_recipe=recipe, input_format=codec)
        if log:
            log.write_json('input_recipe.json', recipe)
        return TrainingTask(kind, x, y, spec)


def prediction_rows(path, spec):
    recipe = spec['input_recipe']
    if recipe['version'] != 1:
        raise ValueError('Unknown frozen input recipe version')
    with package_files(path) as files:
        manifest, descriptor, declared_test = manifest_descriptor(files, 'test')
        if manifest and any(manifest.get(k) != v for k, v in spec['input_schema'].items()):
            raise ValueError('Prediction manifest conflicts with the frozen training schema')
        expected = recipe.get('prediction_member')
        if declared_test and expected and declared_test != expected:
            raise ValueError('Prediction member conflicts with the frozen declaration')
        name, raw = source(files, 'test', expected or declared_test)
        codec = recipe['codec']
        selector = recipe.get('prediction_collection') or descriptor.get('collection')
        if not selector and recipe['train_location'] and codec in {'json-records','xml-records','sqlite','xlsx'}:
            if not roles(recipe['train_location']):
                selector = recipe['train_location']
        if codec == 'fixed-width':
            location = ''
            records = fixed_records(raw, recipe['fixed_fields'], recipe['target'], 'test')
        else:
            detected_codec, location, records = decode_records(raw, 'test', selector=selector,
                delimiter=recipe.get('prediction_delimiter') or descriptor.get('delimiter') or recipe.get('documented_delimiter'),
                expected_columns=[c['name'] for c in spec['input_schema']['columns']])
            check_codec(descriptor, detected_codec)
        different_member = hashlib.sha256(raw).hexdigest() != recipe['train_member_sha256']
        separate_query = different_member and (roles(name) == {'test'} or name == expected)
        selected, group = records_for_role(records, 'test', '', contract=recipe, separate_query=separate_query)
        if (not group and location == recipe['train_location']
                and not different_member):
            raise ValueError('Prediction would reuse the complete training source')
        return selected


def predict_records(path, kind, spec):
    selected = prediction_rows(path,spec)
    header = [c['name'] for c in spec['input_schema']['columns']]
    if any(not set(header) <= row.keys() for row in selected):
        raise ValueError('Required frozen feature fields are missing from prediction records')
    rows = ([scalar(row[n]) for n in header] for row in selected)
    detected, x, _, result = canonical('test', spec['input_schema'], header, rows, spec['metric'], False)
    if detected != kind or result.get('feature_shape') != spec.get('feature_shape'):
        raise ValueError('Prediction records do not match the frozen model')
    return x
