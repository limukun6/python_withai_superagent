"""Read common record containers without importing or executing code from a task."""
import csv
from contextlib import closing
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import zipfile
from xml.etree import ElementTree as ET

from .format_sources import (MAX_BYTES, UnsupportedFormat, checked_name, decode_text,
                             roles, select, uncompress)
from .format_sources import zip_names, zip_read


MAX_ROWS = 500_000
MAX_FIELDS = 1024


def fields(names):
    if (not names or len(names) > MAX_FIELDS or any(not isinstance(n, str) or not n.strip()
            or len(n) > 256 or any(ord(c) < 32 for c in n) for n in names) or len(set(names)) != len(names)):
        raise ValueError('Expected unique, nonempty scalar field names')
    return list(names)


def dictionaries(matrix):
    matrix = iter(matrix)
    names = fields(next(matrix, []))
    for index, row in enumerate(matrix):
        if index >= MAX_ROWS or len(row) != len(names):
            raise ValueError('Row count or record width does not match the supported table contract')
        yield dict(zip(names, row))


def scalar(value):
    if value is None:
        return ''
    if isinstance(value, (str, bool, int)) or isinstance(value, float) and math.isfinite(value):
        return str(value)
    raise ValueError('Selected field contains a nested, binary, formula or non-finite value')


def json_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('Duplicate JSON field')
    return result


def flat(record, parent='', depth=0):
    if not isinstance(record, dict) or depth > 12:
        raise ValueError('Expected a JSON object within the supported nesting limit')
    out = {}
    for name, value in record.items():
        # Escape literal dots and backslashes so nested and literal field names never collide.
        key = parent + name.replace('\\', '\\\\').replace('.', '\\.')
        if isinstance(value, dict):
            out.update(flat(value, key + '.', depth + 1))
        else:
            out[key] = value
    return out


def json_rows(text, role, selector):
    def parse(value):
        return json.loads(value, object_pairs_hook=json_object)
    try:
        document = parse(text)
    except json.JSONDecodeError:
        line_records = (parse(line) for line in io.StringIO(text) if line.strip())
        return 'json-lines', '', line_records
    locations = {}
    def visit(value, path, depth):
        if depth > 12 or len(locations) > 256:
            raise ValueError('JSON container is too deep or ambiguous')
        if isinstance(value, list):
            locations[path] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                if '/' in key or roles(key) - {role}:
                    continue
                visit(child, path + '/' + key if path else key, depth + 1)
    visit(document, '', 0)
    if not locations and isinstance(document, dict):
        if selector or any(roles(k) for k in document):
            raise ValueError('JSON has no matching record collection')
        return 'json-records', '', [document]
    if list(locations) == [''] and not selector:
        path = ''
    else:
        path = select(list(locations), role, explicit=selector, single=True)
    return 'json-records', path, locations[path]


def xml(raw):
    class NoDTD(ET.TreeBuilder):
        def doctype(self, name, pubid, system):
            raise ValueError('XML document type declarations are unsupported')
    return ET.fromstring(raw, parser=ET.XMLParser(target=NoDTD()))


def xml_records(raw, role, selector):
    root = xml(raw)
    locations = {}
    def visit(node, path, depth):
        if depth > 12 or len(locations) > 256:
            raise ValueError('XML record container is too deep or ambiguous')
        groups = {}
        for child in node:
            if roles(child.tag) - {role}:
                continue
            groups.setdefault(child.tag, []).append(child)
        for tag, siblings in groups.items():
            location = (path + '/' + tag).strip('/')
            first = siblings[0]
            if len(siblings) > 1 or first.attrib or len(first) and all(not len(c) for c in first):
                locations[location] = siblings
            else:
                visit(first, location, depth + 1)
    visit(root, '', 0)
    location = select(list(locations), role, explicit=selector, single=True)
    def records():
        for node in locations[location]:
            row = dict(node.attrib)
            for child in node:
                if child.tag in row or len(child) or child.attrib:
                    raise ValueError('XML record fields must have distinct scalar names')
                row[child.tag] = child.text
            yield row
    return 'xml-records', location, records()


def sheet_rows(raw, role, selector):
    main = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    relation = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = zip_names(archive)
        def part(name):
            if name not in names:
                raise ValueError('Missing workbook part')
            return xml(zip_read(archive, name, MAX_BYTES))
        sheets = {s.get('name'): s.get(relation) for s in part('xl/workbook.xml').iter(main+'sheet')
                  if s.get('state', 'visible') == 'visible'}
        name = select(list(sheets), role, explicit=selector, single=True)
        targets = [r.get('Target') for r in part('xl/_rels/workbook.xml.rels')
                   if r.get('Id') == sheets[name] and r.get('TargetMode', 'Internal') == 'Internal'
                   and r.get('Type', '').endswith('/worksheet')]
        if len(targets) != 1 or not targets[0]:
            raise ValueError('Missing or ambiguous internal worksheet relationship')
        path = targets[0][1:] if targets[0].startswith('/xl/') else 'xl/' + targets[0]
        if not checked_name(path).startswith('xl/worksheets/'):
            raise ValueError('Worksheet is outside the workbook')
        strings = ([] if 'xl/sharedStrings.xml' not in names else
                   [''.join(t.text or '' for t in si.iter(main+'t'))
                    for si in part('xl/sharedStrings.xml').iter(main+'si')])
        def cells(row):
            values = []
            for cell in row.findall(main+'c'):
                coord = re.fullmatch(r'([A-Z]{1,3})[1-9][0-9]*', cell.get('r', ''))
                if not coord:
                    raise ValueError('Invalid workbook cell position')
                pos = 0
                for char in coord[1]:
                    pos = pos * 26 + ord(char) - 64
                if not len(values) < pos <= MAX_FIELDS:
                    raise ValueError('Unordered or oversized worksheet')
                values.extend([None] * (pos - 1 - len(values)))
                value = cell.findtext(main+'v')
                if cell.find(main+'f') is not None or cell.get('t') == 'e':
                    value = object()  # Not eligible as a selected scalar.
                elif cell.get('t') == 's':
                    if value is None or not value.isdigit() or int(value) >= len(strings):
                        raise ValueError('Invalid workbook string index')
                    value = strings[int(value)]
                elif cell.get('t') == 'inlineStr':
                    value = ''.join(t.text or '' for t in cell.iter(main+'t'))
                values.append(value)
            return values
        sheet = part(path).find(main+'sheetData')
        if sheet is None:
            raise ValueError('Worksheet has no data')
        rows = iter(sheet.findall(main+'row'))
        header = fields(cells(next(rows, ET.Element('row'))))
        records = []
        for row in rows:
            values = cells(row)
            if not values:
                continue
            if len(values) > len(header) or len(records) >= MAX_ROWS:
                raise ValueError('Worksheet exceeds its header or row limit')
            records.append(dict(zip(header, values + [None] * (len(header) - len(values)))))
        return 'xlsx', name, records


def sqlite_rows(raw, role, selector):
    import tempfile
    # An immutable, read-only standalone DB works on Python builds without deserialize().
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'records.db'
        path.write_bytes(raw)
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            db.execute('PRAGMA trusted_schema=OFF')
            db.execute('PRAGMA query_only=ON')
            tables = {r[1]: r for r in db.execute('PRAGMA main.table_list')
                      if r[2] == 'table' and not r[1].startswith('sqlite_')}
            name = select(list(tables), role, explicit=selector, single=True)
            def quote(identifier):
                return '"' + identifier.replace('"', '""') + '"'
            table_name = quote(name)
            schema = list(db.execute('PRAGMA main.table_xinfo(' + table_name + ')'))
            if any(c[6] for c in schema):
                raise ValueError('Generated or hidden database columns are unsupported')
            names = fields([c[1] for c in schema])
            if tables[name][4]:  # WITHOUT ROWID tables are ordered by their declared primary key.
                order = ','.join(quote(c[1]) for c in sorted(schema, key=lambda c:c[5]) if c[5])
            else:
                order = next((r for r in ('_rowid_', 'rowid', 'oid') if r not in {n.casefold() for n in names}), None)
            if not order:
                raise ValueError('Database rows lack a reproducible ordering')
            sql = 'SELECT ' + ','.join(map(quote, names)) + ' FROM ' + table_name + ' ORDER BY ' + order + ' LIMIT ?'
            records = [dict(zip(names, row)) for row in db.execute(sql, (MAX_ROWS + 1,))]
            if len(records) > MAX_ROWS:
                raise ValueError('Database row limit exceeded')
            return 'sqlite', name, records


def decode_records(raw, role, *, selector=None, delimiter=None, expected_columns=()):
    raw = uncompress(raw)
    if raw.startswith(b'SQLite format 3\x00'):
        return sqlite_rows(raw, role, selector)
    if raw.startswith(b'PK') and zipfile.is_zipfile(io.BytesIO(raw)):
        return sheet_rows(raw, role, selector)
    if raw.startswith((b'PAR1', b'ARROW1', b'\xff\xff\xff\xff')):
        import pyarrow as pa
        if selector:
            raise ValueError('A flat Arrow table has no named collection')
        if raw.startswith(b'PAR1'):
            from pyarrow.parquet import ParquetFile
            source = ParquetFile(io.BytesIO(raw))
            batches = source.iter_batches(batch_size=16384)
            names = source.schema_arrow.names
            codec = 'parquet'
        else:
            source = (pa.ipc.open_file if raw.startswith(b'ARROW1') else pa.ipc.open_stream)(pa.BufferReader(raw))
            batches = (source.get_batch(i) for i in range(source.num_record_batches)) if raw.startswith(b'ARROW1') else source
            names = source.schema.names
            codec = 'arrow-ipc'
        fields(names)
        return codec, '', (row for batch in batches for row in batch.to_pylist())
    text = decode_text(raw)
    if text.lstrip().startswith('<'):
        return xml_records(raw, role, selector)
    if re.search(r'^\s*@relation\b', text, re.I | re.M):
        if selector:
            raise ValueError('An ARFF relation has no selectable inner collection')
        from scipy.io.arff import loadarff
        data, metadata = loadarff(io.StringIO(text))
        names = fields(metadata.names())
        def arff_rows():
            for row in data:
                yield {n:(v.decode('utf-8') if isinstance(v, bytes) else v.item()) for n,v in zip(names, row)}
        return 'arff', '', arff_rows()
    if text.lstrip().startswith(('{', '[')):
        return json_rows(text, role, selector)
    if selector:
        raise ValueError('A delimited file has no named collection')
    first = text.split('\n', 1)[0]
    candidates = [c for c in (',', '\t', ';', '|', '^', '~', '\x1f') if c in first]
    if delimiter is None and not candidates and first.strip().strip('"') in expected_columns:
        delimiter = ','
    if delimiter is None:
        if not candidates:
            raise UnsupportedFormat('Unrecognized binary signature or record delimiter')
        try:
            delimiter = csv.Sniffer().sniff(first[:8192], delimiters=''.join(candidates)).delimiter
        except csv.Error as error:
            if len(candidates) != 1:
                raise UnsupportedFormat('Ambiguous record separator') from error
            delimiter = candidates[0]
    reader = csv.reader(io.StringIO(text, newline=''), delimiter=delimiter, strict=True)
    return 'delimited', delimiter, dictionaries(reader)
