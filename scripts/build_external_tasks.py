"""Build two external task packages from the original UCI downloads.

Both sources are CC BY 4.0 and are kept untouched under data/_raw/. This script
only reshapes them into the manifest-v1 package the executor already reads, so
the new tasks exercise the same entry point as the released tasks.

    python scripts/build_external_tasks.py --raw-dir data/_raw --output-dir data

Deterministic: the same raw archives and the same split seed produce the same
two zips, and the script prints the sha256 of every member it writes.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_SEED = 20261120
RAW = {
    'dry_bean': dict(file='uci_dry_bean_602.zip',
                     sha256='0A64EFF5BE87F48C3DBBFC0A12A56C5D5B5167EF8E61CD45D69B3E7C7130C06F'),
    'sms_spam': dict(file='uci_sms_spam_228.zip',
                     sha256='1587EA43E58E82B14FF1F5425C88E17F8496BFCDB67A583DBFF9EEFAF9963CE3'),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_arff(archive, member):
    """Minimal ARFF reader for a numeric table with a nominal class."""
    text = archive.read(member).decode('utf-8')
    attributes, rows, in_data = [], [], False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        if not in_data:
            if stripped.lower().startswith('@attribute'):
                declaration = stripped[len('@attribute'):].strip()
                if declaration.startswith("'"):
                    name, rest = declaration[1:].split("'", 1)
                else:
                    name, rest = declaration.split(None, 1)
                attributes.append((name, rest.strip()))
            elif stripped.lower().startswith('@data'):
                in_data = True
            continue
        rows.append(next(csv.reader([line])))
    return attributes, rows


def write_member(archive, name, payload):
    # A fixed timestamp keeps the zip byte-identical across rebuilds, so the
    # package hash can be recorded as an identity rather than a build artifact.
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload)
    print(f'  {name}: {len(payload)} bytes sha256={sha256(payload)[:16]}')


def write_csv(archive, name, header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    write_member(archive, name, buffer.getvalue().encode('utf-8'))


def split_rows(y, seed=SPLIT_SEED):
    index = np.arange(len(y))
    train, validation = train_test_split(index, test_size=.2, random_state=seed, stratify=y)
    return np.sort(train), np.sort(validation)


def build_dry_bean(raw_dir, output_dir):
    spec = RAW['dry_bean']
    path = Path(raw_dir) / spec['file']
    if sha256(path.read_bytes()) != spec['sha256'].lower():
        raise SystemExit(f'{path} does not match the recorded download hash')
    with zipfile.ZipFile(path) as source:
        attributes, rows = read_arff(source, 'DryBeanDataset/Dry_Bean_Dataset.arff')
    names = [name for name, _ in attributes[:-1]]
    kinds = [declaration for _, declaration in attributes[:-1]]
    if len(names) != 16 or any(kind.upper() not in ('INTEGER', 'REAL') for kind in kinds):
        raise SystemExit(f'unexpected ARFF schema: {attributes}')
    labels = [row[-1] for row in rows]
    features = [row[:-1] for row in rows]
    missing = sum(value.strip() in ('', '?') for row in features for value in row)
    train, validation = split_rows(np.asarray(labels))
    header = names + ['Class']
    with zipfile.ZipFile(Path(output_dir) / 'dry_bean_numeric_tabular.zip', 'w',
                         zipfile.ZIP_DEFLATED) as target:
        print('dry_bean_numeric_tabular.zip')
        write_csv(target, 'train.csv', header, [[*features[i], labels[i]] for i in train])
        write_csv(target, 'validation.csv', header, [[*features[i], labels[i]] for i in validation])
        manifest = {
            'schema_version': 1, 'objective': 'classification', 'metric': 'accuracy',
            'modality': 'tabular',
            'columns': [{'name': name, 'type': 'numeric'} for name in names],
            'target': 'Class',
            'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                       'validation': {'format': 'csv', 'path': 'validation.csv'}},
            'source': dict(
                dataset='Dry Bean Dataset', uci_id=602,
                url='https://archive.ics.uci.edu/dataset/602/dry+bean+dataset',
                download='https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip',
                licence='CC BY 4.0', licence_url='https://creativecommons.org/licenses/by/4.0/',
                citation='Koklu, M. and Ozkan, I.A. (2020), Multiclass Classification of Dry Beans '
                         'Using Computer Vision and Machine Learning Techniques, Computers and '
                         'Electronics in Agriculture 174, 105507',
                raw_sha256=spec['sha256'], raw_member='DryBeanDataset/Dry_Bean_Dataset.arff',
                rows_original=len(rows), rows_written=len(rows),
                duplicate_rows_removed=0, missing_values=missing,
                split='80/20 stratified, random_state=20261120, source row order preserved '
                      'inside each split',
                transformation='ARFF numeric attributes copied verbatim as CSV columns; '
                               'nominal Class column kept as the string target'),
        }
        write_member(target, 'manifest.json',
                     json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8'))
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    return dict(rows=len(rows), train=len(train), validation=len(validation),
                columns=len(names), classes=counts, missing=missing,
                metrics='accuracy', zip='dry_bean_numeric_tabular.zip')


def build_sms_spam(raw_dir, output_dir):
    spec = RAW['sms_spam']
    path = Path(raw_dir) / spec['file']
    if sha256(path.read_bytes()) != spec['sha256'].lower():
        raise SystemExit(f'{path} does not match the recorded download hash')
    with zipfile.ZipFile(path) as source:
        raw = source.read('SMSSpamCollection').decode('utf-8')
    rows, duplicates = [], 0
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        label, message = line.split('\t', 1)
        label, message = label.strip(), message.strip()
        if not message:
            continue
        if message in seen:
            duplicates += 1
            continue
        seen.add(message)
        rows.append((label, message))
    labels = [label for label, _ in rows]
    messages = [message for _, message in rows]
    train, validation = split_rows(np.asarray(labels))
    header = ['message', 'label']
    with zipfile.ZipFile(Path(output_dir) / 'sms_spam_single_text.zip', 'w',
                         zipfile.ZIP_DEFLATED) as target:
        print('sms_spam_single_text.zip')
        write_csv(target, 'train.csv', header, [[messages[i], labels[i]] for i in train])
        write_csv(target, 'validation.csv', header, [[messages[i], labels[i]] for i in validation])
        manifest = {
            'schema_version': 1, 'objective': 'classification', 'metric': 'macro_f1',
            'modality': 'text', 'columns': [{'name': 'message', 'type': 'text'}],
            'target': 'label',
            'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                       'validation': {'format': 'csv', 'path': 'validation.csv'}},
            'source': dict(
                dataset='SMS Spam Collection v.1', uci_id=228,
                url='https://archive.ics.uci.edu/dataset/228/sms+spam+collection',
                download='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',
                licence='CC BY 4.0', licence_url='https://creativecommons.org/licenses/by/4.0/',
                citation='Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011), Contributions to '
                         'the Study of SMS Spam Filtering: New Collection and Results, '
                         'Proceedings of the 11th ACM Symposium on Document Engineering',
                raw_sha256=spec['sha256'], raw_member='SMSSpamCollection',
                rows_original=5574, rows_written=len(rows),
                duplicate_rows_removed=duplicates,
                split='80/20 stratified, random_state=20261120, after removing exact duplicate '
                      'messages so no text appears in both splits',
                transformation='tab-separated label/message pairs become CSV rows; message text '
                               'is unchanged apart from the outer strip',
                metric_reason='13.4% positive class: macro-F1 is the primary metric because '
                              'accuracy is dominated by the majority class'),
        }
        write_member(target, 'manifest.json',
                     json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8'))
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    return dict(rows=len(rows), train=len(train), validation=len(validation),
                classes=counts, duplicates_removed=duplicates,
                metrics='macro_f1', zip='sms_spam_single_text.zip')


def verify(output_dir):
    """Load both packages through the branch executor's own reader."""
    integration = Path(__file__).resolve().parents[1] / 'runs' / 'integration'
    if not (integration / 'superagent').is_dir():
        integration = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(integration))
    from superagent.data import load_development_task
    from superagent.contracts import TaskSpec
    for name in ('dry_bean_numeric_tabular', 'sms_spam_single_text'):
        task = load_development_task(Path(output_dir) / f'{name}.zip')
        contract = TaskSpec.from_task(task)
        print(json.dumps({'package': name, 'kind': task.kind, 'metric': contract.metric.name,
                          'train': len(task.y_train), 'validation': len(task.y_val),
                          'classes': int(len(np.unique(task.y_train))),
                          'tests_accessed': task.spec['test_accessed']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/_raw')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--verify', action='store_true', help='also load the packages through superagent')
    args = parser.parse_args()
    summary = {'split_seed': SPLIT_SEED,
               'dry_bean': build_dry_bean(args.raw_dir, args.output_dir),
               'sms_spam': build_sms_spam(args.raw_dir, args.output_dir)}
    (Path(args.output_dir) / 'external_tasks_build.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.verify:
        verify(args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
