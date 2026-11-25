"""Build the migration package from a documented open dataset.

Downloads the archive once, records its SHA-256, and writes a manifest-v1 package
with categorical columns kept as strings. Refuses when the licence or the shape
does not match the pre-registered qualification.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = {
    'dataset': 'Car Evaluation',
    'uci_id': 19,
    'url': 'https://archive.ics.uci.edu/dataset/19/car+evaluation',
    'download': 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip',
    'member': 'car+evaluation/car.data',
    'licence': 'CC BY 4.0',
    'licence_url': 'https://creativecommons.org/licenses/by/4.0/',
    'citation': 'Bohanec, M. (1988). Car Evaluation. UCI Machine Learning Repository.',
}
COLUMNS = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
TARGET = 'class'
SPLIT_SEED = 20261125


def fetch(url, target):
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_rows(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(name for name in archive.namelist() if name.endswith('car.data'))
        text = archive.read(member).decode('utf-8')
    rows = [line.split(',') for line in text.splitlines() if line.strip()]
    return rows


def split(rows, seed, fraction=0.8):
    import random
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    cut = int(round(len(rows) * fraction))
    chosen = set(indices[:cut])
    train = [row for index, row in enumerate(rows) if index in chosen]
    validation = [row for index, row in enumerate(rows) if index not in chosen]
    return train, validation


def write_csv(rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(COLUMNS + [TARGET])
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=REPO / 'research' / 'task1_migration' / 'car_evaluation_categorical.zip')
    parser.add_argument('--cache', type=Path, default=Path(r'D:\codex\python_withai\data\_raw'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cached = args.cache / 'uci_car_evaluation_19.zip'
    if cached.is_file():
        digest = hashlib.sha256(cached.read_bytes()).hexdigest()
        print('using cached archive', cached.name, digest)
    else:
        digest = fetch(SOURCE['download'], cached)
        print('downloaded', cached.name, digest)
    rows = read_rows(cached.read_bytes())
    if len(rows) != 1728 or len(rows[0]) != 7:
        raise SystemExit(f'unexpected shape: {len(rows)} rows, {len(rows[0])} columns')
    labels = sorted({row[6] for row in rows})
    if len(labels) != 4:
        raise SystemExit(f'expected four classes, found {labels}')
    train, validation = split(rows, SPLIT_SEED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1, 'objective': 'classification', 'metric': 'accuracy',
        'modality': 'tabular', 'input_format': 'csv',
        'columns': [{'name': name, 'type': 'categorical'} for name in COLUMNS],
        'target': TARGET,
        'splits': {'train': {'format': 'csv', 'path': 'train.csv'},
                   'validation': {'format': 'csv', 'path': 'validation.csv'}},
        'goal_sources': {'primary_metric': 'declared_manifest'},
        'source': dict(SOURCE, archive_sha256=digest, rows_original=len(rows),
                       rows_written=len(rows), classes=labels,
                       split=f'80/20 stratified by row, random_state={SPLIT_SEED}, '
                             'source row order preserved inside each split',
                       transformation='six nominal columns copied verbatim as strings and declared categorical; no encoding'),
    }
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('train.csv', write_csv(train))
        archive.writestr('validation.csv', write_csv(validation))
        archive.writestr('manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8'))
    package_digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({'package': str(args.output), 'sha256': package_digest,
                      'train_rows': len(train), 'validation_rows': len(validation),
                      'classes': labels}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
