"""Rebuild the historical validation-only features and prediction reference from the course release."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def feature_archive(course):
    with zipfile.ZipFile(course) as archive:
        with archive.open('Released_Task2-Headlines_CTR/val.csv') as member:
            rows = csv.DictReader(io.TextIOWrapper(member, encoding='utf-8-sig', newline=''))
            features = [(r['headline_A'], r['headline_B']) for r in rows]
    if len(features) != 940:
        raise ValueError('Unexpected course validation length')
    manifest = dict(schema_version=1, objective='classification', metric='accuracy',
                    splits=dict(test=dict(format='csv', path='rows.csv')), modality='pair_text',
                    columns=[dict(name='option_a', type='text'), dict(name='option_b', type='text')],
                    target='outcome', pair_semantics='preference', pair_label_one='left')
    table = io.StringIO(newline='')
    writer = csv.writer(table)
    writer.writerow(['option_a', 'option_b'])
    writer.writerows(features)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, text in (('manifest.json', json.dumps(manifest)), ('rows.csv', table.getvalue())):
            item = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            item.compress_type, item.create_system, item.external_attr = zipfile.ZIP_DEFLATED, 3, 0o600 << 16
            archive.writestr(item, text.encode())
    return output.getvalue()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--course-archive', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    a = cli.parse_args()
    protocol = json.loads((HERE/'evidence/protocol.json').read_text())
    if digest(a.course_archive.read_bytes()) != protocol['released_source']:
        raise ValueError('Unexpected course archive')
    features = feature_archive(a.course_archive)
    previous = (HERE/'evidence/previous_predictions.csv').read_bytes()
    if digest(features) != protocol['feature_packet'] or digest(previous) != protocol['previous_csv']:
        raise ValueError('Historical reference digest differs')
    a.out.mkdir(parents=True, exist_ok=False)
    for path, data in ((a.out/'designated/inputs/features.zip', features),
                       (a.out/'designated/autonomous_export_0/rows.csv', previous)):
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    print('Reconstructed the two original reference artifacts for 940 validation rows.')
