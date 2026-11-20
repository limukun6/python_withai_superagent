"""Predict with the released development-pool models, then score the sealed CSVs."""
import argparse
import csv
import io
import json
import os
import platform
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from superagent.adaptive_local import save
from superagent.predict import create_predictions
from superagent.pretrained_choice import digest


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / 'research/development_pool_models.json'
PACKAGES = {
    'task1': ('Released_Task1-In_Vehicle_Coupon_Recommendation.zip',
              'b64ab2938b3a96afb997b2e0ca4b0711df767f5e5c873453951e1c0e9bdfd6e5'),
    'task2': ('Released_Task2-Headlines_CTR.zip',
              '56b5439ea3f77c7b933d9c0bc49705f3759b3a1d302da49adee80031904834e4'),
}
FEATURES = {'task1': 'in_vehicle_coupon_recommendation/C_val.npy',
            'task2': 'Released_Task2-Headlines_CTR/val.csv'}
LABELS = {'task1': 'in_vehicle_coupon_recommendation/y_val.npy', 'task2': FEATURES['task2']}
# Keys match the portable exports. These roles are fixed before scoring.
ARMS = {'task1': ('task1', 'default'), 'task2': ('task2', 'lightweight_default'),
        'task2-wide': ('task2', 'new_two_encoder_option')}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_packages(packages):
    for filename, expected in PACKAGES.values():
        if digest(packages / filename) != expected:
            raise ValueError(f'Released archive differs from the training protocol: {filename}')


def read_member(packages, task, member):
    with zipfile.ZipFile(packages / PACKAGES[task][0]) as archive:
        if archive.namelist().count(member) != 1 or member not in (FEATURES[task], LABELS[task]):
            raise ValueError(f'Unexpected designated validation member: {member}')
        return archive.read(member)


def title_columns(content, columns):
    reader = csv.reader(io.StringIO(content.decode('utf-8-sig'), newline=''))
    header = next(reader)
    indices = [header.index(name) for name in columns]
    for row in reader:
        yield [row[index] for index in indices]


def stage_features(packages, task, target):
    data = read_member(packages, task, FEATURES[task])
    # The prediction CLI expects these filenames inside an unlabeled input package.
    with zipfile.ZipFile(target, 'w') as staged:
        if task == 'task1':
            staged.writestr('C_test.npy', data)
        else:
            columns = ('headline_A', 'headline_B')
            text = io.StringIO(newline='')
            writer = csv.writer(text)
            writer.writerow(columns)
            writer.writerows(title_columns(data, columns))
            staged.writestr('test.csv', text.getvalue())


def protocol():
    exports = read_json(EXPORTS)['tasks']
    arms = {}
    for name, (task, role) in ARMS.items():
        record = exports[name]
        directory = ROOT / record['solution_directory']
        for filename, checksum in record['portable_files'].items():
            if digest(directory / filename) != checksum:
                raise ValueError(f'Changed portable model or configuration: {name}/{filename}')
        arms[name] = dict(task=task, role=role, model=record['selected'],
                          directory=record['solution_directory'], files=record['portable_files'])
    return dict(metric='accuracy', split='designated course validation', arms=arms,
                exports_sha256=digest(EXPORTS), evaluator_sha256=digest(Path(__file__)),
                packages={task: dict(filename=spec[0], sha256=spec[1], features=FEATURES[task],
                                     labels=LABELS[task]) for task, spec in PACKAGES.items()},
                selection='Score all three existing exports. Keep the published default and optional roles.',
                procedure='Predict from validation features, seal all CSVs, then compare to the validation labels.')


def prediction_path(out, name):
    return out / name / 'validation_predictions.csv'


def predict(packages, out, l6, l12):
    frozen = protocol()
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'protocol.json', dict(frozen, frozen_utc=now()))
    receipt = {}
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for task in PACKAGES:
            stage_features(packages, task, scratch / f'{task}.zip')
        for name, arm in frozen['arms'].items():
            temporary_csv = scratch / name / 'predictions.csv'
            create_predictions(scratch / (arm['task'] + '.zip'), ROOT / arm['directory'], temporary_csv,
                               minilm_weights=str(l6.resolve()), deep_minilm_weights=str(l12.resolve()))
            manifest = read_json(temporary_csv.parent / 'prediction_manifest.json')
            dest = prediction_path(out, name)
            dest.parent.mkdir()
            shutil.copyfile(temporary_csv, dest)
            usage = manifest.get('prediction_feature_usage',
                                 dict(batches=0, input_tokens=0, output_tokens=0, api_rmb=0, remote_calls=0))
            receipt[name] = dict(prediction_sha256=digest(dest), rows=manifest['prediction_count'],
                                 feature_usage=usage)
            print(f"{name}: {manifest['prediction_count']} predictions sealed", flush=True)
    save(out / 'predictions_frozen.json', dict(
        frozen_utc=now(), protocol_sha256=digest(out / 'protocol.json'), arms=receipt,
        python=platform.python_version(), packages={p: version(p) for p in
            ('numpy', 'scikit-learn', 'joblib', 'catboost', 'torch', 'transformers', 'tokenizers')}))


def read_predictions(path, seal):
    if digest(path) != seal['prediction_sha256']:
        raise ValueError(f'Sealed prediction file changed: {path}')
    values = []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['row_id', 'prediction']:
            raise ValueError('Unexpected prediction columns')
        for index, row in enumerate(reader):
            if int(row['row_id']) != index or row['prediction'] not in ('0', '1'):
                raise ValueError(f'Invalid row or binary prediction at index {index}')
            values.append(int(row['prediction']))
    if len(values) != seal['rows'] or not values:
        raise ValueError('Incomplete predictions')
    return np.asarray(values)


def score(packages, out):
    receipt = read_json(out / 'predictions_frozen.json')
    stored = read_json(out / 'protocol.json')
    stored.pop('frozen_utc')
    if digest(out / 'protocol.json') != receipt['protocol_sha256'] or stored != protocol():
        raise ValueError('Frozen validation protocol has changed')
    predictions = {name: read_predictions(prediction_path(out, name), receipt['arms'][name])
                   for name in ARMS}
    results, masks = {}, {}
    for task in PACKAGES:
        content = read_member(packages, task, LABELS[task])
        truth = (np.load(io.BytesIO(content), allow_pickle=False).reshape(-1) if task == 'task1'
                 else np.asarray([int(row[0]) for row in title_columns(content, ['label_pairwise'])]))
        if not set(np.unique(truth)).issubset({0, 1}):
            raise ValueError('Expected binary validation labels')
        for name, (source, _) in ARMS.items():
            if source != task:
                continue
            if predictions[name].shape != truth.shape:
                raise ValueError(f'Validation labels and predictions differ in length: {name}')
            correct = masks[name] = predictions[name] == truth
            results[name] = dict(correct=int(correct.sum()), rows=len(truth), accuracy=float(correct.mean()))
    old, new = masks['task2'], masks['task2-wide']
    return dict(metric='accuracy', split='designated course validation', arms=results,
                paired_task2=dict(both_correct=int((old & new).sum()), both_wrong=int((~old & ~new).sum()),
                                  corrected_by_wide=int((~old & new).sum()), lost_by_wide=int((old & ~new).sum())),
                predictions_receipt_sha256=digest(out / 'predictions_frozen.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('predict', 'score'))
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--minilm-weights', type=Path)
    parser.add_argument('--deep-minilm-weights', type=Path)
    args = parser.parse_args()
    os.environ.update(SUPERAGENT_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from tools.offline_guard import sitecustomize  # activates the existing process-wide network guard

    verify_packages(args.packages)
    if args.phase == 'predict':
        if not all(p and p.is_dir() for p in (args.minilm_weights, args.deep_minilm_weights)):
            parser.error('predict requires both pinned local MiniLM weight directories')
        predict(args.packages, args.out, args.minilm_weights, args.deep_minilm_weights)
    else:
        result = score(args.packages, args.out)
        save(args.out / 'scores.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
