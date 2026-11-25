"""Probe the shapes an unseen task may arrive in, using the delivery entry only.

Every case builds or reuses a package, runs `unified_agent train` and, when a
test split exists, `unified_agent predict`. No API calls.
"""
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGE_TRAIN = Path(r'D:\codex\python_withai\runs\a95_object_photos\train.zip')
IMAGE_FEATURES = Path(r'D:\codex\python_withai\runs\a95_object_photos\features.zip')


def run(args, cwd=REPO):
    completed = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    return completed


def tabular_package(folder, *, labels, metric, with_validation=False, declared_metric=True):
    train = io.StringIO()
    train.write('city,color,label\n')
    for index in range(240):
        train.write(f'c{index % 7},k{index % 5},{labels[index % len(labels)]}\n')
    test = io.StringIO()
    test.write('city,color\n')
    for index in range(60):
        test.write(f'c{index % 7},k{index % 5}\n')
    splits = {'train': {'format': 'csv', 'path': 'train.csv'},
              'test': {'format': 'csv', 'path': 'test.csv'}}
    if with_validation:
        splits['validation'] = {'format': 'csv', 'path': 'validation.csv'}
    manifest = {'schema_version': 1, 'objective': 'classification',
                'modality': 'tabular', 'input_format': 'csv',
                'columns': [{'name': 'city', 'type': 'categorical'},
                            {'name': 'color', 'type': 'categorical'}],
                'target': 'label', 'splits': splits,
                'source': {'dataset': 'shape probe', 'licence': 'internal'}}
    if declared_metric:
        manifest['metric'] = metric
    package = folder / 'task.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr('train.csv', train.getvalue().encode())
        archive.writestr('test.csv', test.getvalue().encode())
        if with_validation:
            archive.writestr('validation.csv', train.getvalue().encode())
        archive.writestr('manifest.json', json.dumps(manifest).encode())
    return package


def case(root, name, package, predict_package=None):
    run_dir = root / name / 'run'
    trained = run(['-m', 'unified_agent', 'train', str(package), '--run-dir', str(run_dir)])
    record = {'case': name, 'train_rc': trained.returncode}
    if trained.returncode:
        record['train_error'] = (trained.stderr or '')[-400:]
        return record
    outcome = json.loads((run_dir / 'agent' / 'run_outcome.json').read_text(encoding='utf-8'))
    record.update({'fits': outcome['classifier_fit_accounting']['attempted_classifier_fits'],
                   'selected': outcome.get('selected_candidate'),
                   'api': outcome.get('new_api_requests')})
    target = predict_package or package
    output = root / name / 'predictions.csv'
    predicted = run(['-m', 'unified_agent', 'predict', str(target),
                     '--run-dir', str(run_dir), '--output', str(output)])
    record['predict_rc'] = predicted.returncode
    if predicted.returncode:
        record['predict_error'] = (predicted.stderr or '')[-400:]
        return record
    lines = output.read_text(encoding='utf-8').splitlines()
    record['prediction_rows'] = len(lines) - 1
    record['header'] = lines[0]
    return record


def main():
    root = Path(tempfile.mkdtemp(prefix='unseen_probe_'))
    results = []
    folder = root / 'tabular_unlabeled_test'
    folder.mkdir(parents=True)
    results.append(case(root, 'tabular_unlabeled_test',
                        tabular_package(folder, labels=['no', 'yes'], metric='accuracy')))
    folder = root / 'multiclass_macro_f1'
    folder.mkdir(parents=True)
    results.append(case(root, 'multiclass_macro_f1',
                        tabular_package(folder, labels=['lo', 'mid', 'hi'], metric='macro_f1')))
    folder = root / 'binary_roc_auc'
    folder.mkdir(parents=True)
    results.append(case(root, 'binary_roc_auc',
                        tabular_package(folder, labels=['no', 'yes'], metric='roc_auc')))
    if IMAGE_TRAIN.is_file() and IMAGE_FEATURES.is_file():
        results.append(case(root, 'image_folder', IMAGE_TRAIN, predict_package=IMAGE_FEATURES))
    print(json.dumps({'root': str(root), 'results': results}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
