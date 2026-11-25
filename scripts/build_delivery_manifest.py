"""Build the delivery candidate from the autonomous nine-fit run and verify it in isolation.

The candidate is the model the entry itself selected after nine fits; nothing here
picks a model. Verification runs in a separate process whose import path is a
copy of the source package, so the delivered artifact cannot silently depend on
the working tree.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np

from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model

TASK = r'D:\codex\python_withai\data\released_task1.zip'
RUN = REPO / 'research' / 'release_acceptance' / 'unified_rerun' / 'agent'
DELIVERY = REPO / 'research' / 'release_acceptance' / 'delivery'

RELOAD = '''
import hashlib, json, sys
import joblib, numpy as np
sys.path.insert(0, sys.argv[1])
from superagent.data import load_development_task
from superagent.models import predict_with_model
bundle = joblib.load(sys.argv[2])
task = load_development_task(sys.argv[3])
pred = np.asarray(predict_with_model(bundle['kind'], bundle['model'], task.X_val)).reshape(-1)
values = [str(v) for v in pred]
print(json.dumps({'rows': len(values),
                  'sha256': hashlib.sha256(','.join(values).encode()).hexdigest()}))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=DELIVERY)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    if not args.execute:
        print(json.dumps({'status': 'validated_not_executed', 'run': str(RUN), 'output': str(out)}))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    outcome = json.loads((RUN / 'run_outcome.json').read_text(encoding='utf-8'))
    selected = outcome['selected_candidate']
    model_source = RUN / 'selected_model.joblib'
    shutil.copy2(model_source, out / 'model.joblib')
    bundle = joblib.load(out / 'model.joblib')
    if bundle.get('candidate') != selected:
        raise SystemExit(f'model candidate {bundle.get("candidate")} != outcome {selected}')
    task = load_development_task(TASK)
    metric = MetricSpec.for_task(task)
    predictions = [str(v) for v in
                   np.asarray(predict_with_model(bundle['kind'], bundle['model'], task.X_val)).reshape(-1)]
    with (out / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        handle.write('row_id,prediction\n')
        for index, value in enumerate(predictions):
            handle.write(f'{index},{value}\n')
    accuracy = float((np.asarray(predictions) == np.asarray([str(v) for v in task.y_val])).mean())

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / 'source'
        copy.mkdir()
        for name in ('superagent', 'tools'):
            shutil.copytree(REPO / name, copy / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if (REPO / 'unified_agent.py').is_file():
            shutil.copy2(REPO / 'unified_agent.py', copy / 'unified_agent.py')
        script = Path(tmp) / 'reload.py'
        script.write_text(RELOAD, encoding='utf-8')
        completed = subprocess.run([sys.executable, str(script), str(copy), str(out / 'model.joblib'), TASK],
                                   cwd=tmp, capture_output=True, text=True,
                                   env={'PYTHONPATH': str(copy), **{k: v for k, v in
                                        __import__('os').environ.items() if k != 'PYTHONPATH'}})
        if completed.returncode != 0:
            raise SystemExit('isolated reload failed: ' + completed.stderr[-1200:])
        isolated = json.loads(completed.stdout.strip().splitlines()[-1])

    record = {
        'delivery_entry': 'python -m unified_agent train released_task1.zip',
        'planner_mode': outcome['planner_mode'], 'scheduler': outcome['scheduler'],
        'api_calls_in_run': outcome['new_api_requests'],
        'fits_in_run': outcome['classifier_fit_accounting']['attempted_classifier_fits'],
        'selected_candidate': selected, 'run_dir': str(RUN.relative_to(REPO)),
        'task_archive_sha256': sha(TASK),
        'train_rows': int(len(task.y_train)), 'validation_rows': int(len(task.y_val)),
        'metric': metric.record(), 'class_order': list(metric.labels),
        'row_order': 'validation rows as loaded by load_development_task (0..N-1)',
        'model_sha256': sha(out / 'model.joblib'),
        'predictions_sha256': sha(out / 'predictions.csv'),
        'prediction_values_sha256': hashlib.sha256(','.join(predictions).encode()).hexdigest(),
        'validation_accuracy': round(accuracy, 6),
        'validation_correct': int(round(accuracy * len(predictions))),
        'isolated_reload': isolated,
        'isolated_matches_delivery': isolated['sha256'] == hashlib.sha256(
            ','.join(predictions).encode()).hexdigest(),
        'environment': {'python': platform.python_version(),
                        'catboost': __import__('importlib.metadata', fromlist=['version']).version('catboost'),
                        'scikit-learn': __import__('importlib.metadata', fromlist=['version']).version('scikit-learn')},
        'source_digest': hashlib.sha256(json.dumps(
            {path.name: sha(path) for path in sorted((REPO / 'superagent').glob('*.py'))}).encode()).hexdigest(),
    }
    (out / 'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
