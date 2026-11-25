"""Release acceptance: regenerate the delivered Task1 predictions from source and data.

Trains the delivered winning configuration on all 8,117 training rows with the
entry's own worker, predicts the 2,030 designated validation rows, and compares
against the stored delivery file. No API.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model

TASK = r'D:\codex\python_withai\data\released_task1.zip'
DELIVERED = REPO / 'research' / 'unified_raw_1124' / 'designated' / 'table_0' / 'predictions.csv'
WINNER = 'catboost_depth8_reg3.0'
OUT = REPO / 'research' / 'release_acceptance'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT / 'attempt_01')
    parser.add_argument('--timeout', type=float, default=1800.0)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and args.execute:
        print(json.dumps({'status': 'refused', 'reason': f'{out} exists'}))
        return 1
    out.mkdir(parents=True, exist_ok=True)
    task = load_development_task(TASK)
    metric = MetricSpec.for_task(task)
    record = {'task_sha256': sha(TASK), 'candidate': WINNER,
              'train_rows': int(len(task.y_train)), 'validation_rows': int(len(task.y_val)),
              'metric': metric.record(), 'delivered_predictions_sha256': sha(DELIVERED)}
    (out / 'protocol.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n',
                                       encoding='utf-8')
    if not args.execute:
        print(json.dumps({'status': 'validated_not_executed', **record}))
        return 0
    payload = out / 'full_train.joblib'
    joblib.dump(dict(kind=task.kind, X_train=task.X_train, y_train=task.y_train,
                     X_search=task.X_val, y_search=task.y_val, metric_spec=metric.record()), payload)
    admission = out / 'full_train.admission.json'
    admission.write_text(json.dumps({'maximum_classifier_fits': 1,
                                     'ledger_path': str(out / 'full_train.fits.json')}), encoding='utf-8')
    started = time.monotonic()
    model, facts = train_candidate(payload, WINNER, out / 'selected_model.joblib', args.timeout,
                                   fit_admission_path=admission)
    seconds = round(time.monotonic() - started, 3)
    data = joblib.load(payload)
    predictions = [int(v) for v in np.asarray(predict_with_model(task.kind, model, data['X_search'])).reshape(-1)]
    delivered = [int(line.split(',')[1]) for line in DELIVERED.read_text(encoding='utf-8').splitlines()[1:] if line]
    agree = sum(1 for left, right in zip(predictions, delivered) if left == right)
    result = {'validation_rows': len(predictions), 'delivered_rows': len(delivered),
              'identical': predictions == delivered, 'agreeing_rows': agree,
              'disagreeing_rows': len(predictions) - agree,
              'validation_accuracy': round(float((np.asarray(predictions) == np.asarray(task.y_val)).mean()), 6),
              'delivered_accuracy': round(float((np.asarray(delivered) == np.asarray(task.y_val)).mean()), 6),
              'training_seconds': seconds,
              'model_sha256': sha(out / 'selected_model.joblib'),
              'predictions_sha256': hashlib.sha256(','.join(map(str, predictions)).encode()).hexdigest(),
              'fit_record': json.loads((out / 'full_train.fits.json').read_text(encoding='utf-8'))}
    (out / 'acceptance.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
