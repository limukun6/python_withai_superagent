"""Delivery check for packages that declare no test split.

`superagent predict` needs declared prediction features. A development package
that only ships train/validation has none, so the equivalent check is: reload
the frozen model in a fresh process, score the validation features, and compare
with the predictions the run already recorded.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import joblib

from superagent.data import load_development_task
from superagent.models import predict_with_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    task = load_development_task(args.package)
    if task.X_val is None:
        raise SystemExit('Package has no validation split either')
    model = joblib.load(args.run_dir / 'selected_model.joblib')['model']
    predictions = [str(v) for v in predict_with_model(task.kind, model, task.X_val).tolist()]
    saved = json.loads((args.run_dir / 'validation_predictions.json').read_text(encoding='utf-8'))
    saved = [str(v) for v in saved]
    record = {'package': str(args.package), 'run_dir': str(args.run_dir),
              'rows': len(predictions), 'recorded_rows': len(saved),
              'identical': predictions == saved,
              'prediction_sha256': hashlib.sha256(json.dumps(predictions).encode()).hexdigest(),
              'recorded_sha256': hashlib.sha256(json.dumps(saved).encode()).hexdigest(),
              'channel': 'reload_validation'}
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))
    return 0 if record['identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
