"""Identify which frozen classifier changes exported labels using features only and no additional fit."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from superagent import ab_run
from superagent.data import load_prediction_features
from scripts.run_b_autonomous_close_singles import ROOT, env, read, require, seal


def main(args):
    os.environ.update(env(ROOT, args.weights, args.run / 'unused_fit_log'))
    # This script imports no training entry point. A process guard also rejects accidental fitting or network.
    from scripts import a38_predict_guard
    paths = [args.run / name / 'agent' for name in ('control','proposal')]
    freeze = read(args.run / 'models_frozen.json')['arms']
    require(all(ab_run.sha(d / 'selected_model.joblib') == freeze[name]['model']
                for d, name in zip(paths, ('control','proposal'), strict=True)), 'Frozen model changed')
    models = [joblib.load(d / 'selected_model.joblib')['model'].models for d in paths]
    evidence = [read(d / 'prediction_evidence.json') for d in paths]
    require(evidence[0] == evidence[1] and len(evidence[0]) == 7, 'Trial evidence differs')
    require(ab_run.sha(paths[0] / 'refit_data.joblib') == ab_run.sha(paths[1] / 'refit_data.joblib'), 'Refit input differs')
    spec = read(paths[0] / 'task_spec.json')
    rows = load_prediction_features(args.run / 'features.zip', 'pair_text', spec)
    ordered = [(min(a,b), max(a,b)) for a,b in rows]
    direct = np.array([a < b for a,b in rows])
    equal = np.array([a == b for a,b in rows])
    per_component, details = [], []
    for a, b in zip(*models, strict=True):
        a_features, b_features = a.named_steps['features'], b.named_steps['features']
        left, right = a.named_steps['classifier'], b.named_steps['classifier']
        require(type(a_features) is type(b_features) and a_features.encoder_revision == b_features.encoder_revision
                and np.array_equal(a_features.scaler_.scale_, b_features.scaler_.scale_), 'Frozen feature definitions differ')
        for name in ('word_', 'char_'):
            x, y = getattr(a_features, name), getattr(b_features, name)
            require(x.vocabulary_ == y.vocabulary_ and np.array_equal(x.idf_, y.idf_), 'Frozen vocabulary or IDF differs')
        # Both heads see precisely the same matrix instance, so only their fitted parameters can differ.
        shared = a_features.transform(ordered)
        probs = []
        for head in (left, right):
            require(head.classes_.tolist() == [0,1], 'Unexpected class order')
            p = head.predict_proba(shared)[:, 1]
            probs.append(np.where(equal, .5, np.where(direct, p, 1-p)))
        per_component.append(probs)
        details.append(dict(encoder=a_features.encoder_model, iterations=[int(head.n_iter_[0]) for head in (left,right)],
                       coef_max_abs=float(np.max(np.abs(left.coef_-right.coef_))),
                       intercept_max_abs=float(np.max(np.abs(left.intercept_-right.intercept_))),
                       held_max_abs_probability_difference=float(np.max(np.abs(probs[0]-probs[1])))))
    observed = []
    for name in ('control','proposal'):
        with (args.run / name / 'export_0/rows.csv').open(newline='') as f:
            observed.append(np.array([int(row['prediction']) for row in csv.DictReader(f)]))
    crosses = {}
    for j in (0,1):
        for k in (0,1):
            p = np.mean([per_component[0][j], per_component[1][k]], axis=0)
            labels = ((p>.5) | ((p==.5)&direct)).astype(int)
            delta = [int(np.count_nonzero(labels != v)) for v in observed]
            crosses[f'{j}{k}'] = delta
            require(delta[k] == 0, 'Switching only the fitted head did not reproduce its native export')
    difference = observed[0] != observed[1]
    require(int(difference.sum()) == 2, 'The two verified exports differ elsewhere')
    seal(args.output, dict(runner_sha256=ab_run.sha(Path(__file__)), freeze_sha256=ab_run.sha(args.run/'models_frozen.json'),
         search_and_confirmation_records_identical=len(evidence[0]), identical_refit_input=ab_run.sha(paths[0]/'refit_data.joblib'),
         matched_rows=len(rows), differing_native_predictions=int(difference.sum()), components=details,
         predicted_mismatches_vs_control_then_proposal=crosses, new_fits=0, model_api_calls=0, held_labels_read=False))
    print(json.dumps(crosses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'weights', 'output'):
        parser.add_argument('--'+flag, required=True, type=lambda s: Path(s).resolve())
    main(parser.parse_args())
