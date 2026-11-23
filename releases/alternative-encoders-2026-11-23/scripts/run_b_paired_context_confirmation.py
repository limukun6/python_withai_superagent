"""Check paired-context features on the two previously registered title-group partitions."""
import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from scripts.run_b_paired_context import PLAN, fit_and_predict
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_confirmation import split
from superagent.ab_run import sha
from superagent.ab_sentence import AlternativeSentenceFeatures
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint


def confirm(a):
    screen = read(a.screen / 'completion.json')
    old = read(a.control / 'completion.json')
    saved = read(a.control / 'exports.json')['regular']
    model = a.control / 'regular' / 'classifier.joblib'
    predicted = a.control / 'regular' / 'predictions.npy'
    vectors = a.screen / 'context.npy'
    check(screen['results'] == sha(a.screen / 'results.json') and
          read(a.screen / 'results.json')['eligible_for_confirmation'] and
          read(a.screen / 'protocol.json')['plan'] == sha(PLAN), 'Screening evidence differs')
    check(old['protocol'] == sha(a.control / 'protocol.json') and
          old['exports'] == sha(a.control / 'exports.json') and saved['file'] == sha(predicted) and
          saved['fit'] == sha(model.parent / 'fit_completed.json') and
          read(model.parent / 'fit_completed.json')['model'] == sha(model), 'Control differs')
    check(sha(a.training) == INPUT_TRAIN and sha(vectors) == read(a.screen / 'encoding.json')['sha256'],
          'Training input or frozen representations changed')
    task = load_development_task(a.training)
    rows = task.X_train
    groups, ids, seen = split(rows, a.first, a.second)
    parts = {key: dict(rows=len(ii), groups=len(set(groups[ii])), ids=fingerprint(ii)) for key, ii in ids.items()}
    check(task.X_val is None and parts == read(a.control / 'protocol.json')['partitions'],
          'Control and challenger must use identical rows')
    a.out.mkdir(parents=True, exist_ok=True)
    frozen_save(a.out / 'protocol.json', dict(plan=sha(PLAN), runner=sha(Path(__file__)), training=INPUT_TRAIN,
        fit_function=sha(Path(__file__).with_name('run_b_paired_context.py')),
        screening=sha(a.screen / 'completion.json'), control=sha(a.control / 'completion.json'),
        partitions=parts, earlier_search_groups=seen, frozen_vectors=sha(vectors), fit_cap=1, model_api_cap=0))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    y = np.asarray(task.y_train)
    train = ids['train']
    both = np.r_[ids['A'], ids['B']]
    train_rows = [rows[i] for i in train]
    search_rows = [rows[i] for i in both]
    context = np.load(vectors, allow_pickle=False)
    features = AlternativeSentenceFeatures('pair_text', 'e5').fit(train_rows)
    pipe = Pipeline([('features', features), ('classifier', joblib.load(model))])
    base, new, meta = fit_and_predict(a.out / 'challenger', pipe, train_rows, search_rows, y[train],
                                      context[train], context[both])
    np.testing.assert_array_equal(base, np.load(predicted, allow_pickle=False))
    output = a.out / 'predictions.npz'
    if not output.exists():
        with output.open('xb') as f:
            np.savez_compressed(f, base=base, context=new)
    with np.load(output) as frozen:
        np.testing.assert_array_equal(frozen['base'], base)
        np.testing.assert_array_equal(frozen['context'], new)
    frozen_save(a.out / 'exports.json', dict(predictions=sha(output), base=fingerprint(base), new=fingerprint(new),
                                             row_order=fingerprint(both), **meta))
    baseline = base == y[both]
    ctx = new == y[both]
    delta = ctx.astype(int) - baseline.astype(int)
    stats = {}
    for key, panel in [('A', slice(0, len(ids['A']))), ('B', slice(len(ids['A']), None)), ('combined', slice(None))]:
        d = delta[panel]
        stats[key] = dict(rows=len(d), baseline=int(baseline[panel].sum()), context=int(ctx[panel].sum()),
                          gained=int((d > 0).sum()), lost=int((d < 0).sum()), delta=paired(d, groups[both][panel]))
    gains = [stats[k]['context'] - stats[k]['baseline'] for k in ('A','B')]
    result = dict(panels=stats, eligible_for_stronger_control=min(gains) >= 0 and sum(gains) >= 8,
                  physical_fits=1, model_api_calls=0, new_rmb=0, course_validation_or_test=False)
    frozen_save(a.out / 'results.json', result)
    frozen_save(a.out / 'completion.json', dict(results=sha(a.out / 'results.json'), protocol=sha(a.out / 'protocol.json'),
                exports=sha(a.out / 'exports.json'), fits=1, model_api_calls=0))
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'training', 'first', 'second', 'screen', 'control', 'out'):
        p.add_argument('--'+name, required=True, type=lambda v: Path(v).resolve())
    confirm(p.parse_args())
