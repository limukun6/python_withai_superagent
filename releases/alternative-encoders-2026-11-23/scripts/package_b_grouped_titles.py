"""Export experiment receipts and anonymous grouped correctness for the two training-only studies."""
import argparse
from pathlib import Path

import numpy as np

from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_confirmation import split
from scripts.run_b_shared_title_weighting import group_ids
from superagent.ab_run import sha
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint


def copy_receipt(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if not target.exists():
        with target.open('xb') as f:
            f.write(data)
    check(target.read_bytes()==data, 'Different existing receipt '+str(target))


def capture(source, target, extras):
    end = read(source/'completion.json')
    files = ['completion.json', 'protocol.json', 'results.json', *extras]
    files += [str(p.relative_to(source)) for pattern in ('fit_started.json','fit_completed.json')
              for p in source.rglob(pattern)]
    for k in ('results','protocol'):
        check(end[k]==sha(source/(k+'.json')), 'Unbound '+k)
    for name in files:
        copy_receipt(source/name, target/name)


def masks(arms, predictions, labels, groups, panels):
    values = np.zeros(len(labels), dtype=int)
    for j, arm in enumerate(arms):
        pred = predictions[arm]
        check(pred.shape==labels.shape and np.isin(pred,[0,1]).all(), 'Invalid exported binary predictions')
        values |= (pred==labels).astype(int) << j
    result = {}
    for group, panel, value in zip(groups, panels, values):
        found, members = result.setdefault(group, (str(panel), []))
        check(found==panel, 'Group occurs in multiple panels')
        members.append(int(value))
    return dict(arms=arms, groups=[[result[g][0], sorted(result[g][1])] for g in sorted(result)])


def package(a):
    check(sha(a.training)==INPUT_TRAIN, 'Training input changed')
    task = load_development_task(a.training)
    rows, y = task.X_train, np.asarray(task.y_train)
    check(task.X_val is None, 'Expected training rows only')
    arms = ['E5','context']
    seen = {}
    for key, ref in [('18643',a.first),('40531',a.second)]:
        old = read(ref/'primary_split.json')
        ids = np.array(old['search_indices'])
        check(fingerprint(ids)==old['row_ids_sha256'], 'Wrong primary row order')
        ep = read(a.screen/key/'exports.json')
        file = a.screen/key/'predictions.npz'
        check(sha(file)==ep['file'], 'Screen predictions changed')
        with np.load(file) as z:
            pred = dict(E5=z['base'], context=z['context'])
        gg = group_ids([rows[i] for i in ids])
        seen[key] = masks(arms, pred, y[ids], gg, np.repeat('screen',len(ids)))
    gr, parts, _ = split(rows, a.first, a.second)
    both = np.r_[parts['A'], parts['B']]
    ce = read(a.confirm/'exports.json')
    f = a.confirm/'predictions.npz'
    check(sha(f)==ce['predictions'] and ce['row_order']==fingerprint(both), 'Confirmation differs')
    with np.load(f) as z:
        pred = dict(E5=z['base'], context=z['context'])
    curve_exports = read(a.curve/'exports.json')
    for key, record in curve_exports.items():
        f = a.curve/key/'predictions.npy'
        check(sha(f)==record['file'], 'Learning-curve predictions differ')
        pred[key] = np.load(f,allow_pickle=False)
    panels = ['A']*len(parts['A']) + ['B']*len(parts['B'])
    seen['confirmation'] = masks(arms+list(curve_exports), pred, y[both], gr[both], panels)
    a.out.mkdir(parents=True, exist_ok=True)
    for src, key, extras in [(a.screen,'screen',['encoding.json','18643/exports.json','40531/exports.json']),
                             (a.confirm,'confirm',['exports.json']), (a.curve,'curve',['exports.json']),
                             (a.vote,'subset',['exports.json','grouped_correctness.json'])]:
        capture(src, a.out/key, extras)
    frozen_save(a.out/'grouped_correctness.json', seen)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','first','second','screen','confirm','curve','vote','out'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    package(p.parse_args())
