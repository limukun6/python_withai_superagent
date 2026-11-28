"""Replicate a fixed confidence gate on the remaining training folds."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_task1_exposure import paired
from scripts.run_task1_profile_screen import digest, load_split, save
from scripts.task1_exposure_models import PooledPreferences
from superagent.joint_context import ContextTokenTree
from superagent.splits import sample_groups


def run(task, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    exposure = root / 'research/task1_exposure_01'
    ref = json.loads((exposure / 'protocol.json').read_text(encoding='utf-8'))
    earlier = json.loads((exposure / 'results.json').read_text(encoding='utf-8'))[:3]
    initial_net = sum(r['methods']['blend']['low_confidence_correct'] -
                      r['methods']['baseline']['low_confidence_correct'] for r in earlier)
    x, y, members = load_split(task, 'train')
    assert members == ref['train_members'] and initial_net == 10
    groups = sample_groups(SimpleNamespace(X_train=x, kind='categorical_tabular'))
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=112801).split(x, y, groups))
    held_hashes = [digest(held.astype('<i8').tobytes()) for _, held in folds]
    assert held_hashes == [s['held_sha'] for s in ref['splits']]
    assert np.array_equal(np.sort(np.concatenate([held for _, held in folds])), np.arange(len(y)))
    files = [Path(__file__), root / 'scripts/task1_exposure_models.py', root / 'superagent/joint_context.py',
             root / 'docs/batches/TASK1_UNCERTAIN_GATE_PLAN_20261128_zh.md', exposure / 'results.json']
    save(out / 'protocol.json', dict(members=members, held_sha=held_hashes, seed=112801, prior_net=initial_net,
         sources={f.name: digest(f.read_bytes()) for f in files}, threshold_radius=.15, explore_folds=[0, 1, 2],
         confirm_folds=[3, 4], maximum_classifier_fits=4))
    rows = []
    fits = 0
    try:
        for fold in (3, 4):
            train, held = folds[fold]
            assert not set(groups[train]) & set(groups[held])
            fits += 1
            tree = ContextTokenTree(dict(pair_limit=3)).fit(x[train], y[train])
            base = tree.predict_proba(x[held])[:, 1]
            fits += 1
            second = PooledPreferences().fit(x[train], y[train])
            blend = (base + second.predict_proba(x[held])[:, 1]) / 2
            mask = np.abs(base - .5) <= .15
            gated = np.where(mask, blend, base)
            methods = {key: paired(pred, base, y[held], mask)
                       for key, pred in [('baseline', base), ('blend', blend), ('gate', gated)]}
            assert methods['gate']['fixed'] + methods['gate']['broken'] == int(((gated > .5) != (base > .5)).sum())
            rows.append(dict(fold=fold, held=len(held), gate_rows=int(mask.sum()), methods=methods))
            save(out / 'results.json', rows)
            print(fold, {key: scores['correct'] for key, scores in methods.items()}, flush=True)
        gains = [r['methods']['gate']['fixed'] - r['methods']['gate']['broken'] for r in rows]
        five_fold_delta = 100 * (initial_net + sum(gains)) / len(y)
        passed = min(gains) >= 0 and sum(gains) > 0 and five_fold_delta >= .5
        save(out / 'completion.json', dict(status='complete', confirmed=passed, classifier_fits=fits,
             prior_three_net=initial_net, confirm_net=sum(gains), five_fold_delta_pp=five_fold_delta,
             seconds=round(time.monotonic()-start, 3), api_calls=0, course_yuan=0,
             designated_validation_opened=False, formal_test_opened=False))
    except Exception as exc:
        save(out / 'completion.json', dict(status='failed', classifier_admissions=fits, error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.output)
