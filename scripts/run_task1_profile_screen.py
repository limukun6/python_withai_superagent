"""Paired, training-only repeated-profile screen. Run from the repository root."""
import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import catboost
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from superagent.ab_methods import _field_tree
from superagent.models import _clean_categories
from superagent.profile_history import ProfileTree, discover_columns, profile_keys
from superagent.splits import grouped_partition


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_split(task, split):
    with zipfile.ZipFile(task) as archive:
        arrays, fingerprints = [], {}
        for prefix in ('C', 'y'):
            basename = f'{prefix}_{split}.npy'
            name, = [n for n in archive.namelist() if Path(n).name == basename and '__MACOSX' not in n]
            data = archive.read(name)
            fingerprints[name] = digest(data)
            arrays.append(np.load(io.BytesIO(data), allow_pickle=prefix == 'C'))
    return _clean_categories(arrays[0]), arrays[1].reshape(-1), fingerprints


def coverage(x, model):
    keys = profile_keys(x, model.columns)
    return np.array([key in model.history.groups for key in keys])


def run(task, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    x, y, hashes = load_split(task, 'train')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'superagent/profile_history.py', root / 'superagent/field_interactions.py',
               root / 'superagent/models.py', root / 'superagent/splits.py',
               root / 'docs/batches/TASK1_PROFILE_80_PLAN_20261127_zh.md']
    save(out / 'protocol.json', dict(training_members=hashes, rows=len(y), metric='accuracy', max_fits=8,
        seeds=[707,811,823], cold_seed=20927, api_calls=0, validation_opened=False, test_opened=False,
        python=platform.python_version(), catboost=catboost.__version__, sklearn=sklearn.__version__,
        numpy=np.__version__, pandas=pd.__version__,
        sources={str(p.relative_to(root)):digest(p.read_bytes()) for p in sources}))
    fits = 0
    results = []
    try:
        proxy = SimpleNamespace(kind='categorical_tabular', X_train=x, y_train=y, spec={})
        splits = [(str(seed), *grouped_partition(proxy, seed)[:2]) for seed in (707,811,823)]
        cols = discover_columns(x)
        if not cols:
            raise ValueError('No recurring feature subset for cold-group diagnostic')
        groups = pd.factorize(pd.Series(profile_keys(x, cols)))[0]
        train, held = next(StratifiedGroupKFold(5, shuffle=True, random_state=20927).split(x, y, groups))
        splits.append(('cold', train, held))
        for label, train, held in splits:
            record = dict(split=label, train=len(train), held=len(held),
                          train_sha=digest(np.asarray(train, dtype='<i8').tobytes()),
                          held_sha=digest(np.asarray(held, dtype='<i8').tobytes()))
            base_correct = None
            for name in ('baseline','history'):
                since = time.monotonic()
                fits += 1
                save(out / 'last_attempt.json', dict(split=label, method=name, admitted_fits=fits))
                if name == 'baseline':
                    _, pred = _field_tree(x[train], y[train], x[held])
                else:
                    model = ProfileTree(dict(pair_limit=3, backbone_params=dict(depth=6,l2_leaf_reg=3.)))
                    model.fit(x[train], y[train])
                    pred = model.predict(x[held])
                    matched = coverage(x[held], model)
                    record.update(profile_columns=list(model.columns), covered=int(matched.sum()),
                                  cold_grouping_columns=list(cols) if label=='cold' else None)
                correct = np.asarray(pred).reshape(-1) == y[held]
                record[name] = dict(correct=int(correct.sum()), accuracy=float(correct.mean()),
                                    seconds=round(time.monotonic()-since,3),
                                    prediction_sha=digest(np.asarray(pred,dtype='<i8').tobytes()))
                if base_correct is None:
                    base_correct = correct
                else:
                    record.update(fixed=int((correct & ~base_correct).sum()), broken=int((~correct & base_correct).sum()),
                                  delta_pp=float(100*(correct.mean()-base_correct.mean())))
                print(label, name, record[name], flush=True)
                save(out / 'results.json', results + [record])
            results.append(record)
        d = [r['delta_pp'] for r in results if r['split'] != 'cold']
        completion = dict(status='complete', mean_delta_pp=float(np.mean(d)), improved=sum(v>0 for v in d),
                          promote=bool(np.mean(d)>=1 and sum(v>0 for v in d)>=2))
    except Exception as exc:
        save(out / 'completion.json', dict(status='failed',error=repr(exc),admitted_fits=fits,
                                           elapsed_seconds=time.monotonic()-start,api_calls=0))
        raise
    save(out / 'completion.json', dict(**completion,fits=fits,seconds=round(time.monotonic()-start,3),
                                       api_calls=0,tokens=0,course_yuan=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.task,args.output)
