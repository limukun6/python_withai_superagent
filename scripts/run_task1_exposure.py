"""Training-only paired screen and conditional confirmation for recurring-context tasks."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_task1_profile_screen import digest, load_split, save
from scripts.task1_exposure_models import ExposureTree, PooledPreferences, context_axis
from superagent.joint_context import ContextTokenTree
from superagent.splits import sample_groups


def paired(p, baseline, y, uncertain):
    pred = p > .5
    correct = pred == y
    ref = (baseline > .5) == y
    if not np.all(np.isfinite(p)) or p.shape != y.shape:
        raise ValueError('Unexpected probabilities')
    return dict(correct=int(correct.sum()), accuracy=float(correct.mean()),
                delta_pp=float(100*(correct.mean()-ref.mean())),
                fixed=int((correct & ~ref).sum()), broken=int((~correct & ref).sum()),
                low_confidence_correct=int(correct[uncertain].sum()),
                prediction_sha=digest(pred.astype('uint8').tobytes()))


def run(task, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    x, y, members = load_split(task, 'train')
    groups = sample_groups(SimpleNamespace(X_train=x, kind='categorical_tabular'))
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=112801).split(x, y, groups))
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root/'scripts/task1_exposure_models.py', root/'superagent/joint_context.py',
             root/'superagent/profile_history.py', root/'superagent/splits.py',
             root/'docs/batches/TASK1_EXPOSURE_PLAN_20261128_zh.md']
    save(out/'protocol.json', dict(sources={p.name:digest(p.read_bytes()) for p in files}, train_members=members,
        splits=[dict(train_sha=digest(a.astype('<i8').tobytes()), held_sha=digest(b.astype('<i8').tobytes()))
                for a,b in splits], seed=112801, design_folds=3, max_confirmation_fits=4, api_calls=0))
    results = []
    fits = 0
    unsupervised = 0
    winner = None
    summary = {}
    try:
        for fold,(train,held) in enumerate(splits):
            if fold >= 3 and winner is None:
                break
            assert not set(groups[train]) & set(groups[held])
            models = {'baseline':ContextTokenTree(dict(pair_limit=3))}
            if fold < 3 or winner == 'exposure':
                models['exposure'] = ExposureTree(dict(pair_limit=3))
            if fold < 3 or winner in ('pooled','blend'):
                models['pooled'] = PooledPreferences()
            probs = {}
            record = dict(fold=fold, held=len(held), methods={})
            for name,model in models.items():
                fits += 1
                save(out/'last_attempt.json',dict(fold=fold, method=name, classifier_admissions=fits))
                tick = time.monotonic()
                model.fit(x[train], y[train])
                unsupervised += int(name == 'exposure')
                assert list(model.classes_) == [0,1]
                probs[name] = model.predict_proba(x[held])[:,1]
                if name == 'baseline':
                    axis = context_axis(x[train], y[train], model.context)
                    uncertain = np.abs(probs[name] - .5) <= .15
                    record.update(axis=int(axis), uncertain_rows=int(uncertain.sum()))
                    record['baseline_by_context'] = {str(k):dict(rows=int(mask.sum()),
                        correct=int(((probs[name][mask]>.5)==y[held][mask]).sum()))
                        for k in np.unique(x[held,axis]) if (mask := x[held,axis]==k).sum() >= 30}
                stats = paired(probs[name],probs['baseline'],y[held],uncertain)
                stats['seconds'] = round(time.monotonic()-tick,3)
                record['methods'][name] = stats
                save(out/'results.json',results+[record])
                print(fold,name,stats['correct'],len(held),flush=True)
            if fold < 3 or winner=='blend':
                record['methods']['blend'] = paired((probs['pooled']+probs['baseline'])/2,probs['baseline'],y[held],uncertain)
            results.append(record)
            save(out/'results.json',results)
            if fold == 2:
                for name in ('exposure','pooled','blend'):
                    gains = [r['methods'][name]['delta_pp'] for r in results]
                    summary[name] = dict(mean_pp=float(np.mean(gains)), passes=bool(
                        np.mean(gains)>=.5 and sum(g>0 for g in gains)>=2 and min(gains)>=-.5))
                winner = max((n for n in summary if summary[n]['passes']),
                             key=lambda n:(summary[n]['mean_pp'], n!='blend'), default=None)
                save(out/'screen.json',dict(summary=summary,winner=winner))
        confirmed = False
        if winner is not None:
            gains = [r['methods'][winner]['correct']-r['methods']['baseline']['correct'] for r in results]
            confirmed = bool(sum(gains[3:]) > 0 and 100*sum(gains)/sum(r['held'] for r in results)>=.5)
        save(out/'completion.json',dict(status='complete',summary=summary,winner=winner,confirmed=confirmed,
            classifier_fits=fits,unsupervised_cluster_fits=unsupervised,seconds=round(time.monotonic()-start,3),
            api_calls=0,course_yuan=0,designated_validation_opened=False,formal_test_opened=False))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),classifier_admissions=fits,
             seconds=round(time.monotonic()-start,3),api_calls=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.task,args.output)
