"""Learn condition-dependent column alignment on fitting rows, with a rotated placebo."""
import argparse
from collections import defaultdict
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_task1_exposure import paired
from scripts.run_task1_profile_screen import digest, load_split, save
from superagent.joint_context import ContextTokenTree, blocks, tokens
from superagent.models import _clean_categories
from superagent.splits import sample_groups


def transfer_gain(column, labels, half):
    _, codes = np.unique(column, return_inverse=True)
    estimates = []
    for fit in (half, ~half):
        if min(fit.sum(), (~fit).sum()) < 20:
            return -1.
        prior = (labels[fit].sum()+1)/(fit.sum()+2)
        count = np.bincount(codes[fit], minlength=codes.max()+1)
        positives = np.bincount(codes[fit], weights=labels[fit], minlength=len(count))
        p = ((positives+12*prior)/(count+12))[codes[~fit]]
        estimates.append(np.mean(np.where(labels[~fit], np.log(p/prior), np.log((1-p)/(1-prior)))))
    return min(estimates)


class AlignedTree(ContextTokenTree):
    def fit(self, rows, labels):
        x = _clean_categories(rows)
        profile, context = blocks(x)
        families = defaultdict(list)
        for j in profile:
            levels = tuple(v for v in np.unique(x[:,j]) if v != '__MISSING__')
            if 3 <= len(levels) <= 24:
                families[levels].append(j)
        people, indices = np.unique(tokens(x[:,profile]), return_inverse=True)
        half = np.random.default_rng(373).permutation(len(people))[indices] < len(people)//2
        _, y = np.unique(labels, return_inverse=True)
        if len(np.unique(y)) != 2:
            raise ValueError('Alignment screen expects binary classification')
        best = -np.inf
        self.mapping = {}
        for cols in families.values():
            if not 3 <= len(cols) <= 12:
                continue
            for axis in context:
                values = np.unique(x[:,axis])
                if len(values) != len(cols):
                    continue
                gains = []
                for value in values:
                    group = x[:,axis] == value
                    gains.append([transfer_gain(x[group,j],y[group],half[group]) for j in cols])
                cost = np.asarray(gains)
                a,b = linear_sum_assignment(-cost)
                support = float(cost[a,b].mean()-cost.mean())
                if support <= best or support <= .001:
                    continue
                best = support
                order = np.roll(b,1) if self.params.get('rotate') else b
                self.axis = axis
                self.family = cols
                self.mapping = {str(values[i]):cols[k] for i,k in zip(a,order,strict=True)}
        if not self.mapping:
            raise ValueError('No supported alignment in fitting rows')
        return super().fit(rows,labels)

    def transform(self, rows):
        x = _clean_categories(rows)
        base = super().transform(x)
        selected = np.asarray([row[self.mapping[row[self.axis]]] if row[self.axis] in self.mapping else '__UNKNOWN__'
                               for row in x])
        count = (x[:,self.family] == selected[:,None]).sum(axis=1).astype(str)
        interactions = [tokens(np.column_stack((selected,x[:,j]))) for j in self.context]
        return np.column_stack((base,selected,count,*interactions))


def run(task, out):
    out.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    x,y,members = load_split(task,'train')
    groups = sample_groups(SimpleNamespace(X_train=x,kind='categorical_tabular'))
    folds = list(StratifiedGroupKFold(5,shuffle=True,random_state=112801).split(x,y,groups))
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), root/'scripts/run_task1_exposure.py',root/'superagent/joint_context.py',
                    root/'superagent/profile_history.py',root/'docs/batches/TASK1_ALIGNED_FIELDS_PLAN_20261128_zh.md']
    previous = root/'research/task1_exposure_01'
    control = json.loads((previous/'results.json').read_text(encoding='utf-8'))
    control_protocol = json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
    save(out/'protocol.json',dict(members=members,seed=112801,control_sha=digest((previous/'results.json').read_bytes()),
         sources={p.name:digest(p.read_bytes()) for p in source_files},screen_cap=9,confirmation_cap=4))
    rows = []
    fits = 0
    promote = False
    try:
        for fold,(train,held) in enumerate(folds):
            if fold>=3 and not promote:
                break
            assert digest(held.astype('<i8').tobytes())==control_protocol['splits'][fold]['held_sha']
            row = dict(fold=fold,held=len(held),methods={})
            for name in (('baseline','aligned','rotated') if fold<3 else ('baseline','aligned')):
                fits += 1
                save(out/'last_attempt.json',dict(fold=fold,name=name,admissions=fits))
                tick = time.monotonic()
                model = (ContextTokenTree(dict(pair_limit=3)) if name=='baseline'
                         else AlignedTree(dict(pair_limit=3,rotate=name=='rotated')))
                model.fit(x[train],y[train])
                p = model.predict_proba(x[held])[:,1]
                if name=='baseline':
                    ref = p
                    uncertain = np.abs(ref-.5)<=.15
                stats = paired(p,ref,y[held],uncertain)
                if name=='baseline' and fold < len(control):
                    assert stats['prediction_sha']==control[fold]['methods']['baseline']['prediction_sha']
                if name!='baseline':
                    stats.update(mapping=model.mapping, axis=int(model.axis))
                stats['seconds']=round(time.monotonic()-tick,3)
                row['methods'][name] = stats
                save(out/'results.json',rows+[row])
                print(fold,name,stats['correct'],flush=True)
            rows.append(row)
            if fold==2:
                delta = [r['methods']['aligned']['delta_pp'] for r in rows]
                vs_rotated = sum(r['methods']['aligned']['correct']-r['methods']['rotated']['correct'] for r in rows)
                promote = bool(np.mean(delta)>=.5 and min(delta)>=-.5 and sum(d>0 for d in delta)>=2 and vs_rotated>0)
                save(out/'screen.json',dict(mean_delta_pp=float(np.mean(delta)),net_vs_rotated=vs_rotated,promoted=promote))
        net = [r['methods']['aligned']['correct']-r['methods']['baseline']['correct'] for r in rows]
        confirmed = bool(promote and sum(net[3:])>0 and 100*sum(net)/sum(r['held'] for r in rows)>=.5)
        save(out/'completion.json',dict(status='complete',screen_promoted=promote,confirmed=confirmed,
            classifier_fits=fits,seconds=round(time.monotonic()-start,3),api_calls=0,course_yuan=0,
            designated_validation_opened=False,formal_test_opened=False))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),classifier_admissions=fits,
                                        seconds=round(time.monotonic()-start,3),api_calls=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.task,args.output)
