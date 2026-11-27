"""Compare categorical profile/context models on disjoint training-internal folds."""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_task1_profile_screen import digest, load_split, save
from superagent.ab_methods import _field_tree
from superagent.joint_context import ContextTokenTree, JointKernel
from superagent.splits import sample_groups


def run(task, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    x, y, hashes = load_split(task,'train')
    groups = sample_groups(SimpleNamespace(kind='categorical_tabular',X_train=x))
    splits = list(StratifiedGroupKFold(5,shuffle=True,random_state=18643).split(x,y,groups))[:3]
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/'scripts/run_task1_profile_screen.py', root/'superagent/profile_history.py',
               root/'superagent/joint_context.py', root/'superagent/field_interactions.py',root/'superagent/models.py',
               root/'superagent/ab_methods.py', root/'superagent/splits.py',
               root/'docs/batches/TASK1_JOINT_CONTEXT_PLAN_20261127_zh.md']
    save(out/'protocol.json',dict(members=hashes,sources={p.name:digest(p.read_bytes()) for p in sources},
        splits=[dict(train_sha=digest(tr.astype('<i8').tobytes()),held_sha=digest(h.astype('<i8').tobytes()))
                for tr,h in splits],seed=18643,cap=9,api_calls=0,test_opened=False,validation_opened=False))
    results = []
    fits = 0
    try:
        for fold,(train,held) in enumerate(splits):
            record = dict(fold=fold,train=len(train),held=len(held))
            probability = {}
            for name in ('baseline','tokens','kernel','blend'):
                since = time.monotonic()
                if name != 'blend':
                    fits += 1
                    save(out/'last_attempt.json',dict(fold=fold,method=name,admitted=fits))
                if name == 'baseline':
                    model,_ = _field_tree(x[train],y[train],x[held])
                elif name == 'tokens':
                    model = ContextTokenTree(dict(pair_limit=3)).fit(x[train],y[train])
                elif name == 'kernel':
                    model = JointKernel().fit(x[train],y[train])
                if name == 'blend':
                    p = (probability['baseline']+probability['kernel'])/2
                else:
                    if list(model.classes_) != [0,1]:
                        raise ValueError('Expected probabilities in binary label order')
                    p = model.predict_proba(x[held])[:,1]
                if not np.isfinite(p).all():
                    raise ValueError('Nonfinite held scores')
                probability[name] = p
                correct = (p > .5) == y[held]
                base_correct = (probability['baseline'] > .5) == y[held]
                record[name] = dict(correct=int(correct.sum()),accuracy=float(correct.mean()),
                    fixed=int((correct & ~base_correct).sum()),broken=int((~correct & base_correct).sum()),
                    delta_pp=float(100*(correct.mean()-base_correct.mean())),seconds=round(time.monotonic()-since,3),
                    prediction_sha=digest((p > .5).astype('<i8').tobytes()))
                print(fold,name,record[name]['accuracy'],flush=True)
                save(out/'results.json',results+[record])
            results.append(record)
        summary = {}
        for name in ('tokens','kernel','blend'):
            delta = [r[name]['delta_pp'] for r in results]
            summary[name] = dict(mean_delta_pp=float(np.mean(delta)),positive=sum(d>0 for d in delta),
                                 correct=sum(r[name]['correct'] for r in results))
        eligible = [n for n in ('tokens','kernel') if summary[n]['mean_delta_pp']>=1 and summary[n]['positive']>=2]
        promoted = max(eligible,key=lambda n:summary[n]['mean_delta_pp'],default=None)
        save(out/'completion.json',dict(status='complete',fits=fits,api_calls=0,tokens=0,course_yuan=0,
            seconds=round(time.monotonic()-start,3),summary=summary,promoted=promoted))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),admitted_fits=fits,api_calls=0,
                                        seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    run(args.task,args.output)
