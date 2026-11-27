"""Fixed capacity alternatives on paired development folds; no validation member reads."""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.run_task1_profile_screen import digest, load_split, save
from superagent.joint_context import ContextTokenTree
from superagent.splits import sample_groups


CONFIGS = {
    'incumbent': {},
    'deeper': dict(depth=8,iterations=2400,learning_rate=.035,l2_leaf_reg=8,max_ctr_complexity=2),
    'ordered': dict(depth=6,iterations=2400,learning_rate=.035,l2_leaf_reg=8,
                    max_ctr_complexity=2,boosting_type='Ordered'),
}


def run(task,out):
    out.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    x,y,members = load_split(task,'train')
    groups = sample_groups(SimpleNamespace(kind='categorical_tabular',X_train=x))
    splits = list(StratifiedGroupKFold(5,shuffle=True,random_state=18643).split(x,y,groups))[:2]
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__),root/'superagent/joint_context.py',root/'superagent/profile_history.py',
               root/'superagent/field_interactions.py',root/'superagent/models.py',
               root/'docs/batches/TASK1_CONTEXT_CAPACITY_PLAN_20261127_zh.md']
    save(out/'protocol.json',dict(members=members,configs=CONFIGS,
        sources={p.name:digest(p.read_bytes()) for p in sources},
        folds=[dict(train_sha=digest(a.astype('<i8').tobytes()),held_sha=digest(b.astype('<i8').tobytes()))
               for a,b in splits],fit_cap=6,api_calls=0,validation_opened=False,test_opened=False))
    rows = []
    fits = 0
    try:
        for fold,(train,held) in enumerate(splits):
            row = dict(fold=fold,train=len(train),held=len(held))
            reference = None
            for name,params in CONFIGS.items():
                fits += 1
                save(out/'last_attempt.json',dict(fold=fold,name=name,admitted=fits))
                tick = time.monotonic()
                model = ContextTokenTree(dict(pair_limit=3,tree_params=params)).fit(x[train],y[train])
                pred = model.predict(x[held])
                correct = pred==y[held]
                if reference is None:
                    reference = correct
                row[name] = dict(correct=int(correct.sum()),accuracy=float(correct.mean()),
                    delta_pp=float(100*(correct.mean()-reference.mean())),fixed=int((correct&~reference).sum()),
                    broken=int((~correct&reference).sum()),seconds=round(time.monotonic()-tick,3),
                    prediction_sha=digest(np.asarray(pred,dtype='<i8').tobytes()))
                print(fold,name,row[name],flush=True)
                save(out/'results.json',rows+[row])
            rows.append(row)
        summary = {}
        for name in ('deeper','ordered'):
            gains = [r[name]['delta_pp'] for r in rows]
            summary[name] = dict(mean_delta_pp=float(np.mean(gains)),positive=sum(g>0 for g in gains),
                                 promote=bool(min(gains)>0 and np.mean(gains)>=.5))
        promoted = max((n for n in summary if summary[n]['promote']),key=lambda n:summary[n]['mean_delta_pp'],default=None)
        save(out/'completion.json',dict(status='complete',fits=fits,summary=summary,promoted=promoted,
                                        seconds=round(time.monotonic()-start,3),api_calls=0,tokens=0,course_yuan=0))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),admitted_fits=fits,
                                        seconds=round(time.monotonic()-start,3),api_calls=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.task,args.output)
