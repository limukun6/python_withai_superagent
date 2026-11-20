"""Separate the effects of training sample count, fixed-model capacity and identical-feature conflicts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from superagent.data import load_training_task
from superagent.models import search_candidates,predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import sample_groups,take
from tools.frontier_study import expect
from tools.pretrained_study import read,write,train_only


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    ref=read(args.reference/'protocol.json')
    expect(digest(args.archive)==ref['archive_sha256'],'Archive differs')
    opened=[]
    with train_only(args.archive,opened):whole=load_training_task(args.archive)
    expect(whole.kind=='categorical_tabular','This diagnostic is for a categorical baseline')
    groups=sample_groups(whole)
    indices=[]
    seen=set()
    for split in ref['indices']:
        train=np.asarray(split['train'])
        held=np.asarray(split['outer'])
        expect(not seen.intersection(held),'Outer rows repeat')
        seen.update(held)
        partitions=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=47193).split(
            np.zeros(len(train)),whole.y_train[train],groups[train])
        blocks=[bucket for _,bucket in partitions]
        parts=[train[np.sort(np.concatenate(blocks[:n]))].tolist() for n in range(1,5)]
        expect(parts[-1]==split['train'] and set(groups[train]).isdisjoint(groups[held]),'Nested groups differ')
        indices.append(dict(parts=parts,outer=split['outer']))
    repo=Path(__file__).resolve().parents[1]
    source=[Path(__file__),*sorted((repo/'superagent').glob('*.py')),repo/'tools/frontier_study.py',repo/'tools/pretrained_study.py']
    name='catboost_depth6_reg3.0'
    plan=dict(reference_sha256=digest(args.reference/'protocol.json'),archive_sha256=ref['archive_sha256'],indices=indices,
              source={str(p.relative_to(repo)):digest(p) for p in source},candidate=name,inner_group_seed=47193,
              objective='Fit the same classifier on nested 25/50/75/100% groups before any outer scoring. The collision bound is restricted to the observed rows and exact input representation.')
    if (root/'protocol.json').exists():expect(read(root/'protocol.json')==plan,'Frozen learning curve changed')
    else:write(root/'protocol.json',plan)
    if args.freeze:return
    candidate=next(c for c in search_candidates(whole.kind,'expanded') if c.family==name)
    for i,fold in enumerate(indices):
        folder=root/f'fold_{i}';folder.mkdir(exist_ok=True)
        for k,part in enumerate(fold['parts'],1):
            model_file=folder/f'model_{k}.joblib'; receipt=folder/f'model_{k}.json'
            if receipt.exists():
                expect(digest(model_file)==read(receipt)['sha256'],'Model changed');continue
            x=take(whole.X_train,part)
            model,_=candidate.fit_predict(x,whole.y_train[part],x[:1])
            joblib.dump(model,model_file)
            write(receipt,dict(sha256=digest(model_file),rows=len(part)))
        print('Frozen learning curve',i+1,flush=True)
    folds=[]
    for i,fold in enumerate(indices):
        held=fold['outer'];folder=root/f'fold_{i}'
        truth=whole.y_train[held]
        preds=[predict_with_model(whole.kind,joblib.load(folder/f'model_{k}.joblib'),take(whole.X_train,held)) for k in range(1,5)]
        write(folder/'outer_predictions.json',dict(rows=held,truth=truth.tolist(),predictions=[p.tolist() for p in preds]))
        folds.append(dict(rows=len(held),train_counts=list(map(len,fold['parts'])),correct=[int(np.sum(p==truth)) for p in preds]))
    labels=defaultdict(Counter)
    for i in seen:labels[groups[i]][whole.y_train[i]]+=1
    conflict=sum(sum(c.values())-max(c.values()) for c in labels.values())
    report=dict(folds=folds,rows=len(seen),correct=np.sum([f['correct'] for f in folds],axis=0).tolist(),
                exact_input_groups=len(labels),conflicting_groups=sum(len(c)>1 for c in labels.values()),
                observed_exact_input_upper_correct=len(seen)-conflict,unavoidable_conflicting_rows=conflict,
                diagnostic_fits=4*len(folds),api_calls=0,course_rmb=0.,training_members=opened)
    write(root/'results.json',report)
    print({k:v for k,v in report.items() if k!='folds'},flush=True)


if __name__=='__main__':main()
