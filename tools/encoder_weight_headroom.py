"""Fit one encoder mixture weight inside each fold and bound the gain from any such weight."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import brentq

from superagent.data import TrainingTask, load_training_task
from superagent.method_choice import TEXT_DEEP, TEXT_MINILM
from superagent.minilm_features import feature_usage
from superagent.pretrained_choice import digest
from superagent.splits import sample_groups, take
from tools.frontier_study import expect
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


def fit_weight(inner, ids, truth, groups):
    """Overlapping inner splits give equal total weight to each distinct row."""
    p, q = np.concatenate(inner).T
    ids = np.concatenate(ids)
    rows, inverse, visits = np.unique(ids,return_inverse=True,return_counts=True)
    y = truth[ids]
    p_true, q_true = np.where(y==1,p,1-p), np.where(y==1,q,1-q)
    d = p_true-q_true
    weights = 1/(len(rows)*visits[inverse])
    def slope(w):
        return -float(np.dot(weights,d/np.maximum(q_true+w*d,1e-15)))
    if slope(1)<=0:
        raw=1.
    elif slope(0)>=0:
        raw=0.
    else:
        raw=float(brentq(slope,0.,1.))
    gain=(np.log(np.maximum(q_true+raw*d,1e-15))-np.log(np.maximum(p_true,1e-15)))
    per_row=np.bincount(inverse,weights=gain)/visits
    mean=float(per_row.mean())
    _, cluster=np.unique(groups[rows],return_inverse=True)
    sizes=np.bincount(cluster)
    sums=np.bincount(cluster,weights=per_row)-mean*sizes
    se=float(np.sqrt(len(sizes)/(len(sizes)-1)*np.sum(sums**2))/len(rows)) if len(sizes)>1 else None
    acc=float(np.dot(weights,(((raw*p+(1-raw)*q)>.5)==y).astype(int)-((p>.5)==y)))
    supported=se is not None and mean>=max(.002,se) and acc>=-.002
    return dict(raw=raw,guarded=raw if supported else 1.,unique_rows=len(rows),visits=len(ids),groups=len(sizes),
                loss_gain=mean,group_standard_error=se,accuracy_gain=acc)


def binary_choices(p,q):
    """Each row changes class at most once over the interval of mixture weights."""
    p,q=np.asarray(p,dtype=np.longdouble),np.asarray(q,dtype=np.longdouble)
    direction=p-q
    crossings=np.divide(.5-q,direction,out=np.zeros_like(direction),where=direction!=0)
    values=np.unique(np.r_[0.,1.,crossings[(direction!=0)&(crossings>=0)&(crossings<=1)]])
    candidates=np.sort(np.r_[values,(values[1:]+values[:-1])/2])
    def predict(w):
        return np.where(direction>0,w>crossings,np.where(direction<0,w<crossings,q>.5)).astype(int)
    return candidates,predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('archive','reference','vote','output'):
        parser.add_argument('--'+flag,required=True,type=Path)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    reference=read(args.reference/'protocol.json')
    vote_protocol=read(args.vote/'protocol.json')
    expect(reference['archive_sha256']==digest(args.archive) and vote_protocol['reference_sha256']==digest(args.reference/'protocol.json'),
           'Different study inputs')
    freeze_file(root/'protocol.json',dict(reference_sha256=digest(args.reference/'protocol.json'),vote_sha256=digest(args.vote/'protocol.json'),
        source_sha256=digest(Path(__file__)),rule='Use the same two actually fitted component models as the matched vote. '
        'Minimize inner binary log loss with each unique row equally weighted across the three repeated group splits. '
        'Also test a guarded choice requiring a loss gain of at least 0.002 and one grouped standard error, '
        'while losing at most 0.002 accuracy. Freeze every weight before outer evaluation. Compare to all class-change '
        'breakpoints and intervening intervals for the best whole-fold constant weight known only after scoring.'))
    if args.freeze:return
    expect((args.vote/'results.json').exists(),'Paired encoder replay has not finished')
    opened=[]
    with train_only(args.archive,opened):task=load_training_task(args.archive)
    for i,block in enumerate(reference['locations']):
        source=Path(block['encoder-depth'])
        vote=args.vote/f'fold_{i}'
        record=read(vote/'decision.json')
        folder=root/f'fold_{i}'
        folder.mkdir(exist_ok=True)
        choice=dict(raw=1.,guarded=1.,eligible=record['eligible'])
        if record['eligible']:
            full=joblib.load(source/'refit_data.joblib')
            groups=sample_groups(TrainingTask(full['kind'],full['X_train'],full['y_train']),'shared_text')
            numbers={r['candidate']:r['trial'] for r in read(source/'trials.json')}
            names=[record['winner'],next(iter({TEXT_DEEP,TEXT_MINILM}-{record['winner']}))]
            blocks=[]
            indices=[]
            before=feature_usage(source)
            for j in range(3):
                payload=joblib.load(source/('inner_data.joblib' if j==0 else f'confirmation_data_{j}.joblib'))
                ids=np.asarray(read(source/('split.json' if j==0 else f'confirmation_split_{j}.json'))['search_indices'])
                np.testing.assert_array_equal(full['y_train'][ids],payload['y_search'])
                probabilities=[]
                for name in names:
                    n=numbers[name]
                    path=source/f'trial_{n}.joblib' if j==0 else vote/'reconstructed'/f'confirmed_{j}_{n}.joblib'
                    expected=record['splits'][j]['training_artifacts'][name] if j==0 else read(path.with_suffix('.json'))['sha256']
                    expect(digest(path)==expected,'Training classifier changed')
                    model=joblib.load(path)['model']
                    np.testing.assert_array_equal(model.classes_,[0,1])
                    probabilities.append(model.predict_proba(payload['X_search'])[:,1])
                blocks.append(np.column_stack(probabilities))
                indices.append(ids)
            expect(feature_usage(source)==before and feature_usage(vote/'reconstructed')['input_tokens']==0,'Unexpected extra encoding')
            choice.update(fit_weight(blocks,indices,full['y_train'],groups))
        freeze_file(folder/'decision.json',choice)
    folds=[]
    for i,block in enumerate(reference['locations']):
        source=Path(block['encoder-depth'])
        vote=args.vote/f'fold_{i}'
        record=read(vote/'decision.json')
        folder=root/f'fold_{i}'
        choice=read(folder/'decision.json')
        ids=reference['indices'][i]['outer']
        y=task.y_train[ids]
        existing=read(vote/'outer_predictions.json')
        expect(existing['rows']==ids,'Held-out order changed')
        vectors={k:np.asarray(existing['incumbent']) for k in ('incumbent','equal','learned','guarded','oracle_weight')}
        if record['eligible']:
            x=take(task.X_train,ids)
            paths=[source/'selected_model.joblib',source/f"trial_{record['other_trial']}.joblib"]
            probabilities=[]
            for path in paths:
                model=joblib.load(path)['model']
                cache=vote/(model.weight_setting+'_prediction_features')
                model.bind_prediction_weights(read(cache/'pretrained_runtime.json')[model.weight_setting],cache)
                before=feature_usage(cache)
                probabilities.append(model.predict_proba(x)[:,1])
                expect(feature_usage(cache)==before,'New scoring feature calculation')
            candidates,predict=binary_choices(*probabilities)
            best=max(candidates,key=lambda w:int((predict(w)==y).sum()))
            vectors={k:predict(w) for k,w in dict(incumbent=1.,equal=.5,learned=choice['raw'],guarded=choice['guarded'],oracle_weight=best).items()}
        np.testing.assert_array_equal(vectors['incumbent'],existing['incumbent'])
        np.testing.assert_array_equal(vectors['equal'],existing['always_mix'])
        write(folder/'outer_predictions.json',dict(rows=ids,truth=y.tolist(),**{k:a.tolist() for k,a in vectors.items()}))
        folds.append(dict(rows=len(y),correct={k:int((a==y).sum()) for k,a in vectors.items()},learned_weight=choice['raw'],guarded_weight=choice['guarded']))
    total={k:sum(r['correct'][k] for r in folds) for k in folds[0]['correct']}
    write(root/'results.json',dict(folds=folds,rows=sum(r['rows'] for r in folds),correct=total,
        scalar_calibrations=sum(read(root/f'fold_{i}'/'decision.json')['eligible'] for i in range(len(folds))),
        additional_diagnostic_fits=0,additional_encoder_forwards=0,new_api_calls=0,training_members=opened))
    print(total,flush=True)


if __name__=='__main__':main()
