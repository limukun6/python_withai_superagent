"""Replay a wider depth offer with the live blend rule, reusing matching trajectories."""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np

from superagent.data import TrainingTask, load_training_task
from superagent.encoder_mixture import fit_weight
from superagent.method_choice import TEXT_DEEP, TEXT_MINILM
from superagent.minilm_features import feature_usage
from superagent.pretrained_choice import digest
from superagent.splits import sample_groups, take
from tools.encoder_pair_vote import bind_scoring_model, inner_decision
from tools.frontier_study import expect
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


def replay_changed(source, folder, weights):
    current = read(source/'confirmation.json')
    winner = current['selected_candidate']
    recovery = folder/'reconstructed'
    choice = inner_decision(source, winner, recovery, weights)
    result = dict(eligible=choice['eligible'], weight=1., anchor=winner, extra_fits=choice['reconstructed_fits'])
    if not choice['eligible']:
        return result
    other = next(iter({TEXT_MINILM,TEXT_DEEP}-{winner}))
    numbers = {t['candidate']:t['trial'] for t in read(source/'trials.json')}
    full = joblib.load(source/'refit_data.joblib')
    groups = sample_groups(TrainingTask(full['kind'],full['X_train'],full['y_train']), 'shared_text')
    ids = []
    probabilities = []
    before = feature_usage(source)
    for index in range(3):
        data = joblib.load(source/('inner_data.joblib' if index==0 else f'confirmation_data_{index}.joblib'))
        rows = read(source/('split.json' if index==0 else f'confirmation_split_{index}.json'))['search_indices']
        ids.append(rows)
        np.testing.assert_array_equal(full['y_train'][rows], data['y_search'])
        pair = []
        for name in (winner,other):
            number = numbers[name]
            file = source/f'trial_{number}.joblib' if index==0 else recovery/f'confirmed_{index}_{number}.joblib'
            artifact = joblib.load(file)
            model = artifact['model']
            np.testing.assert_array_equal(model.classes_,[0,1])
            scores = model.predict_proba(data['X_search'])
            np.testing.assert_array_equal(scores.argmax(axis=1), artifact['search_predictions'])
            pair.append(scores[:,1])
        probabilities.append(np.column_stack(pair))
    expect(feature_usage(source)==before and feature_usage(recovery)['input_tokens']==0,'Unexpected training encoder work')
    calibration = fit_weight(probabilities,ids,full['y_train'],groups)
    return dict(result,weight=calibration['raw'],other_trial=numbers[other],inner_distinct_rows=calibration['unique_rows'])


def shared_trajectory(source, old):
    current, earlier = (read(d/'trials.json') for d in (source,old))
    if [t['candidate'] for t in current] != [t['candidate'] for t in earlier]:
        return False
    left, right = (read(d/'confirmation.json') for d in (source,old))
    expect(left['selected_candidate']==right['selected_candidate'] and left['shortlist']==right['shortlist'],
           'A matching search did not yield matching finalists')
    indices = {row['candidate']:row['trial'] for row in current}
    for j in range(3):
        split = 'split.json' if j==0 else f'confirmation_split_{j}.json'
        expect(read(source/split)['search_indices']==read(old/split)['search_indices'],'Confirmation rows changed')
        for n in (range(1,8) if j==0 else [indices[k] for k in left['shortlist']]):
            name = f'trial_{n}.joblib' if j==0 else f'confirmation_{j}_{n}.joblib'
            a,b = (joblib.load(d/name)['search_predictions'] for d in (source,old))
            np.testing.assert_array_equal(a,b)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('reference','live','depth-gate','archive','output','small-weights','deep-weights'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--freeze',action='store_true')
    args = parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE']='1'
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    ref=read(args.reference/'protocol.json')
    live=read(args.live/'protocol.json')
    expect(ref['indices']==live['indices'] and ref['archive_sha256']==digest(args.archive),'Different study populations')
    weights = dict(minilm_weights=args.small_weights,deep_minilm_weights=args.deep_weights)
    freeze_file(root/'protocol.json',dict(reference_sha256=digest(args.reference/'protocol.json'),
        live_sha256=digest(args.live/'protocol.json'),gate_protocol_sha256=digest(args.depth_gate/'protocol.json'),
        source_sha256=digest(Path(__file__)),rule='Follow the pre-existing relaxed training gate. Where it selects deeper, use the actual fixed-offer arm, otherwise use the strict comparator. Apply the same training-only log-loss mixture. Reuse equal trajectories after checking every search and every shortlisted confirmation prediction. Reconstruct only discarded component confirmations from divergent trajectories.'))
    if args.freeze:
        return
    expect((args.depth_gate/'results.json').exists() and (args.live/'results.json').exists(),'Waiting for complete comparison runs')
    decisions = []
    for i,(block,live_block) in enumerate(zip(ref['locations'],live['locations'],strict=True)):
        folder = root/f'fold_{i}'
        folder.mkdir(exist_ok=True)
        # The relaxed gate was frozen in the separate depth-gate experiment using only training observations.
        gate_choice=read(args.depth_gate/f'fold_{i}'/'decision.json')
        offer=gate_choice['use_deep']
        source = Path(block[gate_choice['observed_run']])
        physical = Path(live_block['encoder-mixture'])
        equal = shared_trajectory(source,physical)
        if equal:
            fitted = read(physical/'encoder_mixture.json')
            choice = dict(eligible=fitted['eligible'],weight=fitted.get('raw',1.),extra_fits=0)
        else:
            choice = replay_changed(source,folder,weights)
        choice.update(same_trajectory=equal,relaxed_offered_depth=offer,observed_run=gate_choice['observed_run'])
        freeze_file(folder/'decision.json',choice)
        decisions.append((source,choice))
    opened = []
    with train_only(args.archive,opened):
        task = load_training_task(args.archive)
    folds=[]
    for i,(source,choice) in enumerate(decisions):
        folder = root/f'fold_{i}'
        ids=ref['indices'][i]['outer']
        physical=read(args.live/f'fold_{i}'/'outer_predictions.json')
        old=read(args.depth_gate/f'fold_{i}'/'outer_predictions.json')
        y=task.y_train[ids]
        expect(ids==physical['rows']==old['rows'],'Outer ordering differs')
        chosen_arm=choice['observed_run']
        np.testing.assert_array_equal(physical[chosen_arm],old['relaxed'])
        combined=np.asarray(physical[chosen_arm])
        if choice['same_trajectory']:
            np.testing.assert_array_equal(physical[chosen_arm],physical['encoder-depth'])
            combined=np.asarray(physical['encoder-mixture'])
        elif choice['eligible'] and choice['weight']<1:
            x=take(task.X_train,ids)
            first=joblib.load(source/'selected_model.joblib')['model']
            second=joblib.load(source/f"trial_{choice['other_trial']}.joblib")['model']
            donors=[Path(path).parent/(arm+'_prediction_features') for arm,path in ref['locations'][i].items()]
            for model in (first,second):
                bind_scoring_model(model,folder,weights,donors)
            left=first.predict_proba(x)
            right=second.predict_proba(x)
            np.testing.assert_array_equal(left.argmax(axis=1),physical[chosen_arm])
            combined=(choice['weight']*left+(1-choice['weight'])*right).argmax(axis=1)
        arrays={k:np.asarray(physical[k]) for k in ('text-encoder','encoder-depth','encoder-depth-control','encoder-mixture')}
        arrays['relaxed_mix']=combined
        counts={k:int(np.sum(v==y)) for k,v in arrays.items()}
        write(folder/'outer_predictions.json',dict(rows=ids,truth=y.tolist(),**{k:v.tolist() for k,v in arrays.items()}))
        folds.append(dict(rows=len(y),correct=counts,same_trajectory=choice['same_trajectory'],weight=choice['weight']))
    total={k:sum(f['correct'][k] for f in folds) for k in folds[0]['correct']}
    rows=sum(f['rows'] for f in folds)
    wins={k:sum(f['correct']['relaxed_mix']>f['correct'][k] for f in folds) for k in total if k!='relaxed_mix'}
    checks={k:total['relaxed_mix']-total[k]>=.002*rows and w>len(folds)/2 for k,w in wins.items()}
    report=dict(rows=rows,correct=total,folds=folds,wins=wins,meets_each_comparison=checks,
                qualified=all(checks.values()),additional_diagnostic_fits=sum(c['extra_fits'] for _,c in decisions),
                reused_live_predictions=sum(c['same_trajectory'] for _,c in decisions),
                additional_feature_tokens=sum(feature_usage(d)['input_tokens'] for d in root.glob('fold_*/*_prediction_features')),
                new_api_calls=0,training_members=opened)
    write(root/'results.json',report)
    print(total,wins,checks,flush=True)


if __name__=='__main__':
    main()
