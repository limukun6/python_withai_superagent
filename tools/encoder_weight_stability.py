"""Test an accuracy-calibrated blend and bound any globally fixed blend on the same components."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from superagent.data import load_training_task
from superagent.minilm_features import feature_usage
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.encoder_weight_headroom import binary_choices
from tools.frontier_study import expect
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


CONTROLS = ('text-encoder', 'encoder-depth', 'encoder-depth-control')


def accuracy_weight(scores, ids, labels, anchor_weight):
    """Equal total weight per row; tied decisions prefer the log-loss optimum."""
    p, q = np.concatenate(scores).T
    rows = np.concatenate(ids)
    _, inverse, frequency = np.unique(rows, return_inverse=True, return_counts=True)
    expect(np.all((frequency >= 1) & (frequency <= 3)), 'Unexpected number of visits to a row')
    votes = 6 // frequency[inverse]
    boundaries, predict = binary_choices(p, q)
    choices = np.unique(np.r_[boundaries, np.longdouble(anchor_weight)])
    return max(choices, key=lambda w: (np.dot(votes, predict(w) == labels[rows]),
                                       -abs(w-anchor_weight), -abs(w-.5), -w))


def inner_choice(folder):
    decision = read(folder/'encoder_mixture.json')
    if not decision['eligible']:
        return dict(eligible=False, weight='1.0')
    history = {t['candidate']: t['trial'] for t in read(folder/'trials.json')}
    full = joblib.load(folder/'refit_data.joblib')
    expect(np.array_equal(np.unique(full['y_train']), [0, 1]), 'This probe expects binary course labels')
    vectors = []
    indices = []
    for split in range(3):
        ids = read(folder/('split.json' if not split else f'confirmation_split_{split}.json'))['search_indices']
        pair = []
        for name in (decision['anchor'], decision['other']):
            number = history[name]
            file = f'trial_{number}.joblib' if not split else f'confirmation_{split}_{number}.joblib'
            artifact = joblib.load(folder/file)
            expect(np.array_equal(artifact['score_classes'], [0,1]), 'Probability columns differ')
            pair.append(artifact['search_scores'][:,1])
        vectors.append(np.column_stack(pair))
        indices.append(ids)
    learned = accuracy_weight(vectors, indices, full['y_train'], decision['raw'])
    return dict(eligible=True, weight=str(learned), log_loss_weight=decision['raw'])


def global_limit(blocks):
    labels = np.concatenate([b['y'] for b in blocks])
    counts_by_fold = np.asarray([len(b['y']) for b in blocks])
    offsets = np.r_[0, np.cumsum(counts_by_fold)[:-1]]
    p, q = (np.concatenate([b[key] for b in blocks]) for key in ('p','q'))
    weights, predict = binary_choices(p, q)
    seeds = sorted({b['seed'] for b in blocks})
    controls = np.asarray([[b['control_counts'][key] for key in CONTROLS] for b in blocks])
    positions = [np.asarray([i for i,b in enumerate(blocks) if b['seed']==s]) for s in seeds]
    required = [int(np.ceil(.002 * counts_by_fold[pos].sum())) for pos in positions]
    n_pass = 0
    best = None
    for w in weights:
        counts = np.add.reduceat((predict(w) == labels).astype(int), offsets)
        differences = counts[:,None] - controls
        margins = [differences[pos].sum(axis=0) for pos in positions]
        wins = [(differences[pos] > 0).sum(axis=0) for pos in positions]
        qualifies = [bool(np.all(delta >= threshold) and np.all(win > len(pos)/2))
                     for delta,win,pos,threshold in zip(margins,wins,positions,required,strict=True)]
        n_pass += int(all(qualifies))
        minimum = min(int(delta.min()) for delta in margins)
        if best is None or minimum > best['worst_margin_rows']:
            best = dict(worst_margin_rows=minimum, fold_wins_against_each_control=[v.tolist() for v in wins],
                        correct_by_seed=[int(counts[pos].sum()) for pos in positions])
    return dict(seeds=seeds, controls=list(CONTROLS), distinct_boundaries_and_interiors=len(weights),
                upper_best_worst_margin=best, any_fixed_weight_qualifies_everywhere=bool(n_pass),
                qualifying_boundaries_and_interiors=n_pass, threshold_rows=required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'archive', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol = dict(seeds=args.seeds, archive_sha256=digest(args.archive), source_sha256=digest(Path(__file__)),
                    boundary_source_sha256=digest(Path(__file__).with_name('encoder_weight_headroom.py')),
                    rule='Each fold takes the weight maximizing accuracy on the three grouped inner checks, weighting each distinct row once. Ties prefer the log-loss weight, then 0.5 and then the smaller weight. Fold decisions are saved before outer labels are read. Separately, an exhaustive hindsight calculation tests every common anchor weight and compares it to the small, strict-depth and fixed-offer policies in each partition. Only within-fold accuracy fitting is an executable policy.')
    freeze_file(destination/'protocol.json', protocol)
    if args.freeze:
        return
    locations = {}
    protocol_roots = {digest(file):file.parent for file in args.root.glob('*/protocol.json')}
    for seed in args.seeds:
        stage = args.root/f'live-mixture-{seed}'
        frozen = read(stage/'protocol.json')
        replay = read((args.root/f'encoder-weight-{seed}')/'protocol.json')
        vote = protocol_roots[replay['vote_sha256']]
        expect(frozen['archive_sha256']==protocol['archive_sha256'] and (stage/'results.json').exists(), 'Missing frozen live result')
        for number,paths in zip(frozen['fold_numbers'],frozen['locations'],strict=True):
            source = Path(paths['encoder-mixture'])
            choice = inner_choice(source)
            freeze_file(destination/f'choice_{seed}_{number}.json', choice)
            locations[seed,number] = stage, source, choice, vote/f'fold_{number-1}'

    opened = []
    with train_only(args.archive, opened):
        task = load_training_task(args.archive)
    blocks = []
    calibrated = {}
    for (seed, number),(stage, source, choice, vote) in locations.items():
        fold = stage/f'fold_{number-1}'
        outer = read(fold/'outer_predictions.json')
        ids = outer['rows']
        y = task.y_train[ids]
        np.testing.assert_array_equal(y, outer['truth'])
        anchor = np.asarray(outer['encoder-depth'])
        p = q = anchor.astype(float)
        if choice['eligible']:
            evidence = read(source/'encoder_mixture.json')
            # Even if the fitted loss optimum is an endpoint, both search models are available.
            selection = read(source/'final_selection.json')
            first = joblib.load(source/selection['single_model_file'])['model']
            second = joblib.load(source/f"trial_{evidence['other_trial']}.joblib")['model']
            x = take(task.X_train, ids)
            scores = []
            for model in (first,second):
                cache = vote/(model.weight_setting+'_prediction_features')
                runtime = read(cache/'pretrained_runtime.json')
                before = feature_usage(cache)
                model.bind_prediction_weights(runtime[model.weight_setting],cache)
                scores.append(model.predict_proba(x)[:,1])
                expect(feature_usage(cache)==before,'Stability probe performed fresh scoring forwards')
            p,q = scores
        _, predictor = binary_choices(p,q)
        np.testing.assert_array_equal(predictor(1),anchor)
        actual = predictor(np.longdouble(choice['weight']))
        control_counts = {k:int(np.sum(np.asarray(outer[k])==y)) for k in CONTROLS}
        blocks.append(dict(seed=seed,y=y,p=p,q=q,control_counts=control_counts))
        record = calibrated.setdefault(str(seed), dict(rows=0, correct=0, folds=[], wins={k:0 for k in CONTROLS}, controls={k:0 for k in CONTROLS}))
        correct = int(np.sum(actual==y))
        record['rows'] += len(y)
        record['correct'] += correct
        record['folds'].append(dict(fold=number, weight=choice['weight'], correct=correct, rows=len(y)))
        for k,v in control_counts.items():
            record['wins'][k] += int(correct>v)
            record['controls'][k] += v
    for record in calibrated.values():
        record['qualified'] = all(record['correct']-v >= .002*record['rows'] and record['wins'][k]>len(record['folds'])/2
                                  for k,v in record['controls'].items())
    report = dict(protocol_sha256=digest(destination/'protocol.json'), inner_accuracy_choice=calibrated,
                  hindsight_global=global_limit(blocks), extra_classifier_fits=0, extra_encoder_forwards=0,
                  new_api_calls=0, opened_training_members=opened)
    write(destination/'results.json',report)
    print({k:(v['correct'],v['qualified']) for k,v in calibrated.items()}, report['hindsight_global'], flush=True)


if __name__=='__main__':
    main()
