"""Check whether two already trained encoder classifiers justify predicting together."""
import argparse
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.metrics import log_loss

from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.method_choice import TEXT_DEEP, TEXT_MINILM
from superagent.minilm_features import DEEP_FILES, FILES, feature_usage
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


def confirmation_probabilities_model(source, destination, payload, index, number, name, weight_dirs, saved):
    """Restore an omitted confirmation model and check all its original predictions."""
    destination.mkdir(exist_ok=True)
    for file in [source / f'method_{name}.json', *source.glob('minilm*vector*.sqlite')]:
        if not (destination / file.name).exists():
            shutil.copyfile(file, destination / file.name)
    freeze_file(destination / 'pretrained_runtime.json', {k:str(v.resolve()) for k,v in weight_dirs.items()})
    data_path = destination / f'inner_{index}.joblib'
    if not data_path.exists():
        settings = dict(payload, retain_model=True)
        settings.pop('feature_cache_dir', None)
        joblib.dump(settings, data_path)
    path = destination / f'confirmed_{index}_{number}.joblib'
    proof = path.with_suffix('.json')
    if proof.exists():
        expect(digest(path) == read(proof)['sha256'], 'Rebuilt model changed')
    else:
        model, metrics = train_candidate(data_path, name, path, 480)
        expect(model is not None and metrics['classifier_fits'] == 1, str(metrics))
        freeze_file(proof, dict(sha256=digest(path), fits=1))
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored['search_predictions'], saved['search_predictions'])
    expect(feature_usage(destination)['input_tokens'] == 0, 'Unexpected extra feature forward during reconstruction')
    return restored['model']


def inner_decision(source, winner, destination, weight_dirs):
    confirmation = read(source / 'confirmation.json')
    both = {TEXT_DEEP, TEXT_MINILM}
    result = dict(winner=winner, eligible=winner in both and both <= set(confirmation['shortlist']), adopted=False, splits=[])
    result['reconstructed_fits'] = 0
    if not result['eligible']:
        return result
    other = next(iter(both - {winner}))
    numbers = {t['candidate']: t['trial'] for t in read(source / 'trials.json')}
    before = feature_usage(source)
    for index in range(3):
        data = source / ('inner_data.joblib' if index == 0 else f'confirmation_data_{index}.joblib')
        payload = joblib.load(data)
        labels = payload['y_search']
        probabilities = []
        hashes = {}
        for name in (winner, other):
            number = numbers[name]
            trial = source / (f'trial_{number}.joblib' if index == 0 else f'confirmation_{index}_{number}.joblib')
            trained = joblib.load(trial)
            model = trained['model']
            if model is None:
                model = confirmation_probabilities_model(source, destination, payload, index, number, name, weight_dirs, trained)
                result['reconstructed_fits'] += 1
            classes = np.unique(payload['y_train'])
            np.testing.assert_array_equal(model.classes_, classes)
            prob = model.predict_proba(payload['X_search'])
            np.testing.assert_array_equal(classes[prob.argmax(axis=1)], trained['search_predictions'])
            probabilities.append(prob)
            hashes[name] = digest(trial)
        mix = (probabilities[0] + probabilities[1]) / 2
        diff = (classes[mix.argmax(axis=1)] == labels).astype(int) - (classes[probabilities[0].argmax(axis=1)] == labels)
        result['splits'].append(dict(rows=len(labels), net_correct=int(diff.sum()), accuracy_gain=float(diff.mean()),
            loss_gain=float(log_loss(labels, probabilities[0], labels=classes) - log_loss(labels, mix, labels=classes)),
            training_artifacts=hashes, search_payload=digest(data)))
    expect(before == feature_usage(source), 'Additional training-side encoding')
    scores = result['splits']
    result['adopted'] = bool(sum(s['net_correct'] > 0 for s in scores) >= 2
        and np.mean([s['accuracy_gain'] for s in scores]) >= .002 and np.mean([s['loss_gain'] for s in scores]) >= .002)
    result['other_trial'] = numbers[other]
    return result


def bind_scoring_model(model, fold_dir, weight_dirs, donors):
    folder = fold_dir / (model.weight_setting + '_prediction_features')
    folder.mkdir(exist_ok=True)
    filename = model.encoder_class.cache_name
    if not (folder / filename).exists():
        for donor in donors:
            if (donor / filename).is_file():
                shutil.copyfile(donor / filename, folder / filename)
                break
    model.bind_prediction_weights(weight_dirs[model.weight_setting], folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('archive', 'reference', 'output', 'minilm-weights', 'deep-minilm-weights'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--arm', default='encoder-depth')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE'] = '1'
    os.environ.pop('SUPERAGENT_PRETRAINED_RUNTIME', None)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = read(args.reference / 'protocol.json')
    expect(protocol['archive_sha256'] == digest(args.archive), 'Archive changed')
    weight_dirs = dict(minilm_weights=args.minilm_weights, deep_minilm_weights=args.deep_minilm_weights)
    for name, units in (('minilm_weights', FILES), ('deep_minilm_weights', DEEP_FILES)):
        expect(all(digest(weight_dirs[name] / k) == v[1] for k, v in units.items()), 'Weights changed')
    freeze_file(root / 'protocol.json', dict(reference_sha256=digest(args.reference / 'protocol.json'),
        arm=args.arm, source_sha256=digest(Path(__file__)),
        rule='Use an equal probability average only when both depth variants were confirmed and one won. '
             'To adopt, the mean loss improvement and mean accuracy improvement must each reach 0.002, '
             'with strict accuracy wins on at least two of three inner splits. The existing full winner '
             'and the other primary-search model supply predictions. Test always mixing as a matched control. '
             'Recreate discarded confirmation models where necessary and require exact original predictions; log these extra fits.'))
    if args.freeze:
        return
    expect((args.reference / 'results.json').exists(), 'Reference has not finished')
    decisions = []
    for i, block in enumerate(protocol['locations']):
        source = Path(block[args.arm])
        rec = receipt(source)
        folder = root / f'fold_{i}'
        folder.mkdir(exist_ok=True)
        decision = inner_decision(source, rec['winner'], folder / 'reconstructed', weight_dirs)
        decision['final_sha256'] = rec['model_sha256']
        freeze_file(folder / 'decision.json', decision)
        decisions.append(decision)
    opened = []
    with train_only(args.archive, opened):
        task = load_training_task(args.archive)
    folds = []
    for i, (block, decision) in enumerate(zip(protocol['locations'], decisions, strict=True)):
        source = Path(block[args.arm])
        folder = root / f'fold_{i}'
        ids = protocol['indices'][i]['outer']
        observed = read(args.reference / f"fold_{protocol['fold_numbers'][i]-1}" / 'outer_predictions.json')
        expect(ids == observed['rows'], 'Outer row order changed')
        anchor = np.asarray(observed[args.arm])
        mix = anchor
        truth = task.y_train[ids]
        if decision['eligible']:
            x = take(task.X_train, ids)
            donors = [Path(v).parent / (k + '_prediction_features') for k, v in block.items()]
            first = joblib.load(source / 'selected_model.joblib')['model']
            second = joblib.load(source / f"trial_{decision['other_trial']}.joblib")['model']
            for model in (first, second):
                bind_scoring_model(model, folder, weight_dirs, donors)
            np.testing.assert_array_equal(first.classes_, second.classes_)
            scores = first.predict_proba(x)
            np.testing.assert_array_equal(first.classes_[scores.argmax(axis=1)], anchor)
            mix = first.classes_[(scores + second.predict_proba(x)).argmax(axis=1)]
        vectors = dict(incumbent=anchor, always_mix=mix, gated_mix=mix if decision['adopted'] else anchor)
        write(folder / 'outer_predictions.json', dict(rows=ids, truth=truth.tolist(), **{k:v.tolist() for k,v in vectors.items()}))
        folds.append(dict(rows=len(ids), eligible=decision['eligible'], adopted=decision['adopted'],
            correct={k:int((v==truth).sum()) for k,v in vectors.items()}))
    total = {k:sum(f['correct'][k] for f in folds) for k in folds[0]['correct']}
    report = dict(rows=sum(f['rows'] for f in folds), folds=folds, correct=total,
        additional_diagnostic_fits=sum(d['reconstructed_fits'] for d in decisions),
        new_api_calls=0, eligible_folds=sum(f['eligible'] for f in folds), adopted_folds=sum(f['adopted'] for f in folds),
        additional_feature_tokens=sum(feature_usage(d)['input_tokens'] for d in root.glob('fold_*/*_prediction_features')),
        training_members=opened)
    write(root / 'results.json', report)
    print(total, report['eligible_folds'], report['adopted_folds'], report['additional_feature_tokens'])


if __name__ == '__main__':
    main()
