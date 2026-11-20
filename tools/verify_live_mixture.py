"""Check the actual two-encoder policy against its frozen training-only replay."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from superagent.encoder_mixture import EncoderMixture, MIXTURE, learn_from_confirmations
from superagent.minilm_features import feature_usage
from superagent.pretrained_choice import digest
from superagent.data import TrainingTask
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, write
from tools.selected_encoder_audit import portable_predictions


def check_stage(live, replay):
    protocol = read(live / 'protocol.json')
    replay_protocol = read(replay / 'protocol.json')
    replay_results = read(replay / 'results.json')
    result = read(live / 'results.json')
    total_recovered = total_lost = 0
    training_tokens = prediction_tokens = 0
    portable_source = None
    folds = []
    for index, (locations, split) in enumerate(zip(protocol['locations'], protocol['indices'], strict=True)):
        number = protocol['fold_numbers'][index]
        directory = Path(locations['encoder-mixture'])
        incumbent = Path(locations['encoder-depth'])
        fold_dir = live / f'fold_{number-1}'
        replay_dir = replay / f'fold_{index}'
        choice = read(directory / 'encoder_mixture.json')
        original = read(replay_dir / 'decision.json')
        selection = read(directory / 'final_selection.json')
        confirmation = read(directory / 'confirmation.json')
        history = read(directory / 'trials.json')
        strict_history = read(incumbent / 'trials.json')
        primary_changed = []
        for row, earlier in zip(history, strict_history, strict=True):
            expect(row['candidate'] == earlier['candidate'], 'The depth policy tried a different candidate')
            file = f"trial_{row['trial']}.joblib"
            p, q = [joblib.load(folder / file)['search_predictions'] for folder in (directory, incumbent)]
            primary_changed.append(int(np.sum(p != q)))
        expect(primary_changed == [0] * 7, 'The matched trajectory changed before blending')

        full = joblib.load(directory / 'refit_data.joblib')
        rebuilt = learn_from_confirmations(TrainingTask(full['kind'], full['X_train'], full['y_train']), history,
                                           confirmation, directory)
        expect(rebuilt == choice and original['eligible'] == choice['eligible'], 'Calibration decisions differ')
        if choice['eligible']:
            np.testing.assert_allclose(choice['raw'], original['raw'], rtol=0, atol=1e-10)
        bundle = joblib.load(directory / 'selected_model.joblib')
        model = bundle['model']
        mixed = isinstance(model, EncoderMixture)
        weight = choice.get('raw', 1.)
        expect(mixed == (choice['eligible'] and weight < 1.) and bundle['candidate'] == selection['family'],
               'Actual exported model differs from the frozen selection')
        if mixed:
            expect(selection['family'] == MIXTURE and selection['ensemble_components'] == [choice['anchor'], choice['other']]
                   and model.weight == weight, 'Wrong ensemble components or weight')
            if portable_source is None and 0 < weight < 1:
                portable_source = directory
        rec = receipt(directory)
        expect(rec['fits'] == 14 and rec['calls'] == rec['course_rmb'] == rec['api_rmb'] == 0, 'Unexpected resources')
        training_tokens += rec['minilm_tokens']
        prediction_tokens += feature_usage(fold_dir / 'encoder-mixture_prediction_features')['input_tokens']
        actual = read(fold_dir / 'outer_predictions.json')
        previous = read(replay_dir / 'outer_predictions.json')
        expect(split['outer'] == actual['rows'] == previous['rows'] and actual['truth'] == previous['truth'],
               'Outer row ordering changed')
        y, pred, base = (np.asarray(actual[key]) for key in ('truth', 'encoder-mixture', 'text-encoder'))
        np.testing.assert_array_equal(actual['encoder-depth'], previous['incumbent'])
        np.testing.assert_array_equal(pred, previous['learned'])
        recovered = int(np.sum((pred == y) & (base != y)))
        lost = int(np.sum((pred != y) & (base == y)))
        total_recovered += recovered
        total_lost += lost
        folds.append(dict(fold=number, rows=len(y), correct=int(np.sum(pred == y)), eligible=choice['eligible'],
                          confirmed_encoder=confirmation['selected_candidate'], selected=bundle['candidate'],
                          weight_on_full_refit=weight, inner_distinct_rows=choice.get('unique_rows', 0),
                          inner_row_visits=choice.get('visits', 0), correct_small_mistakes=recovered,
                          lose_small_correct=lost, search_prediction_differences=primary_changed,
                          outer_prediction_differences_from_replay=0, model_sha256=selection['model_sha256']))
    expect(result['correct']['encoder-mixture'] == replay_results['correct']['learned'] == sum(f['correct'] for f in folds),
           'Wrong published total')
    expect(total_recovered-total_lost == result['correct']['encoder-mixture']-result['correct']['text-encoder'],
           'Incorrect error attribution')
    stats = dict(rows=result['rows'], correct=result['correct'], wins=result['wins']['encoder-mixture'],
                 qualified=result['qualified']['encoder-mixture'], new_classifier_fits=result['new_fits'],
                 new_generation_calls=result['new_api_calls'], local_feature_training_tokens=training_tokens,
                 local_feature_scoring_tokens=prediction_tokens, correct_small_mistakes=total_recovered,
                 lose_small_correct=total_lost, diagnostic_correct=replay_results['correct'], folds=folds,
                 protocol_sha256=digest(live/'protocol.json'), replay_protocol_sha256=digest(replay/'protocol.json'),
                 replay_reference_sha256=replay_protocol['reference_sha256'])
    return stats, portable_source


def check_portability(source, small_weights, deep_weights, directory):
    evidence = directory/'checks.json'
    if not evidence.exists():
        result = portable_predictions(source, small_weights, directory, deep_weights=deep_weights)
        write(evidence, result)
    result = read(evidence)
    expect(digest(directory/'relocated'/'selected_model.joblib') == digest(source/'selected_model.joblib')
           == result['model_sha256'], 'Relocated model differs')
    expect(digest(directory/'original.csv') == digest(directory/'relocated.csv') == result['prediction_sha256'],
           'Portable predictions changed')
    for label in ('original','relocated'):
        expect(feature_usage(directory/(label+'.csv.features')) == result['usage'][label],
               'Portable feature ledger differs')
    expect(len(result['input_tokens_by_model']['relocated']) == 2 and result['usage']['cached']['input_tokens'] == 0,
           'Both encoders must support a zero-forward cache replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for x in ('root', 'small-weights', 'deep-weights', 'portability-output', 'output'):
        parser.add_argument('--'+x, type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args()
    stages = {}
    selected = None
    for seed in args.seeds:
        live, replay = args.root/f'live-mixture-{seed}', args.root/f'encoder-weight-{seed}'
        expect(read(live/'protocol.json')['seed'] == seed, 'Wrong seed')
        stages[str(seed)], source = check_stage(live, replay)
        if selected is None:
            selected = source
    expect(selected is not None, 'No exported nontrivial two-encoder model')
    portability = check_portability(selected, args.small_weights.resolve(), args.deep_weights.resolve(),
                                    args.portability_output.resolve())
    report = dict(stages=stages, portable_new_synthetic_examples=portability,
                  scope='The three group partitions reuse the same training-only development pool. Each final model is an independent normal-policy fit. Replays and hindsight ceilings are reported separately.')
    write(args.output, report)
    print({seed:stage['correct'] for seed,stage in stages.items()}, flush=True)


if __name__ == '__main__':
    main()
