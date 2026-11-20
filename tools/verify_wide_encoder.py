"""Compare newly trained permissive-gate models with every frozen training and prediction decision."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from superagent.data import TrainingTask
from superagent.encoder_mixture import EncoderMixture, learn_from_confirmations
from superagent.method_choice import TEXT_DEEP
from superagent.minilm_features import feature_usage
from superagent.pretrained_choice import digest
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, write
from tools.verify_live_mixture import check_portability


ARM = 'encoder-mixture-wide'


def check_stage(stage, previous, seed):
    protocol = read(stage/'protocol.json')
    measured = read(stage/'results.json')
    replay = previous/f'relaxed-mixture-{seed}'
    rows = read(replay/'results.json')
    gates = previous/f'depth-gate-{seed}'
    expect(protocol['seed']==seed and measured['rows']==rows['rows'], 'Different study population')
    components = []
    portable = None
    training_tokens = scoring_tokens = 0
    recovered = lost = 0
    for index,(paths,split,number) in enumerate(zip(protocol['locations'],protocol['indices'],protocol['fold_numbers'],strict=True)):
        folder = Path(paths[ARM])
        expected_gate = read(gates/f'fold_{index}'/'decision.json')
        expected_run = Path(paths[expected_gate['observed_run']])
        current_gate = read(folder/'method_decision_7.json')
        used_depth = current_gate['candidate']==TEXT_DEEP
        expect(current_gate['mode']==ARM and current_gate['depth_gain_standard_errors_required']==-1
               and used_depth==expected_gate['use_deep'], 'The online gate differs from replay')
        history = read(folder/'trials.json')
        old_history = read(expected_run/'trials.json')
        for row,old in zip(history,old_history,strict=True):
            expect(row['candidate']==old['candidate'] and row['trial']==old['trial'], 'The online plan differs from replay')
            candidate = f"trial_{row['trial']}.joblib"
            np.testing.assert_array_equal(joblib.load(folder/candidate)['search_predictions'],
                                          joblib.load(expected_run/candidate)['search_predictions'])
        confirmation = read(folder/'confirmation.json')
        old_conf = read(expected_run/'confirmation.json')
        expect(confirmation['shortlist']==old_conf['shortlist'] and confirmation['selected_candidate']==old_conf['selected_candidate'],
               'The confirmed leading model changed')
        evidence = read(folder/'encoder_mixture.json')
        full = joblib.load(folder/'refit_data.joblib')
        rebuilt = learn_from_confirmations(TrainingTask(full['kind'],full['X_train'],full['y_train']),history,confirmation,folder)
        reference = read(replay/f'fold_{index}'/'decision.json')
        expect(evidence==rebuilt and evidence['eligible']==reference['eligible'], 'Training-only calibration changed')
        weight = evidence.get('raw',1.)
        np.testing.assert_allclose(weight,reference['weight'],atol=1e-10,rtol=0)
        rec = receipt(folder)
        expect(rec['fits']==14 and rec['course_rmb']==rec['api_rmb']==rec['calls']==0,'Unexpected study expense')
        bundle = joblib.load(folder/'selected_model.joblib')
        mixed = isinstance(bundle['model'],EncoderMixture)
        expect(mixed==(evidence['eligible'] and weight<1) and bundle['candidate']==rec['winner'],'Wrong exported policy')
        if mixed and 0<weight<1 and portable is None:
            portable=folder
        recorded = read(stage/f'fold_{number-1}'/'outer_predictions.json')
        replayed = read(replay/f'fold_{index}'/'outer_predictions.json')
        expect(recorded['truth']==replayed['truth'] and split['outer']==recorded['rows']==replayed['rows'],'Different held-out rows')
        np.testing.assert_array_equal(recorded[ARM],replayed['relaxed_mix'])
        truth=np.asarray(recorded['truth'])
        new=np.asarray(recorded[ARM])==truth
        baseline=np.asarray(recorded['text-encoder'])==truth
        recovered+=int(np.sum(new & ~baseline))
        lost+=int(np.sum(~new & baseline))
        training_tokens+=rec['minilm_tokens']
        scoring_tokens+=feature_usage(folder.parent/(ARM+'_prediction_features'))['input_tokens']
        components.append(dict(fold=number,rows=len(truth),correct=int(new.sum()),chose_deeper=used_depth,combined=mixed,
                               anchor=confirmation['selected_candidate'],anchor_weight=weight,
                               calibration_rows=evidence.get('unique_rows',0),new_model_sha256=rec['model_sha256'],
                               seconds=rec['seconds'],matched_search_arrays=7,changed_replay_predictions=0))
    counts=measured['correct']
    expect(counts[ARM]==rows['correct']['relaxed_mix'] and recovered-lost==counts[ARM]-counts['text-encoder'],
           'Wrong scored totals')
    wins=measured['wins'][ARM]
    against={name:counts[ARM]-counts[name]>=.002*measured['rows'] and n>len(components)/2 for name,n in wins.items()}
    expect(all(against.values())==measured['qualified'][ARM],'Reported promotion status differs')
    return dict(seed=seed,rows=measured['rows'],correct=counts,wins=wins,meets_each_comparison=against,
                meets_all=all(against.values()),new_fits=measured['new_fits'],api_calls=measured['new_api_calls'],
                local_feature_training_tokens=training_tokens,local_feature_outer_tokens=scoring_tokens,
                correct_small_mistakes=recovered,lose_small_correct=lost,folds=components,
                study_protocol_sha256=digest(stage/'protocol.json'),replay_results_sha256=digest(replay/'results.json')),portable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('root','previous','small-weights','deep-weights','portability-output','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    args=parser.parse_args()
    stats={}
    source=None
    for seed in args.seeds:
        result,model=check_stage(args.root/f'wide-live-{seed}',args.previous,seed)
        stats[str(seed)]=result
        if source is None:source=model
    expect(source is not None,'No nontrivial composite was exported')
    moved=check_portability(source,args.small_weights.resolve(),args.deep_weights.resolve(),args.portability_output.resolve())
    write(args.output,dict(stages=stats,portable_synthetic=moved,
        scope='All validation rows belong to the released training pool. The same rows are re-partitioned across seeds. Seven primary fits, six confirmation fits and one full refit per fold.'))
    print({s:a['correct'] for s,a in stats.items()},flush=True)


if __name__=='__main__':
    main()
