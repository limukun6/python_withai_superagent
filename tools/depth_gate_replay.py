"""Test a more permissive depth decision using both observed, fully trained continuations."""
import argparse
import math
from pathlib import Path

import joblib
import numpy as np

from superagent.method_choice import TEXT_DEEP, TEXT_MINILM
from superagent.pretrained_choice import digest
from tools.frontier_study import expect
from tools.pretrained_study import read, write
from tools.soft_vote_diagnostic import freeze_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    protocol=read(args.reference/'protocol.json')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    freeze_file(root/'protocol.json',dict(reference_sha256=digest(args.reference/'protocol.json'),source_sha256=digest(Path(__file__)),
        rule='After the identical six primary trials, test deeper weights when the smaller hybrid is within one paired-disagreement '
             'standard deviation of the ordinary leader, enough common errors remain, and measured time/availability permits. '
             'Use the identical observed continuation of either the strict or fixed policy. Separately bound per-fold selection '
             'among the three actual complete policies. All branch choices freeze before reading held-out predictions.'))
    if args.freeze:return
    decisions=[]
    for i,locations in enumerate(protocol['locations']):
        strict,fixed=[Path(locations[n]) for n in ('encoder-depth','encoder-depth-control')]
        a,b=[read(p/'trials.json') for p in (strict,fixed)]
        expect([t['candidate'] for t in a[:6]]==[t['candidate'] for t in b[:6]] and a[5]['candidate']==TEXT_MINILM,'First six candidates differ')
        for j in range(1,7):
            np.testing.assert_array_equal(joblib.load(strict/f'trial_{j}.joblib')['search_predictions'],
                                          joblib.load(fixed/f'trial_{j}.joblib')['search_predictions'])
        record=read(strict/'method_decision_7.json')
        ev=record['evidence']
        supported=(ev['primary_metric']=='accuracy' and ev['discordant']>=8 and
                   ev['corrected']-ev['lost']>=-math.sqrt(ev['discordant']) and
                   ev['residual_shared_errors']>=max(12,.05*ev['search_rows']) and TEXT_DEEP in record['eligible'])
        strict_try=a[6]['candidate']==TEXT_DEEP
        expect(not strict_try or supported,'Relaxed gate must include the original positive decisions')
        path='encoder-depth-control' if supported and not strict_try else 'encoder-depth'
        chosen_trial=b[6] if path=='encoder-depth-control' else a[6]
        expect((chosen_trial['candidate']==TEXT_DEEP)==supported,'Requested continuation was not observed')
        decision=dict(use_deep=supported,old_gate_used_deep=strict_try,observed_run=path)
        dest=root/f'fold_{i}'
        dest.mkdir(exist_ok=True)
        freeze_file(dest/'decision.json',decision)
        decisions.append(decision)
    folds=[]
    for i,(indices,decision) in enumerate(zip(protocol['indices'],decisions,strict=True)):
        prior=read(args.reference/f"fold_{protocol['fold_numbers'][i]-1}"/'outer_predictions.json')
        expect(prior['rows']==indices['outer'],'Outer rows differ')
        y=np.asarray(prior['truth'])
        policies=('text-encoder','encoder-depth','encoder-depth-control')
        scores={k:int((np.asarray(prior[k])==y).sum()) for k in policies}
        best=max(policies,key=scores.get)
        choices=dict(small=policies[0],strict=policies[1],fixed=policies[2],relaxed=decision['observed_run'],oracle_three_policies=best)
        write(root/f'fold_{i}'/'outer_predictions.json',dict(rows=prior['rows'],truth=prior['truth'],**{k:prior[n] for k,n in choices.items()}))
        folds.append(dict(rows=len(y),correct={k:scores[n] for k,n in choices.items()}))
    totals={k:sum(row['correct'][k] for row in folds) for k in folds[0]['correct']}
    write(root/'results.json',dict(rows=sum(row['rows'] for row in folds),correct=totals,folds=folds,
        relaxed_depth_trials=sum(d['use_deep'] for d in decisions),changed_actions=sum(d['use_deep']!=d['old_gate_used_deep'] for d in decisions),
        additional_diagnostic_fits=0,additional_encoder_forwards=0,new_api_calls=0))
    print(totals,flush=True)


if __name__=='__main__':main()
