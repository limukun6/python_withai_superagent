"""Bound the effect of moving the trained encoder exploration gate between its two measured endpoints."""
import argparse
import math
from pathlib import Path

import numpy as np

from superagent.pretrained_choice import digest
from tools.frontier_study import expect
from tools.pretrained_study import read, write
from tools.soft_vote_diagnostic import freeze_file


CONTROLS=('text-encoder','encoder-depth','encoder-depth-control','encoder-mixture')


def evaluate(blocks, threshold):
    totals={}
    for seed,data in blocks.items():
        records=[]
        for observation in data:
            # Both policies already take exactly the same action outside their disagreement interval.
            take_wide=observation['strict_deep'] or (observation['relaxed_deep'] and observation['z']>=threshold)
            records.append(observation['wide'] if take_wide else observation['strict'])
        counts=np.asarray(records)
        comp=np.array([[fold['controls'][k] for k in CONTROLS] for fold in data])
        differences=counts[:,None]-comp
        margins=differences.sum(axis=0)
        wins=(differences>0).sum(axis=0)
        rows=sum(fold['rows'] for fold in data)
        totals[seed]=dict(correct=int(counts.sum()),fold_correct=counts.tolist(),wins=dict(zip(CONTROLS,wins.tolist())),
                          margins=dict(zip(CONTROLS,margins.tolist())),
                          qualifies=bool(np.all(margins>=.002*rows) and np.all(wins>len(counts)/2)))
    return totals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=True)
    freeze_file(dest/'protocol.json',dict(seeds=args.seeds,source_sha256=digest(Path(__file__)),
        inputs=[digest((args.root/f'relaxed-mixture-{s}')/'protocol.json') for s in args.seeds],
        rule='The midpoint gate tries the deeper encoder when the first encoder has corrected at least as many rows as it lost. Other eligibility checks stay fixed. In a separate hindsight calculation enumerate every common standardized corrected-minus-lost threshold from minus one to plus one. Both endpoint continuations and their training-fitted loss weights have already been measured. Also allow an oracle to choose the better endpoint separately in every fold.'))
    if args.freeze:return
    blocks={}
    values=[-1.,0.,1.]
    for seed in args.seeds:
        live=args.root/f'live-mixture-{seed}'
        wide=args.root/f'relaxed-mixture-{seed}'
        gates=args.root/f'depth-gate-{seed}'
        protocol=read(live/'protocol.json')
        expect((wide/'results.json').exists(),'Wider calibration incomplete')
        data=[]
        for i,locations in enumerate(protocol['locations']):
            ev=read(Path(locations['encoder-depth'])/'method_decision_7.json')['evidence']
            gate=read(gates/f'fold_{i}'/'decision.json')
            z=(ev['corrected']-ev['lost'])/math.sqrt(ev['discordant']) if ev['discordant'] else 0.
            if gate['use_deep']!=gate['old_gate_used_deep']:
                expect(-1<=z<1, 'Disagreement outside the measured gate interval')
                values.append(z)
            pred=read(wide/f'fold_{i}'/'outer_predictions.json')
            y=np.asarray(pred['truth'])
            controls={k:int((np.asarray(pred[k])==y).sum()) for k in CONTROLS}
            strict=controls['encoder-mixture']
            relaxed=int((np.asarray(pred['relaxed_mix'])==y).sum())
            if gate['use_deep']==gate['old_gate_used_deep']:
                np.testing.assert_array_equal(pred['encoder-mixture'],pred['relaxed_mix'])
            data.append(dict(z=z,rows=len(y),strict_deep=gate['old_gate_used_deep'],relaxed_deep=gate['use_deep'],
                             strict=strict,wide=relaxed,controls=controls))
        blocks[str(seed)]=data
    boundaries=np.unique(values)
    candidates=np.sort(np.r_[boundaries,(boundaries[1:]+boundaries[:-1])/2])
    possible=[evaluate(blocks,t) for t in candidates]
    for s in blocks:
        expected=read((args.root/f'relaxed-mixture-{s}')/'results.json')['correct']
        expect(evaluate(blocks,-1)[s]['correct']==expected['relaxed_mix'] and
               evaluate(blocks,1)[s]['correct']==expected['encoder-mixture'],'Gate endpoints differ')
    upper={s:sum(max(d['strict'],d['wide']) for d in data) for s,data in blocks.items()}
    valid=sum(all(v['qualifies'] for v in outcome.values()) for outcome in possible)
    best={s:max(o[s]['correct'] for o in possible) for s in blocks}
    report=dict(protocol_sha256=digest(dest/'protocol.json'),midpoint_train_rule=evaluate(blocks,0),
                number_of_boundary_and_interior_probes=len(candidates),
                unique_vectors_of_fold_correct_counts=len({tuple(tuple(v['fold_correct']) for v in o.values()) for o in possible}),
                any_shared_threshold_qualifies_in_every_partition=bool(valid),best_constant_threshold_per_seed=best,
                best_branch_chosen_separately_per_fold=upper,extra_classifier_fits=0,extra_encoder_forwards=0,new_api_calls=0)
    write(dest/'results.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
