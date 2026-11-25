"""Measure when two leading recorded experiments compete for fit budget, without running either again."""
import argparse
from pathlib import Path

from scripts.unified_released_study import read, require, save, sha
from superagent.experiment_policy import identity
from superagent.open_methods import materialize


def leading_pair(decision):
    pair = {}
    for action in decision['alternatives']:
        if action['action']=='stop':
            continue
        key = (action['members'][0],action['split'])
        pair.setdefault(key,action)
        if len(pair)==2:
            break
    return list(pair.values())


def audit(root, destination):
    frozen = read(root/'all_predictions_frozen.json')
    tasks = {}
    for case, modes in frozen.items():
        unique = {}
        for mode in modes:
            agent = root/case/mode/'run/agent'
            evidence = read(agent/'reflection_input.json')
            costs = {name:cost for name,_,_,cost in evidence['methods']}
            hist = read(agent/'history.json')
            for d in read(agent/'decisions.json'):
                past = [(h['candidate'],h['split_id'],h['status'],h.get('search_score'))
                        for h in hist[:len(d['observed_trials'])]]
                if sum(h[2]=='completed' for h in past)<3:
                    continue
                pair = leading_pair(d)
                if len(pair)<2:
                    continue
                reg = agent/d['registry']
                require(sha(reg)==d['registry_sha256'],'Changed method registry')
                entries = read(reg)['entries']
                next_fits = []
                for action in pair:
                    member = action['members'][0]
                    if member not in costs:
                        candidate = materialize(entries[member])
                        require(candidate.family==member,'Unknown proposed method')
                        costs[member] = candidate.params['classifier_fits_per_evaluation']
                    next_fits.append(costs[member])
                reserve = max(a['evidence']['reserved_refits'] for a in pair)
                available = d['remaining_fits']
                choices = [(a['members'][0],a['split']) for a in pair]
                state = identity([past,available,choices,next_fits,reserve])
                unique[state] = (available,available - sum(next_fits) - reserve,d['step'])
        stop = read(root/case/'control/run/agent/stop_review.json')
        deltas = [r['delta'] for r in stop.get('comparisons',[])]
        tasks[case] = dict(distinct_states=len(unique), insufficient_for_both=sum(v[1]<0 for v in unique.values()),
                           minimum_spare_after_both_and_refit=min(v[1] for v in unique.values()),
                           fourth_step_limits=sorted({v[0] for v in unique.values() if v[2]==4}),
                           control_stop_fits_left=stop.get('allowance_left'),control_stop_outcome=stop.get('status'),
                           control_stop_challenger_fold_deltas=deltas,
                           control_stop_fold_directions_disagree=any(v>0 for v in deltas) and any(v<0 for v in deltas))
    save(destination,dict(method='Retrospective training-log audit. After three completed trials, use the first two distinct '
        'non-stop (method, fold) actions in priority order. Charge their two immediate fits and the larger recorded '
        'final refit reserve. Collapse identical ordered training history, remaining credit, and leading pair within task.',
        interpretation='A nonnegative remainder establishes affordability of the immediate measurements at the checkpoint; '
        'later calibration, reordering, worker time, or confirming all members of a proposed ensemble can add requirements.',
        tasks=tasks, distinct_states=sum(t['distinct_states'] for t in tasks.values()),
        insufficient_for_both=sum(t['insufficient_for_both'] for t in tasks.values()),
        new_classifier_fits=0,model_api_requests=0,uses_outer_labels=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    audit(args.root,args.output)
