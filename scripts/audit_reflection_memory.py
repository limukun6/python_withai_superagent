"""Test whether the historical novelty counts predict improvement on a different archived training task."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from scripts.unified_released_study import read, require, save, sha
from scripts.verify_planner_lineage import original_traces
from superagent.experiment_policy import identity


def audit(prior, protocol, output):
    task_rows = {}
    for task, directory in original_traces(prior).items():
        history_path, decisions_path = directory/'history.json', directory/'decisions.json'
        require(identity([sha(decisions_path),sha(history_path)])==protocol['history'][task], 'Training trace changed')
        rows = []
        best = None
        for observation, decision in zip(read(history_path), read(decisions_path)):
            if observation['status'] != 'completed' or observation['split_id'] != 'primary':
                continue
            score = observation['search_score']
            if best is not None:
                group = 'different' if decision['choice']['evidence']['novelty'] >= .5 else 'near'
                rows.append((group,int(score > best + 1e-12)))
            best = score if best is None else max(best, score)
        task_rows[task] = rows
    held_out, squared_errors = {}, {'pooled': [], 'by_novelty': []}
    for task, test in task_rows.items():
        train = [r for name, rows in task_rows.items() if name != task for r in rows]
        totals, improvements = Counter(g for g,_ in train), Counter(g for g,y in train if y)
        pooled = (sum(improvements.values())+1)/(sum(totals.values())+2)
        by_group = {g:(improvements[g]+1)/(totals[g]+2) for g in ('near','different')}
        error = {'pooled':[(pooled-y)**2 for _,y in test],
                 'by_novelty':[(by_group[g]-y)**2 for g,y in test]}
        require(bool(test), 'Empty historical held-out task')
        held_out[task] = {'n':len(test), 'improved':sum(y for _,y in test), 'brier':{k:mean(v) for k,v in error.items()}}
        for method, values in error.items():
            squared_errors[method].extend(values)
    wins = sum(x['brier']['by_novelty'] < x['brier']['pooled'] - 1e-12 for x in held_out.values())
    losses = sum(x['brier']['by_novelty'] > x['brier']['pooled'] + 1e-12 for x in held_out.values())
    save(output,dict(question='Does novelty-stratified historical success improve a probability forecast over pooled history?',
         unit='each completed primary exploration after the first, compared with that task’s best previous primary score',
         method='hold out an entire task, Beta(1,1) smoothing for both estimates, Brier loss (lower is better)',
         status='Exploratory diagnostic of the six archived training traces; not an independent end-to-end task evaluation.',
         n=len(squared_errors['pooled']), held_out=held_out, micro_brier={k:mean(v) for k,v in squared_errors.items()},
         macro_brier={k:mean(r['brier'][k] for r in held_out.values()) for k in squared_errors},
         by_novelty_task_wins=wins, by_novelty_task_losses=losses, by_novelty_task_ties=len(held_out)-wins-losses,
         new_classifier_fits=0, model_api_requests=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior',type=Path)
    parser.add_argument('protocol',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    audit(args.prior,read(args.protocol),args.output)
