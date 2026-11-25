"""Score a frozen pairwise exploration rule only where both future outcomes were observed."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_planner_lineage import original_traces, trace_digest


PLAN = ROOT/'research/rsi_lineage_1125/action_audit.json'
PROTOCOL = ROOT/'research/rsi_lineage_1125/action_audit_protocol.json'


def observed_pairs(run):
    hist, decisions = read(run/'history.json'), read(run/'decisions.json')
    primary = {r['candidate']: (i, r['search_score']) for i, r in enumerate(hist)
               if r['split_id'] == 'primary' and r['status'] == 'completed'}
    stop = run/'stop_review.json'
    if stop.exists():
        record = read(stop)
        for row in record['comparisons']:
            if row['fold'] == 'primary':
                name = record['candidate']
                require(name not in primary, 'Expected an untried challenger')
                primary[name] = (len(hist), row['challenger'])
    pairs = {}
    for i, decision in enumerate(decisions):
        history = hist[:i]
        require([r['trial'] for r in history] == decision['observed_trials'], 'Future observations in state')
        previous = [r for r in history if r['split_id'] == 'primary' and r['status'] == 'completed']
        if not previous:
            continue
        gap = np.clip(max(r['train_score']-r['search_score'] for r in previous), 0., 1.)
        feasible = {}
        for action in decision['alternatives']:
            if action['action'] != 'explore' or action['split'] != 'primary':
                continue
            name = action['members'][0]
            if name in primary and primary[name][0] >= i:
                e = action['evidence']
                feasible[name] = (action['utility'], np.array([e['novelty'], e['expected'], e['prior'], e['novelty']*gap]))
        for a, b in combinations(sorted(feasible), 2):
            if (a, b) not in pairs:
                pairs[a,b] = (feasible[a], feasible[b], primary[a][1]-primary[b][1])
    return list(pairs.values())


def learn(memory):
    design, residual = [], []
    for pairs in memory.values():
        non_ties = [(a, b, delta) for a, b, delta in pairs if abs(delta) > 1e-12]
        for (u, x), (v, y), delta in non_ties:
            scale = 1/np.sqrt(len(non_ties))
            design.append(scale*(x-y))
            residual.append(scale*(delta-(u-v)))
    require(bool(design), 'Historical traces contain no two measured alternatives')
    matrix = np.array(design)
    return np.linalg.solve(matrix.T @ matrix + np.eye(4), matrix.T @ residual)


def credit(margin, delta):
    if abs(margin) <= 1e-12:
        return .5
    return float((margin > 0) == (delta > 0))


def grade(pairs, weights):
    result = dict(pairs=len(pairs), measured_non_ties=0, ties=0, base=0., revised=0., improved=0, worsened=0)
    for (u, x), (v, y), delta in pairs:
        if abs(delta) <= 1e-12:
            result['ties'] += 1
            continue
        base = credit(u-v, delta)
        revised = credit((u+np.clip(weights@x,-.1,.1)) - (v+np.clip(weights@y,-.1,.1)), delta)
        result['base'] += base
        result['revised'] += revised
        result['improved'] += int(revised > base)
        result['worsened'] += int(revised < base)
        result['measured_non_ties'] += 1
    return result


def run(root, prior, write):
    done = read(root/'completion.json')
    config = read(root/'protocol.json')
    paths = original_traces(prior)
    require(config['traces'] == {k: trace_digest(v) for k,v in paths.items()}, 'Historical memory changed')
    archive = {k: observed_pairs(path) for k,path in paths.items()}
    weights = learn(archive)
    receipt = dict(weights=weights.tolist(), historical_inputs=config['traces'], protocol=sha(PROTOCOL),
                   code=sha(Path(__file__)))
    save(root/'action_audit_rule.json', receipt)
    cohorts = {'fit_history': {k: grade(v, weights) for k,v in archive.items()}}
    waves = {k: root/k/wave['roles']['parent'] for wave in done['waves'] for k in wave['cases']}
    final = {k: root/k/done['final_roles']['root'] for k in done['final']}
    for cohort, cases in [('waves', waves), ('final', final)]:
        cohorts[cohort] = {}
        for k, folder in cases.items():
            path = folder/'run/agent'
            require(trace_digest(path) == read(folder/'sealed.json')['training_trace'], 'Unsealed training history')
            cohorts[cohort][k] = grade(observed_pairs(path), weights)
    totals = {name: {k: sum(s[k] for s in cases.values()) for k in next(iter(cases.values()))}
              for name,cases in cohorts.items()}
    improved_both = all(totals[g]['revised'] > totals[g]['base'] for g in ('waves','final'))
    changed_final = sum(v['improved']+v['worsened'] > 0 for v in cohorts['final'].values())
    result = dict(rule=receipt, source_completion=sha(root/'completion.json'), cases=cohorts, totals=totals,
                  final_tasks_with_changes=changed_final, supports_live_followup=improved_both and changed_final >= 2,
                  classifier_fits=0, api=0)
    save(root/'action_audit.json',result)
    if write:
        save(PLAN,result)
    print(totals, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('prior', type=Path)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    run(args.root.resolve(), args.prior.resolve(), args.export)
