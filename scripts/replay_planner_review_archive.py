"""Evaluate planning-rule revisions on later tasks using only observed training outcomes."""
import argparse
from pathlib import Path

import numpy as np

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.exploration_rule import features, forecast


PROTOCOL = ROOT / 'research/rsi_planner_1125/archive_protocol.json'


def episode(run):
    decisions = read(run / 'decisions.json')
    history = read(run / 'history.json')
    observations = []
    menus = []
    prior = 0
    for index, decision in enumerate(decisions):
        require(decision['observed_trials'] == [r['trial'] for r in history[:index]],
                'Decision received a future training outcome')
        if prior:
            menus.append((prior, decision['alternatives']))
        if index == len(history):
            break
        row = history[index]
        if row['status'] != 'completed' or row['split_id'] != 'primary':
            continue
        choice = decision['choice']
        require(choice['action'] == 'explore' and choice['members'] == [row['candidate']],
                'The chosen action does not match the measured outcome')
        if prior:
            observations.append((prior, choice['evidence'], row['search_score'], row['trial']))
        prior += 1
    return {'observations': observations, 'menus': menus}


def propose(parent, memory, step):
    vectors, targets = [], []
    for task in memory.values():
        rows = task['observations']
        if not rows:
            continue
        # A long training trace should carry the same total weight as a short trace.
        scale = 1 / np.sqrt(len(rows))
        for prior, evidence, score, _ in rows:
            vectors.append(scale * features(prior, evidence))
            targets.append(scale * (score - forecast(parent, prior, evidence)))
    require(bool(vectors), 'Cannot propose before any eligible training observations')
    matrix = np.asarray(vectors)
    direction = np.linalg.solve(matrix.T @ matrix + np.eye(3), matrix.T @ targets)
    return np.clip(parent + step * direction, -0.12, 0.12)


def top(menu, prior, weights):
    def utility(action):
        if action['action'] != 'explore':
            return action['utility']
        evidence = action['evidence']
        return action['utility'] + (forecast(weights, prior, evidence) - evidence['expected'])

    action = max(menu, key=utility)
    return action['action'], action['members'], action['split']


def challenge(parent, child, task):
    forecasts = [
        {'trial': trial, 'observed': score, 'parent': forecast(parent, prior, evidence),
         'child': forecast(child, prior, evidence)}
        for prior, evidence, score, trial in task['observations']
    ]
    changed = sum(top(menu, prior, parent) != top(menu, prior, child)
                  for prior, menu in task['menus'])
    result = {'measured': len(forecasts), 'menus': len(task['menus']),
              'changed_options': changed, 'forecasts': forecasts, 'gain': None}
    if forecasts:
        for version in ('parent', 'child'):
            result[version + '_mae'] = float(np.mean([abs(row['observed'] - row[version])
                                                      for row in forecasts]))
        result['gain'] = result['parent_mae'] - result['child_mae']
    return result


def run(root):
    destination = root / 'cross-task-review'
    destination.mkdir()
    paths = {key: root / 'floor' / key / 'run' / 'agent' for key in ('table', 'titles')}
    paths.update({key: root / 'paired' / key / 'original' / 'run' / 'agent'
                  for key in ('n17', 'n29', 'n43', 'n61')})
    save(destination / 'protocol.json', {
        'plan': read(PROTOCOL), 'plan_sha256': sha(PROTOCOL),
        'implementation_sha256': sha(Path(__file__)),
    })
    parent, step, parent_id = np.zeros(3), 0.6, 0
    memory = {'table': episode(paths['table'])}
    stages = []
    for version, cases in enumerate((('titles',), ('n17', 'n29'), ('n43', 'n61')), 1):
        require(not set(cases).intersection(memory), 'Challenge task seen before freezing proposal')
        child = propose(parent, memory, step)
        frozen = {'version': version, 'parent': parent_id, 'parent_weights': parent.tolist(),
                  'child_weights': child.tolist(), 'fit_sources': list(memory),
                  'challenge_sources': list(cases), 'proposal_step': step}
        save(destination / f'proposal_{version}.json', frozen)
        challenges = {key: episode(paths[key]) for key in cases}
        scores = {key: challenge(parent, child, task) for key, task in challenges.items()}
        gains = [result['gain'] for result in scores.values()]
        changed = sum(result['changed_options'] for result in scores.values())
        gain = float(np.mean(gains)) if None not in gains else None
        passed = gain is not None and gain >= 0.003 and min(gains) >= -0.02 and changed > 0
        if passed:
            parent_id, parent = version, child
            step = min(1.0, step * 1.25)
        else:
            step = max(0.15, step * 0.5)
        result = dict(frozen, scores=scores, changed_options=changed, mean_gain=gain,
                      status='accepted' if passed else 'rejected', retained_version=parent_id)
        save(destination / f'evaluation_{version}.json', result)
        stages.append(result)
        memory.update(challenges)
    save(destination / 'results.json', {
        'stages': stages, 'active_version': parent_id, 'new_classifier_fits': 0,
        'live_api_requests': 0, 'external_scores_used': False,
        'scope': 'Forecasts on observed actions. Alternative rankings are counterfactual.',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root.resolve())
