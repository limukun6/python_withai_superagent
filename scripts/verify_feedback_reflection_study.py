"""Check the frozen predictions and local accounting independently, then export aggregate evidence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts.unified_released_study import read, require, save, sha
from superagent.feedback_reflection import local_review


def independent_score(metric, y, prediction):
    if metric == 'balanced_accuracy':
        return balanced_accuracy_score(y, prediction)
    if metric == 'macro_f1':
        return f1_score(y, prediction, average='macro')
    if metric == 'accuracy':
        return accuracy_score(y, prediction)
    raise ValueError(f'Unsupported study metric {metric}')


def evaluated_action(packet, index):
    action, members, fold = packet['options'][index][:3]
    return dict(action=action, methods=[packet['methods'][i][0] for i in members], fold=fold)


def later_default(run, step, default):
    for row in read(run/'decisions.json')[step:]:
        choice = row['choice']
        if (choice['action'],choice['members'],choice['split']) == (default['action'],default['methods'],default['fold']):
            return row['step']
    return None


def export(root, destination):
    completion = read(root/'completion.json')
    protocol = read(root/'protocol.json')
    frozen = read(root/'all_predictions_frozen.json')
    ledger = read(root/'pending_merge.json')
    by_request = {e['id']: e for e in ledger['requests']}
    seen = set()
    cases = {}
    for task, result in completion['cases'].items():
        y = np.load(root/'evaluation'/(task+'.npy'), allow_pickle=False)
        arms, hits = {}, {}
        for mode, arm in result['arms'].items():
            path = root/task/mode
            receipt = arm['receipt']
            with (path/'delivery/predictions.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            require([r['row_id'] for r in rows] == list(map(str, range(len(y)))), 'Output order differs')
            predictions = np.asarray([r['prediction'] for r in rows], dtype=y.dtype)
            hits[mode] = predictions == y
            require(sha(path/'delivery/predictions.csv') == frozen[task][mode] == sha(path/'replay/predictions.csv')
                    and int(sum(hits[mode])) == arm['correct']
                    and abs(independent_score(result['metric'], y, predictions) - arm['score']) < 1e-12,
                    'Independent scoring differs')
            reflection = receipt['reflection']
            decision = None
            if reflection:
                packet = read(path/'run/agent/reflection_input.json')
                d = reflection['decision']
                decision = dict(step=reflection['step'], state=reflection['status'], input_hash=reflection['input_digest'],
                                verdict=d['verdict'], default=evaluated_action(packet, 0),
                                chosen=evaluated_action(packet, d['option']), cited=d['supporting_trials'],
                                contrary=d['contrary_trials'], reason=d['reason'], falsifier=d['refute_if'],
                                threshold=d['minimum_score'], internal_check=reflection.get('observed_check'))
                if d['option'] != 0:
                    decision['original_action_later_at_step'] = later_default(path/'run/agent',reflection['step'],decision['default'])
                if mode == 'local':
                    without_prior = local_review({**packet, 'earlier_primary_explorations': {}})
                    decision['same_action_without_archive'] = without_prior['option'] == d['option']
            usage = []
            for e in receipt['calls']:
                key = e['budget_request_id']
                native = by_request[key]
                events = [t['event'] for t in ledger['transitions'] if t['request_id']==key]
                require(key not in seen and native['scope'] == 'feedback-'+task+'-1125'
                        and events == ['reserved', native['status']], 'Request ledger mismatch')
                seen.add(key)
                if e['course_cost_rmb'] is not None:
                    settlement = json.loads(native['payload'])['resolution']
                    expected = e['input_tokens'] + 4 * e['output_tokens']
                    require(native['status']=='known' and settlement['input_tokens']==e['input_tokens']
                            and settlement['output_tokens']==e['output_tokens'] and native['course_units']==1000*expected
                            and abs(e['course_cost_rmb']-expected/1e6)<1e-12, 'Independent pricing differs')
                usage.append({k:e[k] for k in ('served_model','input_tokens','output_tokens','course_cost_rmb','actual_cost_rmb')})
            arms[mode] = dict(score=arm['score'], correct=arm['correct'], n=arm['rows'], fits=receipt['fits'],
                              final_members=receipt['winner'], decision=decision, prediction_sha256=frozen[task][mode],
                              split_sha256=receipt['split'], prefix_sha256=receipt['prefix'], usage=usage)
        for label, comparison in result['comparisons'].items():
            left, right = label.split('_vs_')
            require(int(sum(hits[left] & ~hits[right]))==comparison['corrected'] and
                    int(sum(~hits[left] & hits[right]))==comparison['spoiled'], 'Paired outcome counts differ')
        cases[task] = dict(metric=result['metric'], train_rows=protocol['inputs'][task]['train'],
                           arms=arms, comparisons=result['comparisons'])
    require(set(by_request)==seen and len(seen)==completion['api_calls'], 'Unmatched accounting entries')
    complete = completion['completed_runs']==12
    counts = completion['comparisons']['flash_vs_local']
    valid = sum(c['arms'].get('flash',{}).get('decision',{}).get('state')=='valid'
                for c in cases.values() if c['arms'].get('flash',{}).get('decision'))
    result = dict(protocol_sha256=sha(root/'protocol.json'), source=protocol['source'],
                  base_submission='d501807e61b5df1ac431d8fdf300c4bf4aaa4c2f',
                  complete=complete, valid_flash_replies=valid, flash_superiority_verified=(
                      complete and valid==4 and counts.get('win',0)>=2 and counts.get('loss',0)==0),
                  merge_state=ledger['team_merge_state'], team_total_verified=ledger['team_total_verified'],
                  cases=cases, **{k:completion[k] for k in ('comparisons','completed_runs','fits','api_calls','unknown_calls',
                                                            'known_rmb','known_input','known_output','prior','formal_test_accessed')})
    save(destination, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(args.root, args.destination)
