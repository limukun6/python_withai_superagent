"""Independently regrade frozen CSVs, check physical receipts, and export portable evidence."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.experiment_policy import identity


EVIDENCE = ROOT/'research/rsi_lineage_1125'


def inspect(root, scores):
    cases = read(root/'protocol.json')['cases']
    receipts = list(root.glob('*/*/sealed.json'))
    checked = []
    for f in sorted(receipts):
        folder = f.parent
        rule, key = folder.name, folder.parent.name
        spec = cases[key]
        source = root/key
        require(spec['train'] == sha(source/'p.zip') and spec['features'] == sha(source/'q.zip') and
                spec['truth'] == sha(root/'evaluation'/(key+'.npy')), 'Evaluation inputs differ')
        rec = read(f)
        agent = folder/'run/agent'
        final = read(agent/'final_selection.json')
        manifest = read(folder/'delivery/prediction_manifest.json')
        model = sha(agent/final['model_file'])
        require(rule == identity(read(agent/'exploration_rule.json')) and model == final['model_sha256'] == rec['model'],
                'Rule or model changed')
        pred = folder/'delivery/predictions.csv'
        require(sha(pred) == manifest['prediction_sha256'] == rec['prediction'] == sha(folder/'replay/predictions.csv'),
                'Independent delivery mismatch')
        y = np.load(root/'evaluation'/(key+'.npy'), allow_pickle=False)
        with pred.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        require([r['row_id'] for r in rows] == list(map(str, range(len(y)))), 'Prediction order changed')
        predicted = np.asarray([r['prediction'] for r in rows], dtype=y.dtype)
        if spec['metric'] == 'macro_f1':
            value = f1_score(y, predicted, average='macro')
        elif spec['metric'] == 'balanced_accuracy':
            value = balanced_accuracy_score(y, predicted)
        else:
            require(spec['metric'] == 'accuracy', 'Unexpected metric')
            value = accuracy_score(y, predicted)
        require(abs(value - scores[key,rule]['score']) < 1e-12 and int(np.sum(y==predicted)) == scores[key,rule]['correct'],
                'Independent scoring failed')
        physical = len((folder/'fits.jsonl').read_text().splitlines())
        outcome = read(agent/'run_outcome.json')
        ledger = [json.loads(line) for line in (agent/'llm_usage.jsonl').read_text().splitlines()]
        require(len(ledger) == 1 and ledger[0]['event'] == 'ledger_initialized' and ledger[0]['provider'] is None
                and ledger[0]['input_tokens'] == ledger[0]['output_tokens'] == ledger[0]['course_cost_rmb'] == 0,
                'Unexpected model use in local-only run')
        require(physical == rec['fits'] == scores[key,rule]['fits'] == outcome['completed_classifier_fits'] <= 16
                and not outcome['formal_test_accessed'] and outcome['new_api_requests'] == rec['api'] == 0,
                'Use differs from receipt')
        checked.append(dict(case=key, rule=rule, score=float(value), metric=spec['metric'], rows=len(y),
                       fits=physical, wall_seconds=outcome['wall_seconds'], model=model, prediction=sha(pred),
                       trace=rec['training_trace'], first_level_action_changes=rec['inherited_top_changes']))
    require({(r['case'],r['rule']) for r in checked} == set(scores), 'Incomplete or extra paths')
    return checked


def score_index(panels):
    return {(case,row['rule']):row for panel in panels for case,arms in panel.items() for row in arms.values()}


def verify(main, extra):
    old, new = read(main/'completion.json'), read(extra/'completion.json')
    main_runs = inspect(main, score_index([w['cases'] for w in old['waves']] + [old['final']]))
    extra_runs = inspect(extra, score_index([new['challenge'],new['transfer']]))
    runs = main_runs + extra_runs
    require(len({(r['case'],r['rule']) for r in runs}) == len(runs), 'Duplicate cold run')
    fits = sum(r['fits'] for r in runs)
    require(fits == old['fits'] + new['fits'], 'Physical total differs')
    for file, value in [('completion.json',old), ('descendant_result.json',new),
                        ('input_freeze.json',read(main/'protocol.json')), ('descendant_input_freeze.json',read(extra/'protocol.json')),
                        ('rule_history.json',dict(proposals=[read(main/f'wave_{i}_proposal.json') for i in range(1,4)],
                                                 final=read(main/'final_freeze.json'),next_proposal=read(extra/'proposal.json'))),
                        ('promoted_rule.json',read(main/'rules'/(old['final_roles']['selected']+'.json'))),
                        ('descendant_rule.json',read(extra/'rules'/(new['proposal']['roles']['child']+'.json'))),
                        ('receipts.json',dict(runs=runs, cold_runs=len(runs), physical_fits=fits,
                             model_api_calls=0, independent_sklearn_rescore=True, replayed_in_new_process=True))]:
        save(EVIDENCE/file,value)
    print(len(runs),'predictions independently rescored;', fits,'physical classifier fits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('extension', type=Path)
    args = parser.parse_args()
    verify(args.main.resolve(), args.extension.resolve())
