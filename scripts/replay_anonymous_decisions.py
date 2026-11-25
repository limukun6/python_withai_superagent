"""Intervene on already measured training feedback, preserving costs, actions available and row order."""
import argparse
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import joblib
import numpy as np

from superagent.ab_run import registered_capabilities, sha
from superagent.autonomous_run import installed_methods, planner_observation
from superagent.experiment_policy import actions
from superagent.metrics import MetricSpec
from superagent.open_methods import capability


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def key(choice):
    return choice['action'],tuple(choice['members']),choice['split']


def audit(directory):
    folder=directory/'run/agent'
    spec=read(folder/'task_contract.json')
    profile=read(folder/'training_profile.json')
    history=read(folder/'history.json')
    decisions=read(folder/'decisions.json')
    metric=MetricSpec.from_payload({'metric_spec':spec['metric']})
    selection=read(folder/'final_selection.json')
    payload={}
    for fold in ('primary','confirmation_1','confirmation_2'):
        path=folder/(fold+'.joblib')
        if path.exists():
            payload[fold]=dict(labels=joblib.load(path)['y_search'],sha256=sha(path))
    width=profile['statistics'].get('feature_count') if spec['kind']=='numeric_tabular' else None
    with installed_methods():
        static=registered_capabilities(spec,selection['training_rows'],numeric_kernel_width=width)
    records={fold:{} for fold in payload}
    blocked=set()
    changes,exact,permuted=[],0,0
    for step,decision in enumerate(decisions):
        prior=history[:step]
        menu=decision['alternatives']
        names={m for option in menu for m in option['members']}
        names.update(r['candidate'] for r in prior)
        registry=read(folder/decision['registry'])['entries']
        caps={n:static[n] if n in static else capability(registry[n]) for n in sorted(names)}
        def select(candidates, observations, earlier):
            return key(actions(candidates,profile,earlier,observations,metric,list(records),
                               decision['remaining_fits'],blocked)[0])
        real=key(decision['choice'])
        if select(caps,records,prior)!=real or select(dict(reversed(caps.items())),records,prior)!=real:
            raise ValueError('Recorded decision could not be replayed: '+str(step+1))
        exact+=1
        # Exchange complete observations only between models measured on identical sets of folds.
        # The blinded exchange changes the recalled score, errors, probabilities and training gap together.
        for left,right in combinations(records['primary'],2):
            if any((left in fold)!=(right in fold) for fold in records.values()):
                continue
            altered={f:dict(v) for f,v in records.items()}
            for fold in altered.values():
                if left in fold:
                    fold[left],fold[right]=fold[right],fold[left]
            revised=deepcopy(prior)
            for row in revised:
                if row['candidate'] not in (left,right) or row['status']!='completed':
                    continue
                other=right if row['candidate']==left else left
                source=next(r for r in prior if r['candidate']==other and r['split_id']==row['split_id'])
                row.update(train_score=source['train_score'],search_score=source['search_score'])
            different=select(caps,altered,revised)
            permuted+=1
            if different!=real:
                changes.append(dict(step=step+1,exchange=[left,right],actual=real,alternative=different))
        if step<len(history):
            row=history[step]
            if row['status']=='failed':
                blocked.add(row['candidate'])
                continue
            bundle=joblib.load(folder/f'trial_{row["trial"]}.joblib')
            observations=planner_observation(bundle,bundle['metrics'],payload[row['split_id']],metric)
            records[row['split_id']][row['candidate']]=observations
    return dict(replayed=exact,unchanged_by_menu_order=exact,exchanges=permuted,changed=len(changes),examples=changes[:5])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={p.name:audit(p) for p in sorted(args.directory.iterdir()) if (p/'sealed.json').exists()}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print({n:{k:v for k,v in r.items() if k!='examples'} for n,r in result.items()})
