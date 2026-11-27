"""Measure ranking stability of smaller fitted models on identical held development groups."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.models import predict_with_model
from superagent.prediction_evidence import _checked_predictions, fingerprint
from superagent.splits import sample_groups


PLAN = ROOT / 'research/paired_encoder_search_1127/training_fidelity_protocol.json'
ARMS = ('small', 'large')


def key_for(params):
    body = json.dumps(params, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def histories(agent):
    successes = {}
    for entry in read(agent / 'history.json'):
        if entry['split_id'] != 'primary' or entry['status'] != 'completed':
            continue
        key = key_for(entry['resolved_configuration']['params'])
        require(key not in successes, 'Duplicate executable configuration in the primary fold')
        path = agent / f"trial_{entry['trial']}.joblib"
        successes[key] = dict(name=entry['candidate'], trial=entry['trial'], path=path, sha256=sha(path))
    return successes


def rank_correlation(a, b):
    x, y = rankdata(a), rankdata(b)
    if min(float(np.std(x)), float(np.std(y))) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_ranks(keys, names, search_right, small, large, groups):
    small_correct, large_correct = small.sum(axis=0), large.sum(axis=0)
    def ranked(scores):
        return sorted(range(len(keys)), key=lambda i: (-int(scores[i]), keys[i]))
    large_leaders = np.flatnonzero(large_correct == np.max(large_correct)).tolist()
    def summary(scores):
        order = ranked(scores)
        return dict(first=names[order[0]], tied_first=[names[i] for i in order if scores[i] == scores[order[0]]],
                    first_three=[names[i] for i in order[:3]], retains_large_leader=bool(set(order[:3]) & set(large_leaders)),
                    top_one_regret_rows=int(np.max(large_correct)-large_correct[order[0]]),
                    top_three_oracle_regret_rows=int(np.max(large_correct)-np.max(large_correct[order[:3]])))
    units, inverse = np.unique(groups, return_inverse=True)
    counts = [np.zeros((len(units), len(keys)), dtype=int) for _ in ARMS]
    for data, count in zip((small, large), counts):
        np.add.at(count, inverse, data.astype(int))
    rng = np.random.default_rng(271126)
    draws = []
    for _ in range(1000):
        sample = rng.integers(0, len(units), size=len(units))
        corr = rank_correlation(*(v[sample].sum(axis=0) for v in counts))
        if corr is not None:
            draws.append(corr)
    common_correlation = rank_correlation(small_correct, large_correct)
    common_rule = summary(small_correct)
    return dict(search_vs_large_common=rank_correlation(search_right, large_correct),
                search_vs_small_common=rank_correlation(search_right, small_correct),
                small_vs_large_common=common_correlation,
                common_group_bootstrap_95pct=[float(x) for x in np.quantile(draws, [.025, .975])] if draws else None,
                valid_group_draws=len(draws), grouped_bootstrap_draws=1000, large_common=summary(large_correct),
                small_search=summary(search_right), small_common=common_rule,
                qualifies_for_controlled_trial=bool(common_correlation is not None and common_correlation>.60 and common_rule['retains_large_leader']))


def run(root):
    protocol = read(PLAN)
    out = root / 'training-fidelity'
    out.mkdir(exist_ok=True)
    source = root / 'raw/race/run/agent/refit_data.joblib'
    subset = read(root / protocol['sources']['rows'])
    require(sha(source)==subset['source'], 'Development source changed')
    original = joblib.load(source)
    x, y = original['X_train'], np.asarray(original['y_train'])
    groups = sample_groups(TrainingTask(original['kind'], x, y), 'shared_text')
    common = np.asarray(subset['held'])
    require(fingerprint(common)==subset['held_fingerprint'] and np.unique(common).size==len(common), 'Held row identity changed')
    agents = {arm: root / protocol['sources'][arm] for arm in ARMS}
    entries = {arm: histories(p) for arm, p in agents.items()}
    splits = {arm: read(p/'primary_split.json') for arm,p in agents.items()}
    payload = {arm: joblib.load(p/'primary.joblib') for arm,p in agents.items()}
    base = {'small': np.asarray(subset['train']), 'large': np.arange(len(x))}
    fitting = {arm: base[arm][sp['fit']] for arm,sp in splits.items()}
    searching = {arm: base[arm][sp['held']] for arm,sp in splits.items()}
    for arm in ARMS:
        for rows, field in [(fitting[arm], 'X_train'), (searching[arm], 'X_search')]:
            require(payload[arm][field]==[x[i] for i in rows], 'Mismatch in original row identities')
        require(not (set(groups[common]) & set(groups[fitting[arm]])), 'A fitting group entered the shared comparison')
    require(set(fitting['small'])<=set(fitting['large']) and set(searching['small'])<=set(fitting['large'])
            and set(common)<=set(searching['large']), 'Unexpected nesting of training sizes')
    keys = sorted(set(entries['small']) & set(entries['large']))
    require(len(keys)>2, 'Too few common candidates')
    omissions = {a:[v['name'] for k,v in entries[a].items() if k not in keys] for a in ARMS}
    identities = {k:{a:{n:v for n,v in entries[a][k].items() if n!='path'} for a in ARMS} for k in keys}
    lock(out/'inputs.json',dict(protocol=sha(PLAN), source=sha(source), rows=fingerprint(common),
        split_rows={a:dict(fit=len(fitting[a]), search=len(searching[a]),fit_id=fingerprint(fitting[a]),
                           search_id=fingerprint(searching[a]), payload=sha(agents[a]/'primary.joblib')) for a in ARMS},
        common_rows=len(common), common_groups=len(set(groups[common])), groups_crossing_training=0,
        candidates=identities, unshared=omissions))
    activate_resource_directory(root/'discovery-library')
    os.environ['SUPERAGENT_PAIRED_FEATURE_CACHE'] = str(root/'feature_bank')
    proofs, predictions = {}, []
    locate = {int(index): pos for pos,index in enumerate(searching['large'])}
    offsets = [locate[int(i)] for i in common]
    inputs = [x[i] for i in common]
    for k in keys:
        pair = {a:joblib.load(entries[a][k]['path']) for a in ARMS}
        require(pair['small']['model'] is not None, 'Missing retained small classifier')
        results = {}
        for a,t in pair.items():
            record = t['prediction_record']
            require(record['evaluation_payload_sha256']==splits[a]['payload_sha256'], 'Incorrect recorded search payload')
            results[a] = _checked_predictions(record, payload[a]['y_search'])
        small = pair['small']['model']
        replay = predict_with_model(original['kind'], small, payload['small']['X_search'])
        require(np.array_equal(replay, results['small']), 'Small search predictions did not replay')
        new_pred = predict_with_model(original['kind'], small, inputs)
        large_pred = results['large'][offsets]
        proofs[k] = dict(small_common=fingerprint(new_pred),large_common=fingerprint(large_pred),
                         small_search=fingerprint(replay),large_search=fingerprint(results['large']))
        predictions.append((results['small'],new_pred,large_pred,results['large']))
        print('predicted',len(predictions),'of',len(keys),flush=True)
    lock(out/'predictions.json',dict(inputs=sha(out/'inputs.json'),members=proofs))
    counts = []
    for i, labels in enumerate([y[searching['small']],y[common],y[common],y[searching['large']]]):
        counts.append(np.column_stack([p[i]==labels for p in predictions]))
    names = [entries['small'][k]['name'] for k in keys]
    columns = ('small_search','small_common','large_common','large_search')
    per_member = {name:{column:int(right[:,i].sum()) for column,right in zip(columns,counts)} for i,name in enumerate(names)}
    ranks = evaluate_ranks(keys,names,counts[0].sum(axis=0),counts[1],counts[2],groups[common])
    result = dict(prediction_lock=sha(out/'predictions.json'),rows={c:len(right) for c,right in zip(columns,counts)},
                  compared=len(keys),by_method=per_member,rank_analysis=ranks,retrospective=True,new_classifier_fits=0,
                  model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/'results.json',result)
    print(json.dumps(ranks,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args().root)
