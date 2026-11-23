"""Learn one mixture weight on one panel, then test on the other with the base models frozen."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.optimize import brentq

from scripts.run_b_shared_full_task2 import check, read, seal
from superagent.ab_complement import canonical_probabilities
from superagent.ab_run import sha
from superagent.data import load_development_task
from superagent.models import predict_with_model
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take

PLAN = Path(__file__).resolve().parents[1]/'docs/batches/B_grouped_blend_plan_20261124_zh.md'


def row_weights(keys):
    _, inverse, count = np.unique(keys, return_inverse=True, return_counts=True)
    return 1/(len(count)*count[inverse])


def fit_weight(first, second, y, group):
    """Solve the derivative of the preregistered convex objective, including possible optima at the ends."""
    weights = row_weights(group)
    delta = first-second

    def derivative(w):
        p = second+w*delta
        return float(weights @ ((p-y)*delta/(p*(1-p))) + .01*(w-.5))

    if derivative(0) >= 0:
        return 0.
    if derivative(1) <= 0:
        return 1.
    return float(brentq(derivative, 0., 1., xtol=1e-13))


def classify(p, tie):
    return ((p > .5) | ((p == .5) & tie)).astype(int)


def nll(p, y):
    return float(-np.mean(np.where(y, np.log(p), np.log1p(-p))))


def run(a):
    def offline(event, args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Calibration diagnostic is offline')
    sys.addaudithook(offline)
    a.out.mkdir(parents=True, exist_ok=False)
    from_source = read(a.parent/'protocol.json')
    parent_seal = read(a.shadow/'completion.json')
    check(all(sha(a.shadow/(k+'.json')) == v for k,v in parent_seal.items()), 'Previous shadow study changed')
    shadow_protocol = read(a.shadow/'protocol.json')
    models_seal = read(a.shadow/'frozen.json')
    check(shadow_protocol['source_run_protocol'] == sha(a.parent/'protocol.json') and
          sha(a.parent/'anonymous.zip') == from_source['training'] and
          models_seal['protocol'] == sha(a.shadow/'protocol.json'), 'Parent source or split seal changed')
    files = [a.shadow/f'worker_{i}.joblib' for i in (0,1)]
    check([sha(f) for f in files] == models_seal['models'], 'Frozen base models differ')
    task = load_development_task(a.parent/'anonymous.zip')
    check(task.X_val is None and task.y_val is None, 'Expect a training-only archive')
    groups = sample_groups(task, 'shared_text')
    split = read(a.parent/'agent/primary_split.json')
    buckets = [int(hashlib.sha256(('b-borderline-20261123-v1/'+g).encode()).hexdigest(),16)%8 for g in groups]
    panels = [[i for i in split['train_indices'] if buckets[i] == n] for n in (0,1)]
    train = [i for i in split['train_indices'] if buckets[i] >= 2]
    check(fingerprint(train) == shadow_protocol['training_ids_sha256'] and
          [fingerprint(ids) for ids in panels] == [p['row_ids_sha256'] for p in shadow_protocol['panels']]
          and set(groups[panels[0]]).isdisjoint(groups[panels[1]]) and
          set(groups[train]).isdisjoint(groups[panels[0]+panels[1]]), 'Registered group membership changed')
    seal(a.out/'protocol.json',dict(plan_sha256=sha(PLAN), runner_sha256=sha(Path(__file__)),
         parent_protocol_sha256=sha(a.parent/'protocol.json'), shadow_completion_sha256=sha(a.shadow/'completion.json'),
         model_digests=models_seal['models'], panels=shadow_protocol['panels'], classifier_fit_cap=0,
         calibration_fit_cap=2, model_api_cap=0, ridge_penalty=.005, required_additional_correct=8))
    os.environ.update(SUPERAGENT_DEEP_MINILM_WEIGHTS=str(a.weights/'all-MiniLM-L12-v2'),
                      TOKENIZERS_PARALLELISM='false', TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1')
    models = [joblib.load(f)['model'] for f in files]
    states, receipts = [], []
    for n, ids in enumerate(panels):
        x = take(task.X_train, ids)
        y = task.y_train[ids]
        tie = np.array([left < right for left,right in x])
        first, second = [canonical_probabilities(model,x)[1][:,1] for model in models]
        check(all(np.all(np.isfinite(p) & (p>0) & (p<1)) for p in (first,second)), 'Expected interior probabilities')
        hard = predict_with_model('pair_text',models[0],x)
        equal = classify((first+second)/2, tie)
        check([fingerprint(hard),fingerprint(equal)] == models_seal['predictions'][n],
              'Reference single or equal-vote predictions changed')
        for model in models:
            # Keep padding and batch shape identical for this exact symmetry check.
            forward = canonical_probabilities(model,x[:16])[1][:,1]
            reverse = canonical_probabilities(model,[(r,l) for l,r in x[:16]])[1][:,1]
            np.testing.assert_allclose(reverse,1-forward,rtol=0,atol=1e-12)
        states.append((first,second,y,groups[ids],tie,hard,equal))
        receipts.append(dict(first=fingerprint(first),second=fingerprint(second),single=fingerprint(hard),
                             equal=fingerprint(equal),rows=fingerprint(ids)))
    seal(a.out/'base_predictions.json',dict(panels=receipts, base_model_fits=0, swap_pairs_per_panel=16))
    fitted = []
    for n, (p,q,y,group,*_) in enumerate(states):
        seal(a.out/f'fit_{n}_started.json',dict(source_panel=n,rows=len(y),groups=len(set(group)),
             features=sha(a.out/'base_predictions.json'),maximum_supervised_fits=1))
        weight = fit_weight(p,q,y,group)
        seal(a.out/f'fit_{n}_completed.json',dict(started=sha(a.out/f'fit_{n}_started.json'),
             source_panel=n, target_panel=1-n, weight_on_anchor=weight, supervised_calibration_fits=1))
        fitted.append(weight)
    summaries, evidence = [], []
    for n,(p,q,y,group,tie,hard,equal) in enumerate(states):
        weight = fitted[1-n]
        blend = weight*p+(1-weight)*q
        answer = classify(blend,tie)
        correct = [v==y for v in (hard,equal,answer)]
        counts = [int(v.sum()) for v in correct]
        buckets = defaultdict(lambda: [0]*8)
        for g,one,two,three in zip(group,*correct,strict=True):
            buckets[g][int(one)+2*int(two)+4*int(three)] += 1
        evidence.append([buckets[g] for g in sorted(buckets)])
        summaries.append(dict(rows=len(y),groups=len(buckets),weight_from_other_panel=weight,
             single=counts[0],equal=counts[1],learned=counts[2],net_vs_stronger=counts[2]-max(counts[:2]),
             nll=dict(single=nll(p,y),equal=nll((p+q)/2,y),learned=nll(blend,y)),
             learned_predictions_sha256=fingerprint(answer)))
    net = [entry['net_vs_stronger'] for entry in summaries]
    seal(a.out/'grouped.json', evidence)
    result = dict(panels=summaries,net_vs_stronger=sum(net),pass_gate=min(net)>=0 and sum(net)>=8,
                  new_base_fits=0,supervised_calibration_fits=2,model_api_calls=0,new_rmb=0,
                  designated_validation_used=False, formal_test_used=False)
    seal(a.out/'results.json', result)
    seal(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in ('protocol','base_predictions','grouped','results',
         'fit_0_started','fit_0_completed','fit_1_started','fit_1_completed')})
    print(result)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','shadow','weights','out'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
