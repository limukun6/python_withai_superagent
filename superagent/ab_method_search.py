"""Select a feasible configuration by observed performance and distance from previous experiments."""
from itertools import combinations
import hashlib
import json
import math

import numpy as np

from .metrics import score_of


def attributes(cap):
    params = cap['params']
    representation = str(cap.get('representation','')).lower()
    tags = frozenset(k for k in ('word','char','style','stats','surprisal','topic','latent','hybrid','rbf','direction')
                     if k in representation)
    return (cap['estimator'], params.get('encoder',''), cap['resource'], tags or frozenset([representation]),
            float(params.get('C',params.get('kwargs',{}).get('C',1.))))


def distance(first, second):
    a, b = first, second
    jaccard = 1 - len(a[3] & b[3]) / len(a[3] | b[3])
    regularization = min(1., abs(math.log(a[4]/b[4]))/math.log(100.))
    return .35*(a[0]!=b[0]) + .2*(a[1]!=b[1]) + .1*(a[2]!=b[2]) + .27*jaccard + .08*regularization


def signature(cap):
    # The order of the enumerated menu has no influence, even on ties.
    config = {'estimator':cap['estimator'],'representation':cap['representation'],'params':cap['params']}
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def choose_with_history(state,capabilities,eligible,verified,picker=None):
    budget=state['budget']
    if picker is not None and verified and budget['used_llm_requests']<budget['llm_request_limit']:
        chosen=picker(tuple(eligible))
        if chosen not in eligible:
            raise ValueError('History planner selected outside the feasible runtime menu')
        return dict(chosen=chosen,evidence_trials=[r['trial'] for r in verified])
    return rank(state,capabilities,eligible,verified)


def rank(state, capabilities, eligible, verified):
    profile = state['training_profile']
    rows = int(profile['statistics']['train_rows'])
    lengths = profile['statistics'].get('word_count', {})
    length_factor = math.sqrt(1 + lengths.get('p95', 0) / max(1, lengths.get('median', 1)))
    margin = float(profile['majority_class_fraction'])
    if rows < 1:
        raise ValueError('Missing training profile for local method selection')
    configs = {name:attributes(cap) for name,cap in capabilities.items()}
    structures = list({v[:4]: (*v[:4],1.) for v in configs.values()}.values())
    measured = [configs[r['candidate']] for r in verified]
    observed = [score_of(r) for r in verified]
    predictions = {v['trial']:np.asarray(v['predictions']) for v in state['prediction_evidence']}
    pairwise = [float(np.mean(predictions[a['trial']]!=predictions[b['trial']])) for a,b in combinations(verified,2)]
    disagreement = sum(pairwise)/len(pairwise) if pairwise else 0.
    # More unresolved errors or disagreement justify trying a more distant model.
    best = max(observed, default=margin)
    exploration = ((1-best) + disagreement)/math.sqrt(max(len(verified),1))
    scores = []
    for name in eligible:
        desc = configs[name]
        gaps = [distance(desc, other) for other in measured]
        center_distance = math.fsum(distance(desc,c) for c in structures)/len(structures)
        if observed:
            weights = [math.exp(-6*d) for d in gaps]
            expected = float(np.average(observed,weights=weights))
            novel = min(gaps)
            value = expected + exploration*novel
        else:
            # A central, economical first measurement. Small training sets penalize expensive extraction more.
            complexity = float(capabilities[name]['resource']!='cpu') * length_factor + min(1.,abs(math.log(desc[4]))/5.)
            expected, novel = margin, 0.
            value = -center_distance - complexity/math.sqrt(rows)
        scores.append(dict(candidate=name, strength=expected, nearest_distance=novel, acquisition=value,
                           config_sha256=signature(capabilities[name])))
    scores.sort(key=lambda x:(-x['acquisition'],x['config_sha256'],x['candidate']))
    if not scores:
        raise ValueError('No eligible configuration for local method search')
    return dict(chosen=scores[0]['candidate'], evidence_trials=sorted(r['trial'] for r in verified),
                observed_scores={r['candidate']:score_of(r) for r in verified}, rows=rows, disagreement=disagreement,
                exploration_weight=exploration, candidates=scores)
