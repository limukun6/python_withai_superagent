"""Bounded composition using only the existing validated text recipe factory."""
from itertools import product
import math

from .metrics import score_of
from .recipe_models import materialize_recipe


REGULARIZATION = (.3, 1.)


def recipe_spec(kind, analyzer, representation, strength):
    return dict(kind=kind,analyzer=analyzer,representation=representation,classifier='logistic',C=strength,dimensions=64)


def candidates(kind):
    if kind not in ('single_text','pair_text'):
        return []
    return [materialize_recipe(recipe_spec(kind,analyzer,representation,strength))
            for analyzer,representation,strength in product(('word','char_wb'),('latent','hybrid'),REGULARIZATION)]


ROUTES = {candidate.family:candidate.family.rsplit('_',1)[0] for kind in ('single_text','pair_text') for candidate in candidates(kind)}


def analyzer_from(row):
    params = row['params']
    if row['candidate'] in ROUTES:
        return params['analyzer']
    return {'word':'word','char':'char_wb','char_wb':'char_wb'}.get(params.get('representation'))


def propose(eligible, verified, caps):
    """Construct a legal target from verified primary observations, then recheck feasibility."""
    if not caps or not eligible or eligible[0] not in ROUTES or len(verified)<2:
        return None
    ranked = sorted(verified,key=lambda row:(-score_of(row),row['trial']))
    best = ranked[0]
    train = (best.get('train_metrics') or {}).get('score')
    if type(train) not in (float,int) or not math.isfinite(train) or not 0<=train<=1:
        return None
    by_analyzer = {}
    for row in ranked:
        analyzer = analyzer_from(row)
        if analyzer:
            by_analyzer.setdefault(analyzer,row)
    if len(by_analyzer)==2:
        chosen_analyzer = next(iter(by_analyzer))
        analyzer_basis = 'highest observed primary score across both analyzers'
    elif by_analyzer:
        chosen_analyzer = 'char_wb' if 'word' in by_analyzer else 'word'
        analyzer_basis = 'try the complementary analyzer'
    else:
        chosen_analyzer,analyzer_basis = 'word','first word analyzer without a measured single-analyzer reference'
    gap = train-score_of(best)
    representation = 'latent' if gap>.15 else 'hybrid'
    nominal = best['params'].get('C',.3)
    if type(nominal) not in (int,float) or not math.isfinite(nominal) or nominal<=0:
        nominal = .3
    strength = min(REGULARIZATION,key=lambda c:abs(math.log(c)-math.log(nominal)))
    kind = caps[eligible[0]]['kind']
    target = materialize_recipe(recipe_spec(kind,chosen_analyzer,representation,strength))
    possible = [name for name in eligible if name in ROUTES]
    def distance(name):
        params = caps[name]['params']
        return (params['analyzer']!=chosen_analyzer,params['representation']!=representation,params['C']!=strength)
    choice = min(possible,key=distance)
    spec = caps[choice]['params']
    if materialize_recipe(spec).family!=choice:
        raise ValueError('Proposed recipe does not resolve to its trusted executable')
    evidence = list(dict.fromkeys([best['trial'],*(row['trial'] for row in by_analyzer.values())]))
    return dict(reference_trial=best['trial'],reference_score=score_of(best),train_minus_search=gap,
                analyzer_basis=analyzer_basis,analyzer_reference_trials={a:r['trial'] for a,r in by_analyzer.items()},
                target_spec=target.params,target_id=target.family,target_eligible=target.family in possible,
                chosen_spec=spec,chosen_id=choice,default_id=eligible[0],evidence_trials=evidence,
                feasible_recipe_ids=possible)
