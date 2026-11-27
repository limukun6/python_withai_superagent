"""Sequential decisions from anonymous training facts, measurable alternatives and fit costs."""
from itertools import combinations
import hashlib
import json
import math
import re

import numpy as np


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def option_keys(capabilities, primary=None):
    """Order real executable configurations independently of menu labels and enumeration."""
    from collections import Counter
    fields=("estimator","estimator_class","input_kind","representation","params","resource",
            "classifier_fits","refit_classifier_fits","supports_probabilities","swap_label_rule")
    documents={name:identity({k:cap[k] for k in fields if k in cap}) for name,cap in capabilities.items()}
    duplicates=Counter(documents.values())
    primary=primary or {}
    result={}
    for name, signature in documents.items():
        evidence=""
        if duplicates[signature]>1 and name in primary:
            row=primary[name]
            evidence=identity([np.asarray(row["predictions"]).tolist(),
                               None if row.get("probabilities") is None else np.asarray(row["probabilities"]).tolist()])
        result[name]=(signature,evidence)
    return result


def group_key(members, keys):
    return tuple(sorted(keys[n] for n in members))


def descriptor(cap):
    params = cap['params']
    settings = {**params, **params.get('kwargs', {})}
    features = set(re.findall(r'[a-z]+', cap['representation'].lower()))
    features.difference_update({'and', 'of', 'with', 'columns', 'features'})
    depths = settings.get('depth') or settings.get('max_depth') or 0
    return (cap['estimator'], params.get('encoder'), features, math.log1p(depths),
            math.log(max(1e-9, float(settings.get('C', 1)))),
            math.log1p(float(settings.get('l2_leaf_reg', settings.get('alpha', 0)))))


def similarity(a, b):
    shared = len(a[2] & b[2]) / max(1, len(a[2] | b[2]))
    distance = (.32*(a[0] != b[0]) + .23*(a[1] != b[1]) + .27*(1-shared)
                + .10*min(1, abs(a[3]-b[3])) + .04*min(1, abs(a[4]-b[4]))
                + .04*min(1, abs(a[5]-b[5])))
    return math.exp(-5*distance)


def fit_prior(cap, profile):
    """Weak representation/capacity preferences. Evidence overrides the cold-start prior."""
    stats = profile['statistics']
    n = stats['train_rows']
    _, encoder, features, depth, _, _ = descriptor(cap)
    text_length = stats.get('word_count', {}).get('median', 0)
    cardinalities = stats.get('unique_values_per_column', [])
    degrees = math.log2(1 + n * profile['minority_class_fraction'])
    native = 'native' in features and bool(cardinalities)
    capacity = math.exp(-abs(math.expm1(depth)-max(1., .65*degrees)) / max(1., degrees)) if depth else 0.
    richness = math.log1p(len(features & {'word','char','style','stats','surprisal','topic','cell','gradient'}))
    return (.045*native + .025*capacity + .038*bool(encoder)*min(1., text_length/8)
            + .014*richness - .018*bool(encoder)/math.sqrt(max(1., n/128)))


def measured_options(observations, metric, *, allow_votes=True, member_order=None):
    """Measure single models and equal votes only on identical, class-aligned, ordered rows."""
    primary = observations.get('primary', {})
    result = {}
    ordered = sorted(primary, key=member_order)
    groups = [(name,) for name in ordered]
    if allow_votes:
        groups.extend(combinations(ordered, 2))
    for members in groups:
        by_fold = {}
        for fold, records in observations.items():
            if any(m not in records for m in members):
                continue
            rows = [records[m] for m in members]
            if len({row['row_key'] for row in rows}) != 1:
                raise ValueError('Unmatched evaluation row identity or label order')
            if len(rows)==1:
                by_fold[fold] = rows[0]['score']
            elif all(row['probabilities'] is not None for row in rows):
                matrix = np.mean([row['probabilities'] for row in rows], axis=0)
                labels = np.asarray(metric.labels)
                prediction = labels[matrix.argmax(axis=1)]
                by_fold[fold] = metric.evaluate(rows[0]['labels'], prediction, matrix, labels)['score']
        if 'primary' in by_fold:
            result[members] = by_fold
    return result


def actions(caps, profile, history, observations, metric, folds, remaining, blocked=()):
    primary = observations.get('primary', {})
    keys = option_keys(caps, primary)
    first = {n:primary[n] for n in sorted(primary,key=keys.__getitem__)}
    options = measured_options(observations,metric,allow_votes='linear_only' not in profile.get('method_restrictions',()),
                               member_order=keys.__getitem__)
    def choice_key(action):
        return (group_key(action['members'],keys),action['action'],action['split'] or "",
                group_key(action.get('evidence',{}).get('target_members',()),keys))
    missing = frozenset(blocked)
    descriptors = {n: descriptor(c) for n,c in caps.items()}
    # Row counts, rather than how often a candidate was selected, determine its evidence weight.
    size = {f: len(next(iter(rows.values()))['labels']) for f,rows in observations.items() if rows}
    def mean(values):
        return float(np.average(list(values.values()), weights=[size[f] for f in values]))
    def error_scale(score, n):
        return math.sqrt(max(.02, score*(1-score))/max(n,1))
    maximum = max((r['score'] for r in first.values()), default=profile['majority_class_fraction'])
    prior_center = profile['majority_class_fraction'] if metric.name=='accuracy' else .5
    # Relate numerical uncertainty to observed search outcomes and the scale of the primary split.
    se = error_scale(maximum, size.get('primary', max(1,profile['statistics']['train_rows']//5)))
    choices = []
    def add(kind, members, split, value, why, **evidence):
        choices.append(dict(action=kind, members=list(members), split=split, utility=float(value),
                            reason=why, evidence=evidence))
    refit = lambda members: sum(caps[m]['refit_classifier_fits'] for m in members)
    singles = {key[0]: mean(values) for key,values in options.items() if len(key)==1}
    gaps = {row['candidate']:max(0.,row['train_score']-row['search_score']) for row in history
            if row['status']=='completed' and row['split_id']=='primary'}
    # A vote must improve upon both constituents on at least two matching folds in aggregate.
    for members, values in options.items():
        if refit(members)>remaining:
            continue
        expected = mean(values)
        paired_gain = None
        if len(members)>1:
            paired_gain = expected-max(mean({f: options[(m,)][f] for f in values}) for m in members)
            if len(values)<2 or paired_gain<=0 or any(values[f]<max(options[(m,)][f] for m in members)-se for f in values):
                continue
        value = expected - error_scale(expected, sum(size[f] for f in values)) / math.sqrt(len(values))
        add('stop', members, None, value, 'Best supported deployable result after accounting for uncertainty.',
            scores=values, paired_gain=paired_gain, refit_fits=refit(members))
    stops = [a for a in choices if a['action']=='stop']
    incumbent = max((a['utility'] for a in stops), default=prior_center)
    if first and maximum >= 1 and any(len(options[(m,)])>=2 and min(options[(m,)].values())==1 for m in first):
        return sorted(stops, key=lambda a:(-a['utility'], len(a['members']), choice_key(a)))
    leader = min(stops, key=lambda a:(-a['utility'],refit(a['members']),choice_key(a)), default=None)
    incumbent_cost = refit(leader['members']) if leader else 1
    for name, cap in caps.items():
        reserve = max(incumbent_cost, cap['refit_classifier_fits'])
        if name in first or name in missing or cap['classifier_fits']+reserve>remaining:
            continue
        prior = fit_prior(cap, profile)
        weights = [similarity(descriptors[name], descriptors[m]) for m in first]
        novelty = 1-max(weights, default=0.)
        if first:
            scores = [singles[m] for m in first]
            expected = float(np.average(scores, weights=np.square(weights)))
            prior_term = prior/(1+sum(weights))
            spread = max(se, float(np.std(scores)), (1-maximum)/math.sqrt(1+len(first)))
            overfit = float(np.average([gaps.get(m,0.) for m in first], weights=weights))
            value = expected + prior_term + (spread+overfit*.25)*novelty / math.sqrt(1+len(first)) - se
        else:
            expected, value = prior_center, prior_center + prior
        add('explore', (name,), 'primary', value, 'Estimate from similar measured configurations and the value of new evidence.',
            expected=expected, prior=prior, novelty=novelty, reserved_refits=reserve)
    for members, values in options.items():
        if any(m in missing for m in members):
            continue
        expected = mean(values)
        gain = 0.
        if len(members)>1:
            gain = expected - max(mean({f: options[(m,)][f] for f in values}) for m in members)
            if gain<=0:
                continue
        for fold in folds[1:]:
            absent = [m for m in members if m not in observations.get(fold, {})]
            if not absent:
                continue
            need = sum(caps[m]['classifier_fits'] for m in absent)
            reserve = max(incumbent_cost, refit(members))
            if need + reserve <= remaining and expected + se >= incumbent:
                uncertainty = error_scale(expected, sum(size[f] for f in values))
                value = expected + uncertainty / math.sqrt(len(values)*need) + gain - se*.35
                target = min(absent, key=lambda m:(-singles[m], keys[m]))
                add('confirm', (target,), fold, value, 'Check a competitive result on different training groups.',
                    target_members=list(members), matched_scores=values, paired_gain=gain, reserved_refits=reserve)
            break
    # Never let enumeration order, dataset filenames or display aliases break a tie.
    choices.sort(key=lambda a:(-a['utility'], len(a['members']), choice_key(a)))
    return choices
