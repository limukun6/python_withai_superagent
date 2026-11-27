"""Inspect two-step preference evidence within one fitting set of declared ordered pairs."""
from collections import defaultdict
import math

import numpy as np

from .data import TrainingTask
from .splits import sample_groups


def lower_bound(correct, total):
    if total == 0:
        return 0.
    z = 1.959963984540054
    p = correct/total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def inspect_order(rows, labels):
    y = np.asarray(labels)
    if len(rows) != len(y) or y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned binary preference labels")
    observed = defaultdict(set)
    groups_by_pair = {}
    groups = sample_groups(TrainingTask("pair_text", rows, y), "shared_text")
    for (left, right), label, group in zip(rows, y, groups):
        if left == right:
            continue
        pair = tuple(sorted((left, right)))
        observed[pair].add(left if label else right)
        groups_by_pair[pair] = group
    above, below = defaultdict(set), defaultdict(set)
    for (a, b), winners in observed.items():
        if len(winners) == 1:
            high = next(iter(winners))
            low = b if a == high else a
            above[low].add(high)
            below[high].add(low)
    inferred = defaultdict(set)
    skipped = 0
    for middle in above.keys() & below.keys():
        parents, children = above[middle], below[middle]
        if len(parents)*len(children) > 2048:
            skipped += 1
            continue
        for parent in parents:
            for child in children:
                if parent != child:
                    inferred[tuple(sorted((parent,child)))].add(parent)
    checked = correct = disagreements = 0
    checked_groups = set()
    proposed = []
    for (a, b), winners in sorted(inferred.items()):
        if len(winners) != 1:
            disagreements += 1
            continue
        winner = next(iter(winners))
        direct = observed.get((a, b))
        if direct is None:
            proposed.append((winner, b if winner == a else a))
        elif len(direct) == 1:
            checked += 1
            correct += winner in direct
            checked_groups.add(groups_by_pair[(a,b)])
    lower = lower_bound(correct, checked)
    required = max(20, math.ceil(.01 * len(rows)))
    evidence = dict(fitting_rows=len(rows), observed_distinct_pairs=len(observed),
                    conflicting_observations=sum(len(x)>1 for x in observed.values()),
                    checkable_pairs=checked, correct_pairs=correct, checked_components=len(checked_groups),
                    wilson_lower_95=lower, one_way_new_pairs=len(proposed), ambiguous_paths=disagreements,
                    skipped_dense_midpoints=skipped, minimum_new_pairs=required,
                    eligible=checked>=30 and len(checked_groups)>=10 and lower>.55 and len(proposed)>=required)
    return evidence, proposed
