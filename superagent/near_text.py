"""Connected components of exact title groups and high-overlap multiword titles."""
from collections import Counter, defaultdict
import re
import unicodedata

import numpy as np


def word_pairs(title):
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", title).casefold())
    if len(words) < 6:
        return frozenset()
    pairs = frozenset(zip(words, words[1:]))
    return pairs if len(pairs) >= 5 else frozenset()


def matching_indices(sets):
    frequencies = Counter(item for features in sets for item in features)
    postings = defaultdict(list)
    for i, features in enumerate(sets):
        if not features:
            continue
        # A first common item must occur in both prefixes to leave room for
        # ceil(0.8 * size) common items. A shared order makes this lossless.
        prefix_size = len(features) - (4 * len(features) + 4) // 5 + 1
        prefix = sorted(features, key=lambda term: (frequencies[term], term))[:prefix_size]
        candidates = {j for term in prefix for j in postings[term]}
        for j in candidates:
            other = sets[j]
            if 5 * min(len(features), len(other)) < 4 * max(len(features), len(other)):
                continue
            common = len(features & other)
            if 9 * common >= 4 * (len(features) + len(other)):
                yield j, i
        for term in prefix:
            postings[term].append(i)


def merged_title_groups(rows, exact):
    group_by_title = {title: group for pair, group in zip(rows, exact) for title in pair}
    titles = sorted(group_by_title)
    features = [word_pairs(title) for title in titles]
    parents = {g: g for g in exact}
    def root(g):
        while parents[g] != g:
            parents[g] = parents[parents[g]]
            g = parents[g]
        return g
    for j, i in matching_indices(features):
        a, b = root(group_by_title[titles[j]]), root(group_by_title[titles[i]])
        parents[max(a, b)] = min(a, b)
    return np.asarray([root(g) for g in exact])
