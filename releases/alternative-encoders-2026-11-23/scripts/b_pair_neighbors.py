"""A signed nearest-neighbor preference score with at most one vote per training group."""
import numpy as np


class PairNeighbors:
    def fit(self, x, y, groups):
        if not len(x) == len(y) == len(groups) or set(np.unique(y)) != {0,1}:
            raise ValueError('Invalid training index')
        self.directions = np.asarray(x)
        self.signs = 2*np.asarray(y,dtype=int)-1
        _, self.groups = np.unique(groups,return_inverse=True)
        return self

    def probability(self, queries):
        result = np.full(len(queries), .5)
        for start in range(0,len(queries),128):
            similarity = queries[start:start+128] @ self.directions.T
            ordered = np.argsort(-np.abs(similarity),axis=1,kind='stable')
            for ix,(s,indices) in enumerate(zip(similarity,ordered,strict=True)):
                used = set()
                votes = total = 0.
                for j in indices:
                    group = self.groups[j]
                    if group in used:
                        continue
                    used.add(group)
                    value = float(s[j])
                    weight = value*value
                    votes += weight*np.sign(value)*self.signs[j]
                    total += weight
                    if len(used)==21 or not weight:
                        break
                if total:
                    result[start+ix] = .5*(1+votes/total)
        return result
