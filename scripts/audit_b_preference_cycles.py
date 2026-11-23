"""Count incompatible observed comparisons for a single, transitive title ranking."""
import argparse
import csv
import io
import itertools
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from b_frozen_head_frontier import require,save,sha


def bounds(comparisons):
    edges=Counter()
    identical=0
    for a,b in comparisons:
        if a==b:
            identical+=1
        else:
            edges[a,b]+=1
    titles=sorted({t for pair in edges for t in pair})
    index={s:i for i,s in enumerate(titles)}
    forced=opposing=0
    residual={}
    for a,b in {tuple(sorted(pair)) for pair in edges}:
        ab,ba=edges[a,b],edges[b,a]
        opposing+=int(min(ab,ba)>0)
        forced+=min(ab,ba)
        if ab!=ba:
            key=(index[a],index[b]) if ab>ba else (index[b],index[a])
            residual[key]=abs(ab-ba)
    ii=[e[0] for e in residual]
    jj=[e[1] for e in residual]
    graph=coo_matrix((np.ones(len(ii)),(ii,jj)),shape=(len(titles),len(titles))).tocsr()
    _,components=connected_components(graph,directed=True,connection='strong')
    internal={}
    for (i,j),weight in residual.items():
        if components[i]==components[j]:
            internal.setdefault(int(components[i]),[]).append(weight)
    n=sum(edges.values())
    lower=forced+sum(min(weights) for weights in internal.values())
    upper=forced+sum(sum(weights)//2 for weights in internal.values())
    return dict(distinct_titles=len(titles),nonself_comparisons=n,self_comparisons_excluded=identical,
        unique_unordered_pairs=len({tuple(sorted(k)) for k in edges}),pairs_with_both_outcomes=opposing,
        minimum_errors_from_opposite_observations=forced,majority_graph_cycle_components=len(internal),
        total_edges_inside_cycle_components=sum(sum(w) for w in internal.values()),
        unavoidable_error_lower_bound=lower,unavoidable_error_upper_bound=upper,
        attainable_correct_interval=[n-upper,n-lower])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    # Verify the bounds against every total order on small graphs, without a classifier.
    cases=[[('a','b'),('b','c'),('c','a')],[('a','b'),('a','b'),('b','a')],
           [('a','b'),('b','c'),('c','a'),('a','d'),('d','b'),('b','d')]]
    for case in cases:
        report=bounds(case)
        exact=min(sum(order.index(i)>order.index(j) for i,j in case)
                  for order in itertools.permutations(sorted({v for pair in case for v in pair})))
        require(report['unavoidable_error_lower_bound']<=exact<=report['unavoidable_error_upper_bound'],'Invalid bounds')
    import json
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        require(set(manifest['splits'])=={'train'} and manifest['pair_label_one']=='left','Expected training-only preferences')
        member=manifest['splits']['train']['path']
        with z.open(member) as f:
            rows=list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')))
    require(len(rows)==6608 and {r['outcome'] for r in rows}=={'0','1'},'Unexpected training data')
    pairs=[(r['option_a'],r['option_b']) if r['outcome']=='1' else (r['option_b'],r['option_a']) for r in rows]
    save(a.output,dict(rows=len(rows),literal_titles=bounds(pairs),archive_sha256=sha(a.archive),member=member,
         source_sha256=sha(Path(__file__)),scope='Structural bounds on observed training labels only',
         actual_supervised_fits=0,model_api=0,official_test=False))


if __name__=='__main__':
    main()
