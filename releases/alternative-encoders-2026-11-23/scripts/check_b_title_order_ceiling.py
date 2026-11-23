"""Count contradictory title comparisons and the exact best fixed ordering on the training graph."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
from itertools import permutations
import json
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(adjacency):
    seen=set()
    for first in adjacency:
        if first in seen:
            continue
        group=[]
        queue=[first]
        seen.add(first)
        while queue:
            node=queue.pop()
            group.append(node)
            for peer in adjacency[node]:
                if peer not in seen:
                    seen.add(peer)
                    queue.append(peer)
        yield group


def acyclic(size, edges):
    outgoing=[set() for _ in range(size)]
    indegree=[0]*size
    for u,v in edges:
        outgoing[u].add(v)
        indegree[v]+=1
    queue=[i for i,n in enumerate(indegree) if n==0]
    placed=0
    while queue:
        node=queue.pop()
        placed+=1
        for peer in outgoing[node]:
            indegree[peer]-=1
            if indegree[peer]==0:
                queue.append(peer)
    return placed==size


def maximum_order_matches(size, edges):
    # For a fixed last vertex, every edge pointing from an earlier vertex to it is satisfied.
    incoming=[[] for _ in range(size)]
    for (winner,loser),count in edges.items():
        incoming[loser].append((1<<winner,count))
    best=[0]*(1<<size)
    for mask in range(1,len(best)):
        options=[]
        for last in range(size):
            if mask & (1<<last):
                before=mask^(1<<last)
                options.append(best[before]+sum(n for bit,n in incoming[last] if before & bit))
        best[mask]=max(options)
    return best[-1]


def exhaustive_matches(size, edges):
    highest=0
    for order in permutations(range(size)):
        pos={v:i for i,v in enumerate(order)}
        highest=max(highest,sum(n for (u,v),n in edges.items() if pos[u]<pos[v]))
    return highest


def run(packet,control,plan):
    expected=json.loads(control.read_text())['interfaces']['files']['anonymous.zip']
    if digest(packet)!=expected:
        raise ValueError('Wrong frozen training packet')
    direction=defaultdict(Counter)
    neighbors=defaultdict(set)
    unordered=Counter()
    with zipfile.ZipFile(packet) as z:
        manifest=json.loads(z.read('manifest.json'))
        if set(manifest['splits'])!={'train'} or manifest['pair_label_one']!='left':
            raise ValueError('Wrong target semantics or data scope')
        with z.open('rows.csv') as f:
            for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')):
                a,b=r['option_a'],r['option_b']
                if a==b or r['outcome'] not in ('0','1'):
                    raise ValueError('Unexpected self pair or nonbinary training label')
                winner,loser=(a,b) if r['outcome']=='1' else (b,a)
                direction[winner][loser]+=1
                unordered[tuple(sorted((a,b)))]+=1
                neighbors[a].add(b)
                neighbors[b].add(a)
    total=sum(unordered.values())
    bound=dag_groups=curve_groups=rows_in_undirected_cycles=rows_in_directed_cycles=0
    hist=Counter()
    for group in components(neighbors):
        n=len(group)
        if n>10:
            raise ValueError('Training graph exceeds registered maximum component size')
        hist[n]+=1
        local={t:i for i,t in enumerate(group)}
        edges={(local[t],local[p]):v for t in group for p,v in direction[t].items()}
        observations=sum(edges.values())
        if sum(len(neighbors[t]) for t in group)//2>=n:
            curve_groups+=1
            rows_in_undirected_cycles+=observations
        optimum=maximum_order_matches(n,edges)
        if acyclic(n,edges):
            dag_groups+=1
            if optimum!=observations:
                raise ValueError('Dynamic program disagrees with topological ordering')
        else:
            rows_in_directed_cycles+=observations
            if optimum!=exhaustive_matches(n,edges):
                raise ValueError('Independent direct enumeration disagrees')
        bound+=optimum
    pairs_upper=sum(max(direction[a][b],direction[b][a]) for a,b in unordered)
    return dict(training_packet_sha256=expected,plan_sha256=digest(plan),rows=total,titles=len(neighbors),
        components=sum(hist.values()),sizes=dict(sorted(hist.items())),unique_unordered_pairs=len(unordered),
        repeated_observations=total-len(unordered),undirected_cyclic_components=curve_groups,
        rows_in_those_components=rows_in_undirected_cycles,directed_cyclic_components=sum(hist.values())-dag_groups,
        rows_in_directed_cyclic_components=rows_in_directed_cycles,minimum_direct_pair_conflicts=total-pairs_upper,
        maximum_simultaneously_orderable=bound,minimum_global_order_errors=total-bound,
        held_out_generalization_evaluated=False,new_classifier_fits=0,model_api_requests=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','control','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    plan=Path(__file__).resolve().parents[1]/'docs/batches/B_title_order_ceiling_1123_zh.md'
    result=run(args.packet,args.control,plan)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))
