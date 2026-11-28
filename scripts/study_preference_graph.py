"""Measure two-hop preference evidence using only recorded training comparisons."""
import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, require, sha
from superagent.data import TrainingTask
from superagent.splits import normalized_text, sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/preference_graph_protocol.json"
SOURCES={"course":"analogy-portfolio-raw/portfolio_revised", "movie":"movie-adviser/revised"}


def audit(rows,truth,groups):
    raw=defaultdict(set)
    component={}
    self_pairs=0
    for (left,right),y,group in zip(rows,truth,groups):
        left,right=normalized_text(left),normalized_text(right)
        if left==right:
            self_pairs+=1
            continue
        a,b=sorted((left,right))
        winner=left if y==1 else right
        raw[a,b].add(winner==a)
        component[a,b]=group
    incoming,outgoing=defaultdict(set),defaultdict(set)
    for (a,b),evidence in raw.items():
        if len(evidence)==1:
            winner,loser=(a,b) if True in evidence else (b,a)
            incoming[loser].add(winner)
            outgoing[winner].add(loser)
    paths=defaultdict(set)
    owner={}
    for middle in incoming.keys()&outgoing.keys():
        for a,c in product(incoming[middle],outgoing[middle]):
            if a==c:
                continue
            target=tuple(sorted((a,c)))
            paths[target].add(target[0]==a)
            owner[target]=component[tuple(sorted((a,middle)))]
    correct_by_group=defaultdict(list)
    novel=Counter()
    ambiguous_paths=0
    unresolved_observed=0
    for pair,directions in paths.items():
        if len(directions)!=1:
            ambiguous_paths+=1
        elif pair not in raw:
            novel[owner[pair]]+=1
        elif len(raw[pair])!=1:
            unresolved_observed+=1
        else:
            correct_by_group[owner[pair]].append(raw[pair]==directions)
    correct=sum(sum(x) for x in correct_by_group.values())
    checked=sum(map(len,correct_by_group.values()))
    interval=None
    if checked:
        parts=[correct_by_group[k] for k in sorted(correct_by_group)]
        size=np.asarray([len(x) for x in parts]); wins=np.asarray([sum(x) for x in parts])
        rng=np.random.default_rng(429019)
        estimates=[]
        for _ in range(40):
            idx=rng.integers(len(parts),size=(100,len(parts)))
            estimates.extend(wins[idx].sum(axis=1)/size[idx].sum(axis=1))
        interval=np.quantile(estimates,[.025,.975]).tolist()
    accuracy=correct/checked if checked else None
    accepted=bool(len(correct_by_group)>=20 and accuracy>=.75 and interval[0]>=.60 and
                  sum(novel.values())/len(truth)>=.01 and len(novel)>=20)
    sizes=Counter(groups)
    return dict(rows=len(rows),components=len(sizes),largest_component_rows=max(sizes.values()),
        rows_in_repeated_components=sum(v for v in sizes.values() if v>1),unique_direct_edges=len(raw),
        contradictory_direct_edges=sum(len(x)>1 for x in raw.values()),identical_sides=self_pairs,
        two_way_inferences=ambiguous_paths,unresolved_direct_outcomes=unresolved_observed,
        observed_witnesses=checked,agree=correct,disagree=checked-correct,witness_groups=len(correct_by_group),
        agreement=accuracy,cluster_interval=interval,novel_pairs=sum(novel.values()),novel_groups=len(novel),
        augmentation_warranted=accepted)


def run(root):
    out=root/"preference-graph"
    out.mkdir(exist_ok=False)
    paths={m:root/name/"run/agent/refit_data.joblib" for m,name in SOURCES.items()}
    lock(out/"inputs.json",dict(protocol=sha(PLAN),training={m:sha(p) for m,p in paths.items()}))
    results={}
    for name,path in paths.items():
        d=joblib.load(path)
        rows,y=d["X_train"],d["y_train"]
        require(d["kind"]=="pair_text" and d["pair_semantics"]=="preference" and set(y)=={0,1},
                "No comparable preference contract")
        groups=sample_groups(TrainingTask(d["kind"],rows,y),"shared_text")
        results[name]=audit(rows,y,groups)
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),tasks=results,fits=0,model_api_requests=0,
         evaluation_inputs_opened=False,official_test_opened=False,H2_opened=False))
    print(results,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
