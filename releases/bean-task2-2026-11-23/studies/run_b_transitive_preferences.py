"""Test training-only two-step preference comparisons against the full two-encoder control."""
import argparse
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_confirmation import split
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.ab_run import sha
from superagent.ab_sentence import AlternativeSentenceFeatures
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_transitive_preferences_1123_zh.md'


def pair_key(edge):
    return hashlib.sha256(json.dumps(edge, ensure_ascii=False).encode()).digest(), edge


def weak_comparisons(rows, labels, groups):
    edges_by_group = defaultdict(set)
    for (left, right), label, group in zip(rows, labels, groups, strict=True):
        if label not in (0, 1):
            raise ValueError('Directional preferences require binary labels')
        edges_by_group[group].add((left, right) if label else (right, left))
    weak = []
    info = dict(groups=len(edges_by_group), excluded_cyclic_groups=0, contributing_groups=0,
                direct_with_two_step_agreement=0, direct_with_two_step_conflict=0, eligible_pairs=0)
    for key in sorted(edges_by_group):
        edges = edges_by_group[key]
        neighbors = defaultdict(set)
        for winner, loser in edges:
            neighbors[winner].add(loser)
        paths = {(a,c) for a,b in edges for c in neighbors.get(b, ()) if a != c}
        # Both counts describe observed edges that can be checked via a different intermediate title.
        info['direct_with_two_step_agreement'] += len(edges & paths)
        info['direct_with_two_step_conflict'] += sum((b,a) in paths for a,b in edges)
        try:
            list(TopologicalSorter(neighbors).static_order())
        except CycleError:
            info['excluded_cyclic_groups'] += 1
            continue
        extras = {edge for edge in paths if edge not in edges and edge[::-1] not in edges}
        info['eligible_pairs'] += len(extras)
        info['contributing_groups'] += int(bool(extras))
        weak.extend(sorted(extras, key=pair_key)[:len(edges)])
    return weak, info


def stage(a):
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    check(sha(a.training) == INPUT_TRAIN, 'Wrong anonymous training packet')
    task = load_development_task(a.training)
    check(task.kind == 'pair_text' and task.spec.get('input_schema',{}).get('pair_label_one') == 'left'
          and task.X_val is None and task.y_val is None, 'Expected training-only left-wins task')
    old_curve = read(a.curve/'completion.json')
    old_e5 = read(a.e5/'completion.json')
    curve_seals = read(a.curve/'exports.json')['full']
    fam_file = a.curve/'full/familiarity/model.joblib'
    head_file = a.e5/'regular/classifier.joblib'
    for base, receipt in ((a.curve,old_curve),(a.e5,old_e5)):
        check(all(receipt[k] == sha(base/(k+'.json')) for k in ('protocol','exports','results')),
              'Historical files differ')
    check(sha(a.e5/'completion.json') == read(a.curve/'protocol.json')['control'] and
          sha(a.e5/'regular/fit_completed.json') == read(a.e5/'exports.json')['regular']['fit'] and
          sha(head_file) == read(a.e5/'regular/fit_completed.json')['model'] and
          sha(a.curve/'full/familiarity/fit_completed.json') == curve_seals['fits']['familiarity'] and
          sha(fam_file) == read(a.curve/'full/familiarity/fit_completed.json')['model'] and
          sha(a.curve/'full/predictions.npy') == curve_seals['file'], 'Supervised controls or predictions changed')
    rows, y = task.X_train, np.asarray(task.y_train)
    groups, parts, _ = split(rows,a.first,a.second)
    fitting = parts['train']
    evaluation = np.r_[parts['A'],parts['B']]
    identities = {k:fingerprint(ids) for k,ids in parts.items()}
    check(all(identities[k] == v['ids'] for k,v in read(a.curve/'protocol.json')['partitions'].items())
          and set(groups[fitting]).isdisjoint(groups[evaluation]), 'Shared training and evaluation titles')
    train = [rows[i] for i in fitting]
    query = [rows[i] for i in evaluation]
    a.out.mkdir(parents=True,exist_ok=True)
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN), runner=sha(Path(__file__)), training=INPUT_TRAIN,
                controls=[sha(a.curve/'completion.json'),sha(a.e5/'completion.json')], partitions=identities,
                fits_cap=1, model_api_cap=0))
    weak, info = weak_comparisons(train,y[fitting],groups[fitting])
    w = len(weak)
    frozen_save(a.out/'training_graph.json',dict(**info,observed_rows=len(train),added_weak_pairs=w,
                ordered_pairs_sha256=hashlib.sha256(json.dumps(weak,ensure_ascii=False).encode()).hexdigest(),
                raw_weak_weight=0.5, weight_multiplier=len(train)/(len(train)+0.5*w)))
    if w < 50:
        result = dict(insufficient_weak_pairs=w,eligible_for_confirmation=False,physical_fits=0,model_api_calls=0,new_rmb=0)
    else:
        features = AlternativeSentenceFeatures('pair_text','e5').fit(train)
        head = joblib.load(head_file)
        fam = joblib.load(fam_file)
        current = Pipeline([('features',features),('classifier',head)])
        base = PreferenceProbabilityVote('pair_text',[current,fam],np.array([0,1])).predict(query)
        np.testing.assert_array_equal(base,np.load(a.curve/'full/predictions.npy',allow_pickle=False))
        target = a.out/'challenger.joblib'
        begin, end = a.out/'fit_started.json', a.out/'fit_completed.json'
        if not end.exists():
            check(not target.exists() and not begin.exists(),'Ambiguous previously started fit')
            mat = features.transform([*train,*weak])
            labels = np.r_[y[fitting],np.ones(w,dtype=int)]
            weight = np.r_[np.ones(len(train)),np.full(w,0.5)]
            weight *= len(train)/weight.sum()
            check(np.isclose(weight.sum(),len(train)),'Regularization mass differs')
            save(begin,dict(attempts=1,observed=len(train),weak=w,mirror_rows=2*len(labels),weight_sum=2*len(train)))
            start = time.monotonic()
            challenger = clone(head).fit(vstack([mat,-mat]),np.r_[labels,1-labels],sample_weight=np.tile(weight,2))
            elapsed = time.monotonic()-start
            with target.open('xb') as stream:
                joblib.dump(challenger,stream)
            save(end,dict(fits=1,model=sha(target),start=sha(begin),seconds=elapsed))
        check(read(end)['model']==sha(target) and read(end)['start']==sha(begin),'New head identity differs')
        pipe = Pipeline([('features',features),('classifier',joblib.load(target))])
        vote = PreferenceProbabilityVote('pair_text',[pipe,fam],np.array([0,1]))
        new = vote.predict(query)
        sample = query[:24]
        np.testing.assert_allclose(vote.predict_proba(sample)[:,1],
                                   1-vote.predict_proba([(r,l) for l,r in sample])[:,1],atol=1e-10,rtol=0)
        output = a.out/'predictions.npz'
        if not output.exists():
            with output.open('xb') as stream:
                np.savez_compressed(stream,row_ids=evaluation,base=base,new=new)
        with np.load(output,allow_pickle=False) as saved:
            for k,v in [('row_ids',evaluation),('base',base),('new',new)]:
                np.testing.assert_array_equal(saved[k],v)
        frozen_save(a.out/'exports.json',dict(file=sha(output),baseline=fingerprint(base),new=fingerprint(new),
                    rows=fingerprint(evaluation),model=sha(target),swap_rows=len(sample)))
        before, after = base==y[evaluation], new==y[evaluation]
        stats, evidence = {}, {}
        for k,ix in {'A':slice(0,len(parts['A'])),'B':slice(len(parts['A']),None),'combined':slice(None)}.items():
            delta = after[ix].astype(int) - before[ix].astype(int)
            keys = groups[evaluation][ix]
            stats[k] = dict(rows=len(delta),baseline=int(before[ix].sum()),new=int(after[ix].sum()),
                            gained=int((delta>0).sum()),lost=int((delta<0).sum()),paired=paired(delta,keys))
            buckets = defaultdict(list)
            for g,old,changed in zip(keys,before[ix],after[ix],strict=True):
                buckets[g].append(int(old)+2*int(changed))
            evidence[k] = [sorted(buckets[g]) for g in sorted(buckets)]
        frozen_save(a.out/'grouped_correctness.json',evidence)
        nets = [stats[k]['new']-stats[k]['baseline'] for k in ('A','B')]
        result = dict(panels=stats,eligible_for_confirmation=min(nets)>=0 and sum(nets)>=8,
                      observed_rows=len(train),added_weak_pairs=w,physical_fits=1,model_api_calls=0,new_rmb=0,
                      designated_validation_or_test_opened=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',{name:sha(a.out/(name+'.json')) for name in
                ('protocol','training_graph','results','exports','grouped_correctness','fit_completed')
                if (a.out/(name+'.json')).exists()})
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace','training','first','second','e5','curve','out'):
        parser.add_argument('--'+key,type=lambda s:Path(s).resolve(),required=True)
    stage(parser.parse_args())
