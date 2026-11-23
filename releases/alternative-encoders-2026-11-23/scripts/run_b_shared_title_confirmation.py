"""Confirm the weighted and standard E5 heads on two group-disjoint panels in training data."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_weighting import SEEDS, bind, group_ids
from superagent.ab_run import sha
from superagent.ab_sentence import AlternativeSentenceFeatures
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import normalized_text

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_shared_title_confirmation_1123_zh.md'


def split(rows, first, second):
    groups = group_ids(rows)
    title_to_group = {normalized_text(t):g for pair,g in zip(rows,groups) for t in pair}
    seen = {title_to_group[normalized_text(t)] for p in (first,second)
            for row in joblib.load(p/'primary.joblib')['X_search'] for t in row}
    assignment = {}
    for g in set(groups):
        digest = hashlib.sha256(('title-weight-confirm-1123-v1:' + g).encode('ascii')).digest()
        bucket = int.from_bytes(digest,'big') % 5
        assignment[g] = ('A','B')[bucket] if g not in seen and bucket < 2 else 'train'
    ids = {s: np.array([i for i,g in enumerate(groups) if assignment[g] == s], dtype=int)
           for s in ('train','A','B')}
    unique = {s:set(groups[idx]) for s,idx in ids.items()}
    check(all(len(v)>0 for v in ids.values()) and len(rows)==sum(map(len,ids.values()))
          and not ((unique['A']|unique['B']) & seen) and unique['A'].isdisjoint(unique['B']) and
          unique['train'].isdisjoint(unique['A']|unique['B']), 'Confirmation groups or earlier evaluations overlap')
    return groups, ids, len(seen)


def train_pair(out, rows, labels, ids, groups):
    train = ids['train']
    both = np.concatenate([ids['A'],ids['B']])
    train_rows = [rows[i] for i in train]
    search_rows = [rows[i] for i in both]
    fe = AlternativeSentenceFeatures('pair_text','e5').fit(train_rows)
    x = fe.transform(train_rows)
    query = fe.transform(search_rows)
    x = vstack((x,-x))
    y = np.r_[labels[train],1-labels[train]]
    counts = Counter(groups[train])
    w = np.array([counts[g]**-.5 for g in groups[train]])
    w /= w.mean()
    seals = {}
    for arm in ('regular','weighted'):
        folder = out/arm
        folder.mkdir(exist_ok=True)
        target = folder/'classifier.joblib'
        if not target.exists():
            save(folder/'fit_started.json', dict(model='LogisticRegression',arm=arm,training_rows=len(train),attempts=1))
            head = LogisticRegression(C=.3,max_iter=2000,random_state=42)
            head.fit(x, y, sample_weight=np.tile(w,2) if arm=='weighted' else None)
            with target.open('xb') as f:
                joblib.dump(head,f)
            save(folder/'fit_completed.json',dict(model=sha(target),start=sha(folder/'fit_started.json'),fits=1))
        receipt = read(folder/'fit_completed.json')
        check(receipt['model']==sha(target) and receipt['start']==sha(folder/'fit_started.json') and receipt['fits']==1,
              'Missing fit evidence; no automatic refit')
        pred = joblib.load(target).predict(query)
        f = folder/'predictions.npy'
        if not f.exists():
            with f.open('xb') as stream:
                np.save(stream,pred)
        np.testing.assert_array_equal(np.load(f,allow_pickle=False),pred)
        seals[arm] = dict(file=sha(f), predictions=fingerprint(pred), fit=sha(folder/'fit_completed.json'))
    frozen_save(out/'exports.json',seals)
    return both


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    refs = read(a.screen/'protocol.json')['references']
    check(read(a.screen/'results.json')['eligible_for_confirmation'] and
          read(a.screen/'completion.json')['results']==sha(a.screen/'results.json') and sha(a.training)==INPUT_TRAIN,
          'Screening prerequisite or anonymous input differs')
    for s,folder in zip(SEEDS,(a.first,a.second)):
        check(bind(folder,s)==refs[str(s)],'Earlier E5 control differs')
    task = load_development_task(a.training)
    rows = task.X_train
    check(len(rows)==6608 and task.X_val is None,'Wrong training packet')
    groups, ids, n_seen = split(rows,a.first,a.second)
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN), runner=sha(Path(__file__)), screening=sha(a.screen/'completion.json'),
        train_packet=INPUT_TRAIN, seed_sources=refs, earlier_search_groups=n_seen,
        partitions={k:dict(rows=len(v),groups=len(set(groups[v])),ids=fingerprint(v)) for k,v in ids.items()},
        shared_groups_between_panels=0, shared_groups_with_prior_scoring=0, fits_cap=2, model_api_cap=0))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    y = np.asarray(task.y_train)
    both = train_pair(a.out,rows,y,ids,groups)
    saved = read(a.out/'exports.json')
    hits = {}
    for arm in ('regular','weighted'):
        f = a.out/arm/'predictions.npy'
        check(sha(f)==saved[arm]['file'],'Prediction file changed')
        hits[arm] = np.load(f,allow_pickle=False) == y[both]
    delta = hits['weighted'].astype(int)-hits['regular'].astype(int)
    stats = {}
    for panel, ix in [('A',slice(0,len(ids['A']))),('B',slice(len(ids['A']),None)),('combined',slice(None))]:
        d = delta[ix]
        stats[panel] = dict(rows=len(d),regular=int(hits['regular'][ix].sum()),weighted=int(hits['weighted'][ix].sum()),
                            gained=int((d>0).sum()),lost=int((d<0).sum()),delta=paired(d,groups[both][ix]))
    nets = [stats[k]['delta']['net_correct'] for k in ('A','B')]
    result = dict(panels=stats,eligible_for_agent=min(nets)>=0 and sum(nets)>=8,physical_fits=2,model_api_calls=0,new_rmb=0,
                  formal_test_opened=False,course_designated_opened=False,development_reserve_opened=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
                exports=sha(a.out/'exports.json'),fits=2,model_api_calls=0))
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','training','first','second','screen','out'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args())
