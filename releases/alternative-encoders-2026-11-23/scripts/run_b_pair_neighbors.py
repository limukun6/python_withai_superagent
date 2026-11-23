"""Evaluate a signed nearest-neighbor vote over text-pair differences, using training labels only."""
import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.b_pair_neighbors import PairNeighbors
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_weighting import SEEDS, bind, group_ids
from superagent.ab_run import sha
from superagent.ab_sentence import encode_alternative
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import normalized_text

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_pair_neighbors_1123_zh.md'


def directions(rows):
    titles=sorted({t for r in rows for t in r})
    vectors,tokens=encode_alternative(titles,'e5')
    pos={s:i for i,s in enumerate(titles)}
    deltas=np.asarray([vectors[pos[l]]-vectors[pos[r]] for l,r in rows],dtype=np.float32)
    norm=np.linalg.norm(deltas,axis=1,keepdims=True)
    unit=np.divide(deltas,norm,out=np.zeros_like(deltas),where=norm>0)
    return unit,dict(titles=len(titles),local_encoder_input_tokens=tokens,zero_directions=int((norm==0).sum()))


def index_once(folder, x, y, groups):
    file=folder/'index.joblib'
    if not file.exists():
        save(folder/'fit_started.json',dict(supervised_index='PairNeighbors',attempts=1,rows=len(y),groups=len(set(groups))))
        index=PairNeighbors().fit(x,y,groups)
        with file.open('xb') as f:
            joblib.dump(index,f)
        save(folder/'fit_completed.json',dict(index=sha(file),start=sha(folder/'fit_started.json'),fits=1))
    receipt=read(folder/'fit_completed.json')
    check(receipt['fits']==1 and receipt['index']==sha(file) and receipt['start']==sha(folder/'fit_started.json'),'Index fit incomplete')
    return joblib.load(file)


def hard(prob, left_first):
    return ((prob>.5)|((prob==.5)&left_first)).astype(int)


def emit(folder, data, worker, ref, train, query, rows, vectors):
    folder.mkdir(exist_ok=True)
    train_rows=[rows[i] for i in train]; query_rows=[rows[i] for i in query]
    np.testing.assert_array_equal(train_rows,data['X_train'])
    np.testing.assert_array_equal(query_rows,data['X_search'])
    titles={normalized_text(t) for row in train_rows for t in row}
    check(not any(normalized_text(t) in titles for row in query_rows for t in row),'Title crosses split')
    p=np.asarray(worker['search_scores'])[:,list(worker['score_classes']).index(1)]
    record=worker['prediction_record']
    check(record['evaluation_payload_sha256']==ref['payload'] and record['label_sha256']==ref['labels'],'E5 score source changed')
    left_first=np.array([l<r for l,r in query_rows])
    base=hard(p,left_first)
    np.testing.assert_array_equal(base,worker['search_predictions'])
    check(fingerprint(base)==record['prediction_sha256'],'E5 predictions do not reproduce')
    index=index_once(folder,vectors[train],np.asarray(data['y_train']),group_ids(train_rows))
    neighbors=index.probability(vectors[query])
    reverse=index.probability(-vectors[query])
    np.testing.assert_allclose(reverse,1-neighbors,atol=1e-12,rtol=0)
    blend=.8*p+.2*neighbors
    check(np.all(np.isfinite(neighbors)) and np.all((neighbors>=0)&(neighbors<=1)),'Invalid neighbor probabilities')
    predictions={'base':base,'neighbors':hard(neighbors,left_first),'blend':hard(blend,left_first)}
    out=folder/'predictions.npz'
    if not out.exists():
        with out.open('xb') as f:
            np.savez_compressed(f,**predictions)
    with np.load(out) as store:
        for key,pred in predictions.items(): np.testing.assert_array_equal(store[key],pred)
    frozen_save(folder/'export.json',dict(predictions=sha(out),ids=fingerprint(query),swapped_probability_checks=len(query),
            max_swap_error=float(np.abs(reverse+neighbors-1).max()),fit=sha(folder/'fit_completed.json')))


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    check(sha(a.training)==INPUT_TRAIN,'Unexpected training scope')
    task=load_development_task(a.training)
    rows=task.X_train
    assay=dict(zip(SEEDS,(a.first,a.second)))
    refs={str(s):bind(d,s) for s,d in assay.items()}
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),
        estimator=sha(Path(__file__).with_name('b_pair_neighbors.py')),training=INPUT_TRAIN,
        sources=refs,neighbors=21,neighbor_fraction=.2,supervised_fit_cap=2,model_api_cap=0))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    target=a.out/'directions.npy'
    if not target.exists():
        v,summary=directions(rows)
        with target.open('xb') as f:np.save(f,v)
        save(a.out/'encoding.json',dict(**summary,file=sha(target),rows=len(rows)))
    check(sha(target)==read(a.out/'encoding.json')['file'],'Frozen directions changed')
    v=np.load(target,allow_pickle=False)
    panels={}
    for seed,src in assay.items():
        ref=refs[str(seed)]
        split=read(src/'primary_split.json')
        ids=np.array(split['search_indices']);train=np.array(split['train_indices'])
        data=joblib.load(src/'primary.joblib')
        check(fingerprint(ids)==split['row_ids_sha256'],'Unmatched row ordering')
        emit(a.out/str(seed),data,joblib.load(src/ref['trial']),ref,train,ids,rows,v)
        panels[str(seed)]=(ids,ref)
    result={}
    for key,(ids,ref) in panels.items():
        y=np.asarray(task.y_train)[ids]
        check(fingerprint(y)==ref['labels'],'Scoring target changed')
        groups=group_ids([rows[i] for i in ids])
        with np.load(a.out/key/'predictions.npz') as records:
            hits={name:records[name]==y for name in records.files}
        d=hits['blend'].astype(int)-hits['base'].astype(int)
        result[key]=dict(rows=len(ids),correct={k:int(x.sum()) for k,x in hits.items()},gained=int((d>0).sum()),
                         lost=int((d<0).sum()),delta=paired(d,groups))
    gains=[r['delta']['net_correct'] for r in result.values()]
    record=dict(panels=result,eligible_for_confirmation=min(gains)>=0 and max(gains)>=5,actual_fits=2,
                model_api_calls=0,new_rmb=0,course_validation_or_test=False)
    frozen_save(a.out/'results.json',record)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
                exports={k:sha(a.out/k/'export.json') for k in result},fits=2,model_api_calls=0))
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('workspace','training','first','second','out'):
        p.add_argument('--'+k,type=lambda x:Path(x).resolve(),required=True)
    run(p.parse_args())
