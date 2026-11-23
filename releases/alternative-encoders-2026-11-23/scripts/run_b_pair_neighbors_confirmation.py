"""Score the frozen nearest-pair rule on two disjoint confirmation panels with one new supervised index."""
import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from scripts.run_b_pair_neighbors import hard, index_once
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_confirmation import split
from superagent.ab_run import sha
from superagent.ab_sentence import AlternativeSentenceFeatures
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_pair_neighbors_confirmation_1123_zh.md'


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    check(sha(a.training)==INPUT_TRAIN and read(a.screen/'results.json')['eligible_for_confirmation'] and
          read(a.screen/'completion.json')['results']==sha(a.screen/'results.json'),'Prior screen or input differs')
    task=load_development_task(a.training)
    rows,y=task.X_train,np.asarray(task.y_train)
    groups,ids,_=split(rows,a.first,a.second)
    control=read(a.control/'protocol.json')
    for k,v in ids.items():
        check(fingerprint(v)==control['partitions'][k]['ids'],'Earlier confirmation partition differs')
    old_head=a.control/'regular/classifier.joblib'
    check(read(a.control/'regular/fit_completed.json')['model']==sha(old_head) and
          read(a.control/'completion.json')['results']==sha(a.control/'results.json'),'Regular control changed')
    vec=a.screen/'directions.npy'
    check(sha(vec)==read(a.screen/'encoding.json')['file'],'Frozen representation differs')
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),estimator=sha(Path(__file__).with_name('b_pair_neighbors.py')),
        screen=sha(a.screen/'completion.json'),control=sha(a.control/'completion.json'),training=INPUT_TRAIN,
        partitions=control['partitions'],classifier=sha(old_head),directions=sha(vec),index_fits_cap=1,model_api_cap=0))
    env=offline_environment(a.workspace)
    os.environ.update(env)
    sys.addaudithook(offline)
    x=np.load(vec,allow_pickle=False)
    train=ids['train']; both=np.r_[ids['A'],ids['B']]
    out_index=a.out/'index'
    out_index.mkdir(exist_ok=True)
    index=index_once(out_index,x[train],y[train],groups[train])
    near=index.probability(x[both])
    np.testing.assert_allclose(index.probability(-x[both]),1-near,atol=1e-12,rtol=0)
    query_rows=[rows[i] for i in both]
    fe=AlternativeSentenceFeatures('pair_text','e5').fit([rows[i] for i in train])
    head=joblib.load(old_head)
    prob=head.predict_proba(fe.transform(query_rows))[:,head.classes_.tolist().index(1)]
    first=np.array([l<r for l,r in query_rows])
    base=hard(prob,first)
    previous=a.control/'regular/predictions.npy'
    check(sha(previous)==read(a.control/'exports.json')['regular']['file'],'Earlier control export differs')
    np.testing.assert_array_equal(base,np.load(previous,allow_pickle=False))
    pred={'base':base,'neighbors':hard(near,first),'blend':hard(.8*prob+.2*near,first)}
    file=a.out/'predictions.npz'
    if not file.exists():
        with file.open('xb') as f: np.savez_compressed(f,**pred)
    with np.load(file) as sealed:
        for name,h in pred.items():np.testing.assert_array_equal(sealed[name],h)
    frozen_save(a.out/'exports.json',dict(file=sha(file),ids=fingerprint(both),prior_regular=sha(previous),fit=sha(out_index/'fit_completed.json')))
    hits={name:h==y[both] for name,h in pred.items()}
    d=hits['blend'].astype(int)-hits['base'].astype(int)
    summary={}
    for name,ix in [('A',slice(0,len(ids['A']))),('B',slice(len(ids['A']),None)),('combined',slice(None))]:
        summary[name]=dict(rows=len(d[ix]),correct={k:int(v[ix].sum()) for k,v in hits.items()},
              gained=int((d[ix]>0).sum()),lost=int((d[ix]<0).sum()),delta=paired(d[ix],groups[both][ix]))
    gains=[summary[k]['delta']['net_correct'] for k in ('A','B')]
    result=dict(panels=summary,eligible_for_joint=min(gains)>=0 and sum(gains)>=8,actual_fits=1,
                model_api_calls=0,new_rmb=0,course_validation_or_test=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
        exports=sha(a.out/'exports.json'),fits=1,model_api_calls=0))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('workspace','training','first','second','screen','control','out'):
        p.add_argument('--'+k,type=lambda x:Path(x).resolve(),required=True)
    run(p.parse_args())
