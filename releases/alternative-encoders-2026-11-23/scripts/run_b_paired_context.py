"""Screen one contextual representation against the exact same fitted lexical and E5 preprocessing."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from sklearn.base import clone

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_contextual_pair_features import contextual_differences, norm_scale
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check,read,save
from scripts.run_b_quote_transfer import INPUT_TRAIN
from scripts.run_b_shared_title_weighting import SEEDS,bind,group_ids
from superagent.ab_run import sha
from superagent.ab_sentence import encode_alternative
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_paired_context_1123_zh.md'


def technical_probe():
    raw=[('the first wording','a different alternative'),('a repeated sentence','a repeated sentence'),
         ('word '*600,'other '*700)]
    a,usage=contextual_differences(raw)
    b,reverse=contextual_differences([(r,l) for l,r in raw])
    np.testing.assert_allclose(a,-b,atol=1e-6,rtol=0)
    check(float(np.linalg.norm(a[0]))>0 and np.all(a[1]==0) and usage['max_input_tokens']==reverse['max_input_tokens']==512,
          'Segment, symmetry or truncation probe failed')
    return dict(max_length_tested=512,identical_feature_exact_zero=True,swapped_separate_forward=bool(np.array_equal(a,-b)),
                max_swap_error=float(np.max(np.abs(a+b))),synthetic_probe_tokens=usage['local_input_tokens']+reverse['local_input_tokens'])


def cached_context(out,rows):
    file=out/'context.npy'
    receipt=out/'encoding.json'
    if not file.exists():
        probe=technical_probe()
        start=time.monotonic()
        values,usage=contextual_differences(rows)
        with file.open('xb') as f:np.save(f,values)
        save(receipt,dict(sha256=sha(file),rows=len(rows),seconds=time.monotonic()-start,usage=usage,probe=probe))
    check(read(receipt)['sha256']==sha(file),'Joint representations changed')
    return np.load(file,allow_pickle=False)


def fit_and_predict(folder, original, train_rows, search_rows, train_y, ctx_train, ctx_search):
    folder.mkdir(exist_ok=True)
    features,head=original.steps[0][1],original.steps[1][1]
    check(features.variant=='e5' and head.C==.3,'Unexpected reference E5 pipeline')
    before=features.usage_['input_tokens']
    x=features.transform(train_rows);q=features.transform(search_rows)
    dim=ctx_train.shape[1]
    base_x=x[:,-dim:].toarray();base_q=q[:,-dim:].toarray()
    # A direct local E5 call binds the final feature columns to the separate title difference.
    probe,probe_tokens=encode_alternative([s for pair in train_rows[:3] for s in pair],'e5')
    np.testing.assert_allclose(base_x[:3],probe[::2]-probe[1::2],atol=2e-5,rtol=0)
    residual=ctx_train-base_x
    scale=norm_scale(base_x,residual)
    dst=folder/'head.joblib'
    if not dst.exists():
        mat=hstack([x,csr_matrix(scale*residual)],format='csr')
        save(folder/'fit_started.json',dict(attempts=1,train_rows=len(train_rows),estimator=type(head).__name__,
             train_only_scale=scale,augmented_rows=2*len(train_rows)))
        fitted=clone(head).fit(vstack((mat,-mat)),np.r_[train_y,1-np.asarray(train_y)])
        with dst.open('xb') as f:joblib.dump(fitted,f)
        save(folder/'fit_completed.json',dict(model=sha(dst),start=sha(folder/'fit_started.json'),fits=1))
    end=read(folder/'fit_completed.json')
    check(end['model']==sha(dst) and end['start']==sha(folder/'fit_started.json') and
          scale==read(folder/'fit_started.json')['train_only_scale'],'Physical fit evidence or scale differs')
    fq=hstack([q,csr_matrix(scale*(ctx_search-base_q))],format='csr')
    fitted=joblib.load(dst)
    return head.predict(q), fitted.predict(fq), dict(scale=scale,fit=sha(folder/'fit_completed.json'),
          base_feature_forward_tokens=features.usage_['input_tokens']-before,direct_probe_tokens=probe_tokens)


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    check(sha(a.training)==INPUT_TRAIN,'Unexpected training packet')
    task=load_development_task(a.training)
    rows=task.X_train
    dirs=dict(zip(SEEDS,(a.first,a.second)))
    refs={str(s):bind(d,s) for s,d in dirs.items()}
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),training=INPUT_TRAIN,
        representation=sha(Path(__file__).with_name('b_contextual_pair_features.py')),sources=refs,fit_cap=2,model_api_cap=0))
    os.environ.update(offline_environment(a.workspace))
    sys.addaudithook(offline)
    context=cached_context(a.out,rows)
    panels={}
    for seed,src in dirs.items():
        key=str(seed);p=a.out/key
        r=refs[key]
        s=read(src/'primary_split.json')
        train,search=np.array(s['train_indices']),np.array(s['search_indices'])
        data=joblib.load(src/'primary.joblib')
        worker=joblib.load(src/r['trial'])
        check(s['search_rows_sharing_text']==0 and fingerprint(search)==s['row_ids_sha256']
              and worker['prediction_record']['evaluation_payload_sha256']==r['payload'],'Different split or worker')
        np.testing.assert_array_equal([rows[i] for i in train],data['X_train'])
        np.testing.assert_array_equal([rows[i] for i in search],data['X_search'])
        base,new,meta=fit_and_predict(p,worker['model'],data['X_train'],data['X_search'],data['y_train'],context[train],context[search])
        np.testing.assert_array_equal(base,worker['search_predictions'])
        file=p/'predictions.npz'
        if not file.exists():
            with file.open('xb') as f:np.savez_compressed(f,base=base,context=new)
        with np.load(file) as saved:
            np.testing.assert_array_equal(saved['base'],base);np.testing.assert_array_equal(saved['context'],new)
        frozen_save(p/'exports.json',dict(file=sha(file),prior=fingerprint(base),new=fingerprint(new),**meta))
        panels[key]=search
    result={}
    for k,ids in panels.items():
        y=np.asarray(task.y_train)[ids]
        check(fingerprint(y)==refs[k]['labels'],'Scoring rows differ')
        with np.load(a.out/k/'predictions.npz') as saved:
            base,new=saved['base']==y,saved['context']==y
        delta=new.astype(int)-base.astype(int)
        result[k]=dict(rows=len(y),baseline=int(base.sum()),context=int(new.sum()),gained=int((delta>0).sum()),
            lost=int((delta<0).sum()),delta=paired(delta,group_ids([rows[i] for i in ids])))
    gains=[v['delta']['net_correct'] for v in result.values()]
    record=dict(panels=result,eligible_for_confirmation=min(gains)>=0 and max(gains)>=5,physical_fits=2,
                model_api_calls=0,new_rmb=0,course_validation_or_test=False)
    frozen_save(a.out/'results.json',record)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
        exports={k:sha(a.out/k/'exports.json') for k in result},fits=2,model_api_calls=0))
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('workspace','training','first','second','out'):
        p.add_argument('--'+k,type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args())
