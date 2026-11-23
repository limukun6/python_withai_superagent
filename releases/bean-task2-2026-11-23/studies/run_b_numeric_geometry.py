"""Compare two numeric methods to the original three-fold supervised Dry Bean control."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from superagent.ab_run import sha
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_numeric_geometry_1123_zh.md'
PARTS=('primary','confirmation_1','confirmation_2')
BASE='numeric_tabular_logistic_C1.0'
METHODS=('quantile_qda','distance_neighbors')
TRAIN='f181e6244d88b3785535484983e8a9fc56308ce6410217106c3f877b124bdcb6'


def build(name,n):
    impute=SimpleImputer(strategy='median',keep_empty_features=True)
    if name=='quantile_qda':
        transform=QuantileTransformer(n_quantiles=min(256,n),output_distribution='normal',subsample=None,random_state=42)
        return make_pipeline(impute,transform,QuadraticDiscriminantAnalysis(reg_param=.10))
    if name=='distance_neighbors':
        return make_pipeline(impute,StandardScaler(),KNeighborsClassifier(n_neighbors=11,weights='distance'))
    raise ValueError('Unknown numeric method')


def fit_once(folder,name,x,y,q):
    folder.mkdir(parents=True,exist_ok=True)
    file=folder/'classifier.joblib'
    begin,end=folder/'fit_started.json',folder/'fit_completed.json'
    if not end.exists():
        check(not begin.exists() and not file.exists(),'Ambiguous previous physical fit')
        save(begin,dict(method=name,rows=len(y),features=x.shape[1],attempts=1))
        head=build(name,len(y)).fit(x,y)
        with file.open('xb') as stream:
            joblib.dump(head,stream)
        save(end,dict(fits=1,start=sha(begin),model=sha(file)))
    receipt=read(end)
    check(receipt['start']==sha(begin) and receipt['model']==sha(file),'Fitted numeric model changed')
    model=joblib.load(file)
    labels=model.predict(q)
    prob=model.predict_proba(q)
    check(np.array_equal(model.classes_,np.unique(y)) and np.isfinite(prob).all() and np.all(prob>=0)
          and np.allclose(prob.sum(axis=1),1), 'Probability or label domain mismatch')
    np.testing.assert_array_equal(model.classes_[prob.argmax(axis=1)], labels)
    return labels


def reference(folder):
    manifest=read(folder/'run_manifest.json')
    selected=read(folder/'final_selection.json')
    check(manifest['task_archive_sha256']==TRAIN and manifest['seed']==707 and manifest['planner_mode']=='local'
          and manifest['scheduler']=='ab-fixed' and selected['family']==BASE and selected['training_rows']==8710,
          'Incorrect C1 reference')
    history=read(folder/'history.json')
    refs={}
    for part in PARTS:
        row,=[r for r in history if r['candidate']==BASE and r['split_id']==part and r['status']=='completed']
        file=folder/('trial_'+str(row['trial'])+'.joblib')
        payload=folder/(part+'.joblib')
        split=folder/(part+'_split.json')
        meta=read(split)
        check(row['evaluation_payload_sha256']==sha(payload)==meta['evaluation_payload_sha256'] and
              row['label_sha256']==meta['label_sha256'] and meta['search_duplicate_rows']==0, 'Wrong archived split')
        refs[part]=dict(trial=file.name,worker=sha(file),payload=sha(payload),split=sha(split),label=meta['label_sha256'])
    return refs


def paired_scores(base,new,y,g):
    before=base==y
    after=new==y
    delta=after.astype(int)-before.astype(int)
    by_class={}
    for name in np.unique(y):
        loc=y==name
        by_class[str(name)]=dict(rows=int(loc.sum()),baseline=int(before[loc].sum()),new=int(after[loc].sum()))
    return dict(rows=len(y),baseline=int(before.sum()),new=int(after.sum()),gained=int((delta>0).sum()),
                lost=int((delta<0).sum()),by_class=by_class,delta=paired(delta,g))


def measure(part,names,ref,folder,out,full_x,full_y,groups,seen):
    destination=out/part
    destination.mkdir(exist_ok=True)
    splits=read(folder/(part+'_split.json'))
    train,search=np.array(splits['train_indices']),np.array(splits['search_indices'])
    data=joblib.load(folder/(part+'.joblib'))
    for kind,idx in [('train',train),('search',search)]:
        np.testing.assert_array_equal(full_x[idx],data['X_'+kind])
        np.testing.assert_array_equal(full_y[idx],data['y_'+kind])
    y=full_y[search]
    check(fingerprint(search)==splits['row_ids_sha256'] and fingerprint(y)==ref['label'] and
          set(groups[train]).isdisjoint(groups[search]),'Feature groups cross the fold')
    keep=~np.isin(groups[search],list(seen))
    check(keep.any(),'No new evaluation groups')
    worker=joblib.load(folder/ref['trial'])
    control=worker['model'].predict(full_x[search])
    np.testing.assert_array_equal(control,worker['search_predictions'])
    check(fingerprint(control)==worker['prediction_record']['prediction_sha256'] and
          worker['prediction_record']['evaluation_payload_sha256']==ref['payload'],'Original control was not reproduced')
    arms={'control':control}
    for name in names:
        arms[name]=fit_once(destination/name,name,full_x[train],full_y[train],full_x[search])
    saved=destination/'predictions.npz'
    if not saved.exists():
        with saved.open('xb') as stream:
            np.savez_compressed(stream,**arms)
    with np.load(saved,allow_pickle=False) as archived:
        for key,v in arms.items():
            np.testing.assert_array_equal(archived[key],v)
    frozen_save(destination/'exports.json',dict(file=sha(saved),predictions={k:fingerprint(v) for k,v in arms.items()},
                full_order=fingerprint(search),scored_order=fingerprint(search[keep]),fitting_rows=len(train),
                full_search=len(search),scored_rows=int(keep.sum()),fits={k:sha(destination/k/'fit_completed.json') for k in names}))
    results={k:paired_scores(control[keep],arms[k][keep],y[keep],groups[search][keep]) for k in names}
    # Preserve group order for exact confidence-interval reproduction, without publishing labels or features.
    evidence={}
    for name in names:
        counts={}
        for group,old,new in zip(groups[search][keep],control[keep]==y[keep],arms[name][keep]==y[keep]):
            bucket=counts.setdefault(group,[])
            bucket.append(int(old) + 2*int(new))
        evidence[name]=[sorted(counts[k]) for k in sorted(counts)]
    frozen_save(destination/'grouped_correctness.json',evidence)
    return results,set(groups[search])


def run(a):
    sys.addaudithook(offline)
    refs=reference(a.control)
    full=a.control/'refit_data.joblib'
    original=joblib.load(full)
    x,y=original['X_train'],np.asarray(original['y_train'])
    check(x.shape==(8710,16) and len(np.unique(y))==7,'Unexpected numeric data')
    groups=sample_groups(SimpleNamespace(kind='numeric_tabular',X_train=x),'exact_rows')
    a.out.mkdir(parents=True,exist_ok=True)
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),c1_training=TRAIN,full=sha(full),
        baseline=BASE,methods=list(METHODS),splits=refs,fit_cap=4,model_api_cap=0))
    results={}
    first,seen=measure('primary',METHODS,refs['primary'],a.control,a.out,x,y,groups,set())
    results['primary']=first
    best=max(METHODS,key=lambda k:first[k]['new'])
    qualified=first[best]['new']-first[best]['baseline']>=4
    frozen_save(a.out/'screen_decision.json',dict(best=best,continue_confirmation=qualified,
                screen_exports=sha(a.out/'primary/exports.json'),scores={k:v['new'] for k,v in first.items()}))
    if qualified:
        for part in PARTS[1:]:
            res,inspected=measure(part,[best],refs[part],a.control,a.out,x,y,groups,seen)
            seen.update(inspected)
            results[part]=res
    gains=[results[p][best]['new']-results[p][best]['baseline'] for p in PARTS[1:] if p in results]
    fits=sum(len(r) for r in results.values())
    record=dict(panels=results,selected=best,eligible_for_outer=len(gains)==2 and min(gains)>=0 and sum(gains)>=8,
                physical_fits=fits,model_api_calls=0,new_rmb=0,official_or_external_holdout_opened=False)
    frozen_save(a.out/'results.json',record)
    frozen_save(a.out/'completion.json',dict(results=sha(a.out/'results.json'),protocol=sha(a.out/'protocol.json'),
        screen_decision=sha(a.out/'screen_decision.json'),exports={k:sha(a.out/k/'exports.json') for k in results},
        groups={k:sha(a.out/k/'grouped_correctness.json') for k in results},fits=fits,model_api_calls=0))
    print(json.dumps(dict(correct={p:{k:[v['baseline'],v['new'],v['rows']] for k,v in panel.items()}
                 for p,panel in results.items()},eligible=record['eligible_for_outer'],fits=fits)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','out'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args())
