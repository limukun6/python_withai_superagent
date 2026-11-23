"""Fill the two missing cells in the archived two-by-two Dry Bean kernel comparison."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.base import clone

from scripts.ab_acceptance_guard.sitecustomize import offline
from scripts.b_frozen_head_frontier import paired
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from superagent.ab_run import sha
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/batches/B_numeric_two_generation_plan_1123_zh.md'


def fit(folder, spec, x, y):
    path, begin, end = folder/'classifier.joblib',folder/'fit_started.json',folder/'fit_completed.json'
    if not end.exists():
        check(not path.exists() and not begin.exists(),'Ambiguous prior attempt')
        save(begin,dict(rows=len(y),attempts=1,training_labels=fingerprint(y)))
        head=clone(spec).fit(x,y)
        with path.open('xb') as f:
            joblib.dump(head,f)
        save(end,dict(fits=1,started=sha(begin),model=sha(path)))
    check(read(end)==dict(fits=1,started=sha(begin),model=sha(path)),'Invalid fitted pipeline')
    return joblib.load(path)


def read_labels(csv_file):
    with csv_file.open(newline='',encoding='utf-8') as f:
        records=list(csv.DictReader(f))
    check([int(r['row_id']) for r in records]==list(range(len(records))),'Original export order differs')
    return np.asarray([r['prediction'] for r in records])


def run(a):
    sys.addaudithook(offline)
    a.out.mkdir(parents=True,exist_ok=True)
    legacy=a.bn/'full/run'
    current=a.a94/'agent'
    original=[legacy/'selected_model.joblib',current/'selected_model.joblib']
    check([sha(f) for f in original]==['04d21d61d49d95ea5f61bb8cb536c6cea602ab2e023a4e1ab8173f6cccd90620',
          '9ba6894303b5830192261fec79e7c1422b28e02c1251fdc4ae8fe63f4acac50c'],'Different baseline models')
    old,new=[joblib.load(f)['model'] for f in original]
    check(old.named_steps['features'].gamma=='scale' and old.named_steps['features'].random_state==42
          and 'kernel' in new.named_steps,'Unrecognized Fourier maps')
    inputs={'bn01':(legacy,a.bn_features,a.bn/'full/export_0/rows.csv'),
            'a94':(current,a.a94_evaluation/'features.zip',a.a94/'new_first/rows.csv')}
    fingerprints={k:dict(train=sha(p/'refit_data.joblib'),features=sha(q),original=sha(ref))
                  for k,(p,q,ref) in inputs.items()}
    check(fingerprints['a94']['features']=='7ce0e8007b92dcd5ebdc58c8cb49d75a1f86d95784ece90194ade0ab3c35abd4'
          and fingerprints['bn01']['features']==read(a.bn/'protocol.json')['train_and_features']['features.zip']
          and sha(a.source)=='c290b2e43756bdcab095e7f82c0131977afa7b18ed08b679a71bc84edf7d0638',
          'Unexpected historical input package')
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN), runner=sha(Path(__file__)), files=fingerprints,
                models=[sha(f) for f in original], ids=sha(a.bn_ids), fit_cap=2,model_api_cap=0))
    local_x, receipts={},{}
    for key,(p,q,ref) in inputs.items():
        folder=a.out/key
        folder.mkdir(exist_ok=True)
        train=joblib.load(p/'refit_data.joblib')
        x,y=train['X_train'],train['y_train']
        check(x.shape==(8714 if key=='bn01' else 8710,16),'Unexpected fitting scope')
        query=load_prediction_features(q,'numeric_tabular',read(p/'task_spec.json'))
        local_x[key]=query
        base,newer=(old,fit(folder,new,x,y)) if key=='bn01' else (fit(folder,old,x,y),new)
        old_pred,new_pred=base.predict(query),newer.predict(query)
        np.testing.assert_array_equal(old_pred if key=='bn01' else new_pred,read_labels(ref))
        f=folder/'predictions.npz'
        if not f.exists():
            with f.open('xb') as stream:
                np.savez_compressed(stream,old=old_pred,new=new_pred)
        with np.load(f,allow_pickle=False) as stored:
            np.testing.assert_array_equal(old_pred,stored['old'])
            np.testing.assert_array_equal(new_pred,stored['new'])
        receipts[key]=dict(file=sha(f),fit=sha(folder/'fit_completed.json'),rows=len(query))
    frozen_save(a.out/'exports.json',receipts)
    initial=load_training_task(a.source)
    bn_ids=json.loads(gzip.decompress(a.bn_ids.read_bytes()))['outer']['held']
    np.testing.assert_array_equal(initial.X_train[bn_ids],local_x['bn01'])
    check(sha(a.a94_evaluation/'holdout.joblib')=='c1ddc6ba8b78ff419b53d97063e83a824d8735ef1198018dc8b3b0145199f0d3',
          'Unregistered A94 evaluation source')
    held=joblib.load(a.a94_evaluation/'holdout.joblib')
    np.testing.assert_array_equal(held['X'],local_x['a94'])
    stats,grouped={},{}
    for key,truth in {'bn01':initial.y_train[bn_ids],'a94':held['y']}.items():
        path=a.out/key/'predictions.npz'
        check(sha(path)==receipts[key]['file'],'Changed export')
        with np.load(path,allow_pickle=False) as f:
            before,after=f['old']==truth,f['new']==truth
        delta=after.astype(int)-before.astype(int)
        groups=sample_groups(SimpleNamespace(kind='numeric_tabular',X_train=local_x[key]))
        counts={}
        for g,b,n in zip(groups,before,after,strict=True):
            counts.setdefault(g,[]).append(int(b)+2*int(n))
        grouped[key]=[sorted(counts[g]) for g in sorted(counts)]
        stats[key]=dict(rows=len(truth),old_kernel=int(before.sum()),new_kernel=int(after.sum()),
                        gained=int((delta>0).sum()),lost=int((delta<0).sum()),delta=paired(delta,groups))
    check(stats['bn01']['old_kernel']==2008 and stats['a94']['new_kernel']==2005,'Legacy score differs')
    # Count shared full feature vectors, independent of source-specific numeric row identifiers.
    groups={k:set(sample_groups(SimpleNamespace(kind='numeric_tabular',X_train=x))) for k,x in local_x.items()}
    same_groups=len(groups['bn01'] & groups['a94'])
    frozen_save(a.out/'grouped_correctness.json',grouped)
    result=dict(panels=stats,overlapping_feature_groups=same_groups,
                improved_both_by_three=all(v['new_kernel']-v['old_kernel']>=3 for v in stats.values()),
                physical_fits=2,model_api_calls=0,new_rmb=0)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',{k:sha(a.out/(k+'.json')) for k in
                ('protocol','exports','results','grouped_correctness')})
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('bn','bn-features','bn-ids','a94','a94-evaluation','source','out'):
        parser.add_argument('--'+key,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
