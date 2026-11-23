"""Freeze two native runs before scoring an anonymous numeric ROC-AUC development task."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, command, read
from superagent.ab_run import sha, sources
from superagent.numeric_kernel import NAME
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, sample_groups

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/batches/B_numeric_auc_transfer_1123_zh.md'


def archive(path, split, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    manifest = dict(schema_version=1,objective='classification',modality='tabular',metric='roc_auc',
                    positive_label=1,splits={split:dict(format='npz',path='rows.npz')})
    packet = io.BytesIO()
    with zipfile.ZipFile(packet, 'w') as z:
        for name, data in [('manifest.json',json.dumps(manifest).encode()),('rows.npz',buffer.getvalue())]:
            z.writestr(zipfile.ZipInfo(name,date_time=(2020,1,1,0,0,0)), data)
    if path.exists():
        check(path.read_bytes()==packet.getvalue(),'Data bytes differ')
    else:
        with path.open('xb') as f:
            f.write(packet.getvalue())


def read_scores(path, count):
    with path.open(newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    meta = read(path.parent/'prediction_manifest.json')
    order = meta['score_class_order']
    check(order==[0,1] and meta['test_labels_read'] is False and meta['prediction_count']==count and
          [int(r['row_id']) for r in rows]==list(range(count)), 'Unexpected class, row order or labels')
    matrix = np.array([[float(r['score_0']), float(r['score_1'])] for r in rows])
    proposed = np.array([int(r['prediction']) for r in rows])
    check(np.isfinite(matrix).all() and np.all((0<=matrix)&(matrix<=1))
          and np.allclose(matrix.sum(axis=1),1), 'Invalid class probabilities')
    return matrix[:,order.index(1)], proposed


def pair_ranks(scores, labels):
    margin = scores[labels==1,None] - scores[labels==0][None,:]
    return (margin>0).astype(float) + 0.5*(margin==0)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    x,y = load_breast_cancer(return_X_y=True)
    check(x.shape==(569,30) and set(y)=={0,1},'Unexpected installed dataset')
    task = SimpleNamespace(kind='numeric_tabular',X_train=x,y_train=y)
    groups = sample_groups(task)
    train,held,_ = grouped_partition(task,11323)
    check(set(groups[train]).isdisjoint(groups[held]) and len(set(groups[held]))==len(held),
          'Dependencies in held observations')
    training, query = out/'anonymous.zip',out/'features.zip'
    archive(training,'train',X=x[train],y=y[train])
    archive(query,'test',X=x[held])
    frozen_save(out/'protocol.json',dict(plan=sha(PLAN), runner=sha(Path(__file__)), source=sources(),
                training=sha(training), features=sha(query), train_ids=fingerprint(train),
                held_ids=fingerprint(held), train_rows=len(train), held_rows=len(held), positive=1,
                fit_cap_each=11, model_api_cap=0))
    env=dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/ab_acceptance_guard',ROOT))),
             AB_B1_OFFLINE='1', AB_B1_PER_WORKER_FIT_LIMIT='1', OPENBLAS_NUM_THREADS='2',
             OMP_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    models, menus = {}, {}
    for arm in ('standard','expanded'):
        directory = out/arm
        counter = out/(arm+'.jsonl')
        if not (directory/'run_outcome.json').exists():
            check(not directory.exists() and not counter.exists(), 'Incomplete earlier run')
            args=['run',str(training),'--planner','local','--scheduler','ab-v1','--portfolio','ab-v1',
                  '--fit-limit','11','--time-limit','1200','--trial-timeout','300','--seed','707','--run-dir',str(directory)]
            if arm=='expanded':
                args.append('--numeric-kernel')
            status,_ = command(args,out/(arm+'.log'),dict(env,AB_B1_FIT_LOG=str(counter)),1260)
            check(status==0, 'Autonomous path failed for '+arm)
        manifest = read(directory/'run_manifest.json')
        outcome = read(directory/'run_outcome.json')
        fits = read(directory/'classifier_fits.json')
        selected = read(directory/'final_selection.json')
        actions = read(directory/'actions.json')
        actual = [json.loads(s)['candidate'] for s in counter.read_text().splitlines()]
        check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and
              outcome['new_api_requests']==0 and not outcome['formal_test_accessed'] and 0<len(actual)<=11
              and len(actual)==fits['completed_classifier_fits']==fits['attempted_classifier_fits']
              and actual==[w['candidate'] for w in fits['workers']] and manifest['source_files']==sources()
              and selected['model_sha256']==sha(directory/'selected_model.joblib'), 'Unverified run outcome')
        menus[arm] = manifest['capabilities']
        models[arm] = dict(fits=len(actual),model=selected['model_sha256'],family=selected['family'],
                           actions=[(v['action'],v['candidate_id'],v['split_id']) for v in actions],
                           splits=[sha(directory/(k+'.joblib')) for k in ('primary','confirmation_1','confirmation_2')],
                           physical=sha(counter))
    check(models['standard']['splits']==models['expanded']['splits'] and
          NAME in menus['expanded'] and {k:v for k,v in menus['expanded'].items() if k!=NAME}==menus['standard'],
          'Mismatched task or methods')
    # JSON normalizes action tuples to lists for a deterministic resumption check.
    frozen_save(out/'models_frozen.json',json.loads(json.dumps(models)))
    exports = {}
    for arm in models:
        for i in (0,1):
            key = f'{arm}_{i}'
            csv_path = out/key/'rows.csv'
            if not (out/key/'prediction_manifest.json').exists():
                status,_ = command(['predict',str(query),'--run-dir',str(out/arm),'--output',str(csv_path)],
                                   out/(key+'.log'),env,300)
                check(status==0,'Independent export failed')
            exports[key] = sha(csv_path)
        check(exports[arm+'_0']==exports[arm+'_1'],'Two independent exports differ')
    frozen_save(out/'exports.json',dict(models=sha(out/'models_frozen.json'),csv=exports))
    truth = y[held]
    outcomes, ranks = {}, {}
    for arm in models:
        scores, pred = read_scores(out/(arm+'_0')/'rows.csv',len(held))
        ranks[arm] = pair_ranks(scores,truth)
        auc = float(roc_auc_score(truth,scores))
        check(np.isclose(auc,ranks[arm].mean()),'Probability ranking does not reproduce AUC')
        outcomes[arm] = dict(auc=auc, correct=int(np.sum(pred==truth)), selected=models[arm]['family'],
                             fits=models[arm]['fits'])
    delta = ranks['expanded']-ranks['standard']
    rng = np.random.default_rng(11323)
    changes = []
    for _ in range(5000):
        pos = rng.integers(0,delta.shape[0],size=delta.shape[0])
        neg = rng.integers(0,delta.shape[1],size=delta.shape[1])
        changes.append(delta[np.ix_(pos,neg)].mean())
    result = dict(arms=outcomes, rows=len(held), class_counts=np.bincount(truth).tolist(),
                  held_auc_difference=float(delta.mean()), paired_stratified_95=np.quantile(changes,[.025,.975]).tolist(),
                  better_rankings=int((delta>0).sum()), worse_rankings=int((delta<0).sum()),
                  ties_or_unchanged=int((delta==0).sum()), actual_fits=sum(v['fits'] for v in models.values()),
                  model_api_calls=0, new_rmb=0)
    frozen_save(out/'results.json',result)
    frozen_save(out/'completion.json',{k:sha(out/(k+'.json')) for k in ('protocol','models_frozen','exports','results')})
    print(json.dumps(result))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
