"""Quantify fixed-prediction score differences by resampling paired, whole input groups."""
import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as rec
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_training_task
from superagent.splits import sample_groups


def metrics(matrix):
    tp = np.diagonal(matrix,axis1=-2,axis2=-1)
    denominator = matrix.sum(axis=-1)+matrix.sum(axis=-2)
    f1 = np.divide(2*tp,denominator,where=denominator>0,out=np.zeros_like(tp,dtype=float))
    return {'accuracy':tp.sum(axis=-1)/matrix.sum(axis=(-1,-2)),'macro_f1':f1.mean(axis=-1)}


def grouped(truth,predictions,groups,primary,signed):
    labels,y = np.unique(truth,return_inverse=True)
    group_ids,index = np.unique(groups,return_inverse=True)
    size = len(labels)
    counts = np.zeros((len(group_ids),2,size,size),dtype=np.int64)
    for arm,prediction in enumerate(predictions):
        codes = np.searchsorted(labels,prediction)
        if not np.array_equal(labels[codes],prediction):
            raise ValueError('A prediction is outside the registered class domain')
        np.add.at(counts,(index,arm,y,codes),1)
    point = metrics(counts.sum(axis=0))
    for name in ('accuracy','macro_f1'):
        expected = [signed[k]['scores'][name] for k in ('original','extended')]
        if not np.allclose(point[name],expected,rtol=0,atol=1e-12):
            raise ValueError('All groups taken once do not reproduce the reported point estimate')
    rng = np.random.default_rng(112204)
    samples = {name:[] for name in point}
    # Bounded batches keep the group-level confusion matrices small.
    for _ in range(80):
        draw = rng.integers(0,len(group_ids),size=(50,len(group_ids)))
        resampled = metrics(counts[draw].sum(axis=1))
        for name,values in resampled.items():
            samples[name].extend((100*(values[:,1]-values[:,0])).tolist())
    result = {}
    for name,values in samples.items():
        bounds = np.percentile(values,[2.5,97.5]).tolist()
        result[name] = dict(point_delta_pp=100*float(point[name][1]-point[name][0]),percentile_95_pp=bounds)
    return dict(primary=primary,groups=len(group_ids),rows=len(y),max_group_rows=int(np.bincount(index).max()),
                changed_prediction_rows=int(np.sum(predictions[0]!=predictions[1])),metrics=result)


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    old_title,old_sms = args.headlines.resolve(),args.sms.resolve()
    new_title,new_sms = args.title_runs.resolve(),args.sms_runs.resolve()
    rec.write(out/'protocol.json',dict(batch='B-TEXT-KERNEL-04',runner_sha256=ab_run.sha(Path(__file__)),
              plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/B_TEXT_KERNEL_04_grouped_uncertainty_plan.md'),
              resamples=4000,seed=112204,fit_limit=0,api_limit=0))
    try:
        sources = [('headlines',old_title,new_title,'b_text_kernel_titles'),('sms',old_sms,new_sms,'b_text_kernel_sms')]
        if (ab_run.sha(args.staged)!=rec.read(rec.ROOT/'research/ba19_released_delivery/preflight.json')['headlines']['source_staged_sha256']
                or ab_run.sha(old_title/'preflight.json')!='f41552bfd5d4780cd4a64db1024dc6e81a8c67beb11130fd7a4eaff7fdbb3d95'
                or ab_run.sha(old_sms/'preflight.json')!='630a11762c8154429d5b82d67ae97f1b7fd53736c66c29bfd600e5129f754378'):
            raise ValueError('Original training-only inputs or registered row fingerprints changed')
        whole = load_training_task(args.staged)
        all_groups = sample_groups(whole,'shared_text')
        reports = {}
        for task,old,new,archive in sources:
            if ab_run.sha(new/'results.json')!=ab_run.sha(rec.ROOT/'research'/archive/'results.json'):
                raise ValueError('Signed comparison changed')
            pairs = rec.read(new/'results.json')['pairs']
            pre = rec.read(old/'preflight.json')
            for k,pair in pairs.items():
                if task=='headlines':
                    raw = gzip.decompress((old/k/'outer_ids.json.gz').read_bytes())
                    if rec.sha(raw)!=pre['tasks'][k]['outer_ids_sha256']:
                        raise ValueError('Headline source rows changed')
                    ids = json.loads(raw)['held']
                    truth,groups = whole.y_train[ids],all_groups[ids]
                    previous = old/(k+'_original')
                else:
                    held = old/k/'holdout.joblib'
                    if ab_run.sha(held)!=pre[k]['files']['holdout.joblib']:
                        raise ValueError('Held SMS data changed')
                    data = joblib.load(held)
                    truth = data['y']
                    groups = sample_groups(SimpleNamespace(kind='single_text',X_train=data['X']),'exact_rows')
                    previous = old/k
                predictions = []
                for arm,folder in [('original',previous),('extended',new/k)]:
                    path = folder/'export_0/rows.csv'
                    if ab_run.sha(path)!=pair['grades'][arm]['csv_sha256']:
                        raise ValueError('Scored predictions changed')
                    predictions.append(np.asarray(read_export(path,len(truth)),dtype=truth.dtype))
                primary = pair['grades']['original']['scores']['primary_metric']
                reports[task+'_'+k] = grouped(truth,predictions,groups,primary,pair['grades'])
        if rec.trace(out):
            raise ValueError('Unexpected supervised fitting')
        rec.write(out/'results.json',reports)
        rec.write(out/'completion.json',dict(status='completed',fixed_prediction_pairs=len(reports),resamples_per_pair=4000,
                  new_fits=0,new_predictions=0,new_model_api_requests=0,new_course_rmb=0))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),new_fits=len(rec.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('headlines','sms','title-runs','sms-runs','staged','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
