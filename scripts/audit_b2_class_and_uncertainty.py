"""Audit paired, class-specific changes and grouped uncertainty on B2's existing exports."""
import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as record
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_training_task
from superagent.splits import sample_groups
from tools.ab_registered_rows import content_digest


PAIRS = (('original','expanded'),('expanded','composed'),('original','composed'))


def analyze(truth, first, second, group_ids, expected, primary):
    labels,actual = np.unique(truth,return_inverse=True)
    groups,index = np.unique(group_ids,return_inverse=True)
    counts = np.zeros((len(groups),2,len(labels),len(labels)),dtype=np.int64)
    details = {}
    for arm,pred in enumerate((first,second)):
        numeric = np.searchsorted(labels,pred)
        if not np.array_equal(labels[numeric],pred):
            raise ValueError('Predicted an unregistered class')
        np.add.at(counts,(index,arm,actual,numeric),1)
    def scores(matrix):
        tp = np.diagonal(matrix,axis1=-2,axis2=-1)
        total = matrix.sum(axis=(-1,-2))
        denominator = matrix.sum(axis=-2)+matrix.sum(axis=-1)
        macro = np.divide(2*tp,denominator,where=denominator>0,out=np.zeros_like(tp,dtype=float)).mean(axis=-1)
        return tp.sum(axis=-1)/total,macro
    points = dict(zip(('accuracy','macro_f1'),scores(counts.sum(axis=0)),strict=True))
    for metric,values in points.items():
        if not np.allclose(values,[r['scores'][metric] for r in expected],rtol=0,atol=1e-12):
            raise ValueError('Frozen primary or secondary scores do not reproduce')
    for label in labels:
        true = truth==label
        old,new = first==label,second==label
        details[str(label)] = dict(rows=int(true.sum()),first_tp=int((true&old).sum()),second_tp=int((true&new).sum()),
            first_recall=float(old[true].mean()),second_recall=float(new[true].mean()),
            first_precision=float(true[old].mean()) if old.any() else None,second_precision=float(true[new].mean()) if new.any() else None,
            gained=int((true&~old&new).sum()),lost=int((true&old&~new).sum()))
    rng = np.random.default_rng(112204)
    samples = {'accuracy':[],'macro_f1':[]}
    for _ in range(80):
        draw = rng.integers(0,len(groups),size=(50,len(groups)))
        values = scores(counts[draw].sum(axis=1))
        for k,v in zip(samples,values,strict=True):
            samples[k].extend((100*(v[:,1]-v[:,0])).tolist())
    return dict(rows=len(truth),groups=len(groups),max_group_rows=int(np.bincount(index).max()),primary=primary,
        metrics={k:dict(delta_pp=100*float(v[1]-v[0]),percentile_95_pp=np.percentile(samples[k],[2.5,97.5]).tolist())
                 for k,v in points.items()},class_details=details)


def audit(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    origin = args.previous.resolve()
    signed = record.ROOT/'research/b2_bounded_text_composition'
    for n in ('preflight','protocol','results','all_six_models_frozen','completion'):
        if ab_run.sha(origin/(n+'.json'))!=ab_run.sha(signed/(n+'.json')):
            raise ValueError('B2 parent differs from published data')
    pre_registration = record.read(signed/'protocol.json')['previous_registrations']
    if (ab_run.sha(args.headlines/'preflight.json')!=pre_registration['headlines']
            or ab_run.sha(args.sms/'preflight.json')!=pre_registration['sms']
            or ab_run.sha(args.staged)!=record.read(record.ROOT/'research/ba19_released_delivery/preflight.json')['headlines']['source_staged_sha256']):
        raise ValueError('Registered data source changed')
    record.write(out/'protocol.json',dict(batch='B2-03',parent_results_sha256=ab_run.sha(signed/'results.json'),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B2_03_uncertainty_plan.md'),driver_sha256=ab_run.sha(Path(__file__)),
        contrasts=PAIRS,seed=112204,resamples=4000,new_fit_limit=0,new_inference_limit=0,model_api_limit=0))
    try:
        results = {}
        paths = record.read(signed/'results.json')['paths']
        for name in ('headlines','sms'):
            if name=='headlines':
                src = load_training_task(args.staged)
                raw = gzip.decompress((args.headlines/'45102/outer_ids.json.gz').read_bytes())
                trusted = record.read(args.headlines/'preflight.json')['tasks']['45102']
                if record.sha(raw)!=trusted['outer_ids_sha256']:
                    raise ValueError('Held title IDs changed')
                ids = json.loads(raw)['held']
                truth = src.y_train[ids]
                if content_digest([src.X_train[i] for i in ids],truth)!=trusted['held_content_sha256']:
                    raise ValueError('Title rows or truth changed')
                groups = sample_groups(src,'shared_text')[ids]
            else:
                path = args.sms/'sms/holdout.joblib'
                if ab_run.sha(path)!=record.read(args.sms/'preflight.json')['sms']['files']['holdout.joblib']:
                    raise ValueError('SMS held input changed')
                data = joblib.load(path)
                truth = data['y']
                groups = sample_groups(SimpleNamespace(kind='single_text',X_train=data['X']))
            exported = {}
            for arm in ('original','expanded','composed'):
                files = [origin/(name+'_'+arm)/f'export_{i}/rows.csv' for i in range(2)]
                if any(ab_run.sha(f)!=paths[name+'_'+arm]['grade']['csv_sha256'] for f in files):
                    raise ValueError('Native or relocated predictions changed')
                exported[arm] = np.asarray(read_export(files[0],len(truth)),dtype=truth.dtype)
            results[name] = {}
            for left,right in PAIRS:
                grade = [paths[name+'_'+key]['grade'] for key in (left,right)]
                primary = grade[0]['scores']['primary_metric']
                results[name][right+'_minus_'+left] = analyze(truth,exported[left],exported[right],groups,grade,primary)
        record.write(out/'results.json',results)
        record.write(out/'completion.json',dict(status='completed',contrasts=6,resamples_per_pair=4000,new_fits=0,
                    new_model_predictions=0,new_model_api_requests=0,new_course_rmb=0))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','headlines','sms','staged','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    audit(parser.parse_args())
