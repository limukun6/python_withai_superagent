"""Locate errors using confidence from four frozen models on registered development rows."""
import argparse
from contextlib import ExitStack
import gzip
import json
import os
from pathlib import Path
from unittest.mock import patch

from catboost import CatBoostClassifier
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts import run_b1_ab_acceptance as record
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_development_task, load_prediction_features, load_training_task
from superagent.models import probability_with_model
from tools.ab_registered_rows import content_digest


def describe(indices, confidence, correct):
    n = len(indices)
    return dict(rows=n,errors=int((~correct[indices]).sum()),correct=int(correct[indices].sum()),
                coverage=n/len(correct),accuracy=float(correct[indices].mean()) if n else None,
                mean_confidence=float(confidence[indices].mean()) if n else None)


def diagnose(probabilities, classes, truth, ids, exported):
    p = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(classes).reshape(-1)
    if (set(labels.tolist())!={0,1} or p.shape!=(len(truth),2) or not np.all(np.isfinite(p))
            or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1.,rtol=0,atol=1e-8)):
        raise ValueError('Model probabilities or class order are not valid for this binary task')
    index = np.argmax(p,axis=1)
    if not np.array_equal(labels[index],exported):
        raise ValueError('Probabilities disagree with the frozen prediction file')
    confidence = p[np.arange(len(p)),index]
    sorted_ids = np.lexsort((np.asarray(ids),confidence))
    bands = np.array_split(sorted_ids,5)
    correct = exported==truth
    quintiles = [describe(b,confidence,correct) for b in bands]
    bins = np.minimum(np.floor(confidence*10).astype(int),9)
    calibration = [describe(np.flatnonzero(bins==b),confidence,correct) for b in range(10)]
    ece = sum(r['coverage']*abs(r['accuracy']-r['mean_confidence']) for r in calibration if r['rows'])
    positive = p[:,np.flatnonzero(labels==1)[0]]
    return dict(rows=len(p),correct=int(correct.sum()),confidence_quintiles=quintiles,
        fraction_errors_in_lowest_quintile=quintiles[0]['errors']/int((~correct).sum()) if (~correct).any() else None,
        highest_four_quintiles=describe(np.concatenate(bands[1:]),confidence,correct),
        thresholds={str(t):describe(np.flatnonzero(confidence>=t),confidence,correct) for t in (.8,.9)},
        positive_class_brier=float(np.mean((positive-(truth==1))**2)),top_label_ece_10=ece,
        equal_width_bins=calibration,ordered_probabilities_sha256=record.sha(p.astype('<f8').tobytes()))


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    research = record.ROOT/'research'
    source = record.read(research/'ba19_released_delivery/preflight.json')
    context = args.headline_packets.resolve()
    expected_context = record.read(research/'b_route_titles/protocol.json')['registered_inputs_sha256']
    if ab_run.sha(context/'preflight.json')!=expected_context:
        raise ValueError('Original registered headline rows changed')
    datasets = {'coupon':(args.coupons.resolve(),args.coupon_source.resolve(),'b_route_coupons'),
                'headlines':(args.headlines.resolve(),args.headline_source.resolve(),'b_route_titles')}
    record.write(out/'protocol.json',dict(batch='B-RELEASED-CONFIDENCE-01',driver_sha256=ab_run.sha(Path(__file__)),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_RELEASED_CONFIDENCE_01_plan.md'),
        seeds=[45101,45102],quintiles=5,thresholds=[.8,.9],calibration_bins=10,positive_class=1,
        input_results={k:ab_run.sha(research/v[2]/'results.json') for k,v in datasets.items()},
        new_fit_limit=0,model_api_limit=0))
    try:
        reports = {}
        with ExitStack() as guard:
            for estimator in (LogisticRegression,CatBoostClassifier):
                guard.enter_context(patch.object(estimator,'fit',side_effect=AssertionError('Frozen probability audit cannot train')))
            for name,(origin,staged,collection) in datasets.items():
                signed = research/collection
                for file in ('all_models_frozen.json','results.json','protocol.json','completion.json'):
                    if ab_run.sha(origin/file)!=ab_run.sha(signed/file):
                        raise ValueError('Earlier frozen experiment differs from published data')
                if ab_run.sha(staged)!=source[name]['source_staged_sha256']:
                    raise ValueError('Expected the already staged training-only course archive')
                whole = load_training_task(staged)
                headlines = name=='headlines'
                if not headlines and ab_run.sha(origin/'preflight.json')!=ab_run.sha(signed/'preflight.json'):
                    raise ValueError('Registered coupon rows changed')
                pre = record.read((context if headlines else origin)/'preflight.json')
                frozen = record.read(signed/'all_models_frozen.json')
                grade = record.read(signed/'results.json')['pairs']
                for seed in ('45101','45102'):
                    packet = (context if headlines else origin)/seed
                    case = origin/(seed if headlines else seed+'_representative')
                    entry = pre['tasks'][seed] if headlines else pre['entries'][seed]
                    expected_features = pre['entries'][seed+'_original']['features_sha256'] if headlines else entry['features_sha256']
                    if ab_run.sha(packet/'features.zip')!=expected_features:
                        raise ValueError('Prediction rows changed')
                    ids = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
                    if record.sha(ids)!=entry['outer_ids_sha256']:
                        raise ValueError('Held row IDs changed')
                    ids = json.loads(ids)['held']
                    truth = whole.y_train[ids]
                    task = load_development_task(packet/'train.zip')
                    features = load_prediction_features(packet/'features.zip',task.kind,task.spec)
                    if content_digest(features,truth)!=entry['held_content_sha256'] or len(set(ids))!=len(ids):
                        raise ValueError('Held rows or labels changed')
                    choice = record.read(case/'relocated/final_selection.json')
                    path = case/'relocated'/choice['model_file']
                    identity = frozen[seed]['representative'] if headlines else frozen[seed+'_representative']
                    if ab_run.sha(path)!=identity or choice['model_sha256']!=identity:
                        raise ValueError('Relocated model differs from frozen selection')
                    prior = grade[seed]['grade']['representative']
                    export = case/'export_0/rows.csv'
                    if any(ab_run.sha(case/f'export_{i}/rows.csv')!=prior['csv_sha256'] for i in range(2)):
                        raise ValueError('Original prediction files changed')
                    reference = np.asarray(read_export(export,len(ids)),dtype=truth.dtype)
                    model = joblib.load(path)['model']
                    classes,p = probability_with_model(task.kind,model,features)
                    data = diagnose(p,classes,truth,ids,reference)
                    if data['correct']!=prior['correct'] or data['rows']!=prior['rows']:
                        raise ValueError('Prior development score did not reproduce')
                    reports[name+'_'+seed] = dict(model=choice['family'],model_sha256=identity,
                        csv_sha256=prior['csv_sha256'],held_rows_sha256=entry['outer_ids_sha256'],**data)
                record.write(out/'checkpoint.json',reports)
        record.write(out/'results.json',reports)
        record.write(out/'completion.json',dict(status='completed',frozen_models=len(reports),new_fits=0,
            new_model_api_requests=0,new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error),new_fits=len(record.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coupons','headlines','coupon-source','headline-source','headline-packets','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
