"""Prototype only: recover a valid completed primary checkpoint without retraining or API."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts.classifier_fit_observer import observe_classifier_fits
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from superagent.data import load_training_task,DevelopmentTask,load_prediction_features
from superagent.events import RunLogger
from superagent.finalization import finalize
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model


def recover(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources()
    original=read(args.run/'run_outcome.json');manifest=read(args.run/'run_manifest.json')
    if original.get('requested_planner_completed') or original.get('status')!='failed':
        raise ValueError('This prototype requires a preserved incomplete run, not a completed solution')
    if sha(args.training)!=manifest['task_archive_sha256']:raise ValueError('Training package differs from the original run')
    training=load_training_task(args.training)
    task=DevelopmentTask(training.kind,'training-only partial recovery',training.X_train,training.y_train,None,None,training.spec)
    metric=MetricSpec.for_task(task)
    history=read(args.run/'history.json')
    rows=[row for row in history if row.get('status')=='completed' and row.get('split_id')=='primary'
          and row.get('primary_metric')==metric.name and math.isfinite(row.get('search_score',float('nan')))]
    if not rows:raise ValueError('No completed primary checkpoint can be safely recovered')
    chosen=sorted(rows,key=lambda row:(-row['search_score'],row['trial']))[0]
    artifact=args.run/f'trial_{chosen["trial"]}.joblib';counts=read(artifact.with_suffix('.fits.json'))
    if counts['status']!='completed' or counts['completed_classifier_fits']<1 or counts['completed_classifier_fits']!=counts['attempted_classifier_fits']:
        raise ValueError('Incomplete fit artifact cannot be exported')
    bundle=joblib.load(artifact)
    actual=bundle['metrics']['resolved_configuration']
    if actual['id']!=chosen['candidate'] or actual['params']!=chosen['params']:
        raise ValueError('Resolved model differs from recorded successful trial')
    payload=joblib.load(args.run/'primary.joblib')
    if len(payload['y_train'])>=len(task.y_train):raise ValueError('Expected a search-partition checkpoint, not a full refit')
    save(root/'protocol.json',dict(batch='A39-partial-prototype',source_files=frozen,original_training_source=manifest['source_files'],
        runner_sha256=sha(Path(__file__)),guard_sha256=sha(Path(__file__).resolve().parents[1]/'scripts/a38_predict_guard.py'),
        plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A39_partial_delivery_prototype_plan.md'),
        original_outcome_sha256=sha(args.run/'run_outcome.json'),checkpoint_sha256=sha(artifact),
        rule='Best completed primary score, ties by trial; no external validation',actual_classifier_fit_limit=0,API_limit=0,
        automatic_recovery_in_live_agent=False))
    def no_training(*a,**kw):raise RuntimeError('Retraining forbidden in partial recovery')
    def no_network(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('Network forbidden in partial recovery')
    sys.addaudithook(no_network)
    logger=RunLogger(root)
    with observe_classifier_fits() as calls:
        result=finalize(task,history,bundle['model'],chosen['candidate'],chosen['search_score'],payload,root,logger,no_training,
            time.monotonic(),60,60,False,False)
        if calls:raise ValueError('Unexpected fit during partial export')
    selected=read(root/'final_selection.json')
    if selected['refit'] or selected['training_rows']!=len(payload['y_train']):raise ValueError('Partial training scope was misreported')
    x=load_prediction_features(args.features,task.kind,read(root/'task_spec.json'))
    expected=[str(value) for value in predict_with_model(task.kind,bundle['model'],x).tolist()]
    guard=Path(__file__).resolve().parents[1]/'scripts/a38_predict_guard.py'
    with (root/'predict.log').open('w',encoding='utf-8') as log:
        code=subprocess.run([sys.executable,'-m','scripts.a38_predict_guard',str(args.features),'--run-dir',str(root),
            '--output',str(root/'predictions/rows.csv')],cwd=Path(__file__).resolve().parents[1],stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
    if code:raise RuntimeError('Partial model guarded prediction failed')
    import csv
    with (root/'predictions/rows.csv').open(encoding='utf-8',newline='') as file:predicted=[row['prediction'] for row in csv.DictReader(file)]
    if predicted!=expected:raise ValueError('Partial export predictions differ')
    save(root/'completion.json',dict(status='valid_partial_delivery_prototype',delivery_status='partial',
        requested_planner_completed=False,automatic_recovery_in_live_agent=False,partial_solution_available=True,
        source_trial=chosen['trial'],source_candidate=chosen['candidate'],checkpoint_training_rows=len(payload['y_train']),
        full_training_rows=len(task.y_train),refit_performed=False,rows_predicted=len(predicted),reload_identical=True,
        external_labels_opened=False,actual_classifier_fits=0,real_api_requests=0,original_run_modified=False,
        source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(status='valid_partial_delivery_prototype',training_rows=len(payload['y_train']),
        full_training_rows=len(task.y_train),predictions=len(predicted),fits=0,API=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','training','features','output'):parser.add_argument('--'+name,type=Path,required=True)
    recover(parser.parse_args())
