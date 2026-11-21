"""Run the native local scheduler once on registered source-isolated ordered text data."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
from sklearn.metrics import confusion_matrix

from scripts import run_b1_05_pair_pilot as pilot
from scripts import run_b1_ab_acceptance as rec
from scripts.prepare_b_ordered_relation import read_train
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.ab_catalog import capabilities_for
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import content_digest, registered_partitioner


def run(prepared, source, output):
    output.mkdir(parents=True,exist_ok=False)
    rec.offline(output)
    os.environ.update(SUPERAGENT_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    archive = rec.ROOT/'research/br_01_relation_groups'
    registration = rec.read(archive/'results.json')
    started = time.monotonic()
    files = {prepared/'optimizer/train.zip':registration['train_sha256'],
             prepared/'optimizer/inner.json':registration['signed_split_sha256'],
             prepared/'delivery/features.zip':registration['features_sha256'],
             prepared/'results.json':ab_run.sha(archive/'results.json')}
    rec.write(output/'protocol.json',dict(batch='BR-02',input_sha256=registration['train_sha256'],
        source_sha256=rec.digest(ab_run.sources()),plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/BR_02_relation_run_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),splits=registration['signed_split_sha256'],
        menu=rec.digest(registration['menu']),seed=707,fit_cap=11,time_limit=1200,worker_limit=300,api_cap=0))
    try:
        if any(ab_run.sha(path)!=digest for path,digest in files.items()):
            raise ValueError('A registered input changed')
        case = output/'agent'
        case.mkdir()
        shutil.copy2(prepared/'optimizer/train.zip',case/'train.zip')
        task = load_development_task(case/'train.zip')
        metric = MetricSpec.for_task(task)
        menu = capabilities_for(TaskSpec.from_task(task).record(),train_rows=len(task.y_train))
        if menu!=registration['menu'] or list(menu)!=list(registration['menu']) or task.X_val is not None:
            raise ValueError('Unregistered input, candidate order or menu')
        registered = case/'registered'
        registered.mkdir()
        inner = prepared/'optimizer/inner.json'
        signed = registration['signed_split_sha256']
        partitioner = registered_partitioner(task,inner,signed)
        with patch.object(ab_run,'grouped_partition',partitioner):
            _,splits = ab_run.freeze_partitions(task,registered,707,metric)
        entry = dict(name='ordered_relation',files={'train.zip':registration['train_sha256']},
                     splits=splits,catalog_sha256=rec.digest(menu))
        trained = pilot.train_one(case,entry,rec.digest(ab_run.sources()),
                     launcher=('-m','tools.ab_registered_run','--split-manifest',str(inner),'--split-sha256',signed,'--','run'))
        if trained['fits']!=len(rec.trace(output)):
            raise ValueError('Independent observation differs from supervisor fit counter')
        rec.write(output/'frozen_model.json',dict(model=trained['model_sha256'],candidate=trained['selected']))
        official = case/'run'
        selection = rec.read(official/'final_selection.json')
        relocated = output/'relocated'
        relocated.mkdir()
        for name in ('task_spec.json','final_selection.json',selection['model_file']):
            shutil.copy2(official/name,relocated/name)
        exported = []
        for number,model_root in enumerate((official,relocated)):
            destination = output/f'export_{number}'/'rows.csv'
            pilot.child(['-m','superagent','predict',prepared/'delivery/features.zip','--run-dir',model_root,
                         '--output',destination],output/f'predict_{number}.log',pilot.TRIAL)
            read_export(destination,registration['outer']['held_rows'])
            exported.append(destination)
        if ab_run.sha(exported[0])!=ab_run.sha(exported[1]):
            raise ValueError('Relocated predictions disagree')

        raw = gzip.decompress((prepared/'source_ids.json.gz').read_bytes())
        if rec.sha(raw)!=registration['outer_source_ids_sha256'] or raw!=gzip.decompress((archive/'source_ids.json.gz').read_bytes()):
            raise ValueError('Held source IDs changed')
        ids = json.loads(raw)['held']
        records = {r[0]:r for r in read_train(source)}
        correct = np.array([records[i][3] for i in ids])
        query = load_prediction_features(prepared/'delivery/features.zip',task.kind,task.spec)
        if (query.values.tolist()!=[[records[i][1],records[i][2]] for i in ids] or
                content_digest(query,correct)!=registration['held_content_sha256']):
            raise ValueError('Source rows, feature sequence or grading labels do not match')
        predicted = np.asarray(read_export(exported[0],len(ids)))
        scores = metric.evaluate(correct,predicted)
        counts = {k:int((task.y_train==k).sum()) for k in metric.labels}
        majority = max(metric.labels,key=counts.get)
        majority_scores = metric.evaluate(correct,np.full(len(correct),majority))
        rec.write(output/'results.json',dict(path=trained,official_metric=scores,correct=int((correct==predicted).sum()),
            total=len(ids),training_majority=majority,majority_correct=int((correct==majority).sum()),
            majority=majority_scores,labels=list(metric.labels),
            confusion_actual_by_prediction=confusion_matrix(correct,predicted,labels=list(metric.labels)).tolist(),
            two_identical_prediction_exports=ab_run.sha(exported[0]),outer_source_overlap=registration['outer']['families_shared']))
        rec.write(output/'completion.json',dict(status='completed',full_autonomous_runs=1,fit_calls=len(rec.trace(output)),
            model_api_requests=0,new_course_cost_rmb=0,new_official_course_scores=0,
            development_outer_rows=len(ids),fresh_prediction_processes=2,wall_seconds=time.monotonic()-started))
    except Exception as error:
        rec.write(output/'completion.json',dict(status='failed',reason=str(error),
                  fits=len(rec.trace(output)),model_api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    run(args.prepared.resolve(),args.source.resolve(),args.output.resolve())
