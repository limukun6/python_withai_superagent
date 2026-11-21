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
from superagent.ab_methods import RELATION
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import content_digest, registered_partitioner


def run(prepared, source, output, baseline=None):
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
    batch,plan = ('BR-03','BR_03_relation_method_plan.md') if baseline else ('BR-02','BR_02_relation_run_plan.md')
    rec.write(output/'protocol.json',dict(batch=batch,input_sha256=registration['train_sha256'],
        source_sha256=rec.digest(ab_run.sources()),plan_sha256=ab_run.sha(rec.ROOT/'docs/batches'/plan),
        driver_sha256=ab_run.sha(Path(__file__)),splits=registration['signed_split_sha256'],
        baseline_menu_sha256=rec.digest(registration['menu']),seed=707,fit_cap=11,time_limit=1200,worker_limit=300,api_cap=0))
    try:
        if any(ab_run.sha(path)!=digest for path,digest in files.items()):
            raise ValueError('A registered input changed')
        case = output/'agent'
        case.mkdir()
        shutil.copy2(prepared/'optimizer/train.zip',case/'train.zip')
        task = load_development_task(case/'train.zip')
        metric = MetricSpec.for_task(task)
        menu = capabilities_for(TaskSpec.from_task(task).record(),train_rows=len(task.y_train))
        common = {k:v for k,v in menu.items() if baseline is None or k!=RELATION}
        if common!=registration['menu'] or list(common)!=list(registration['menu']) or task.X_val is not None:
            raise ValueError('Unregistered input, candidate order or menu')
        if baseline:
            original = rec.ROOT/'research/br_02_relation'
            for name in ('protocol.json','results.json','frozen_model.json','completion.json'):
                if ab_run.sha(baseline/name)!=ab_run.sha(original/name):
                    raise ValueError('Original reference differs from the archived BR-02 result')
            old_manifest = rec.read(baseline/'agent/run/run_manifest.json')
            old_sources,new_sources = old_manifest['source_files'],ab_run.sources()
            changes = sorted(k for k in set(old_sources)|set(new_sources) if old_sources.get(k)!=new_sources.get(k))
            if (list(menu)!=list(common)+[RELATION] or changes!=['ab_catalog.py','ab_methods.py']
                    or old_manifest['task_archive_sha256']!=ab_run.sha(case/'train.zip')
                    or old_manifest['capabilities']!=common or old_manifest['fit_limit']!=11
                    or old_manifest['seed']!=707 or old_manifest['time_limit_seconds']!=1200
                    or old_manifest['trial_timeout_seconds']!=300 or old_manifest['planner_mode']!='local'
                    or old_manifest['scheduler']!='ab-v1'):
                raise ValueError('Baseline inputs, software difference, budget or menu differ from registration')
        registered = case/'registered'
        registered.mkdir()
        inner = prepared/'optimizer/inner.json'
        signed = registration['signed_split_sha256']
        partitioner = registered_partitioner(task,inner,signed)
        with patch.object(ab_run,'grouped_partition',partitioner):
            _,splits = ab_run.freeze_partitions(task,registered,707,metric)
        if baseline and any(rec.read(registered/(s+'_split.json'))!=rec.read(baseline/'agent/run'/(s+'_split.json'))
                            for s in splits):
            raise ValueError('New and baseline training splits differ')
        rec.write(output/'preflight.json',dict(candidate_ids=list(menu),catalog_sha256=rec.digest(menu),splits=splits,
                  changed_production_files=changes if baseline else [],common_anchor=next(iter(menu))))
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
        result = dict(path=trained,official_metric=scores,correct=int((correct==predicted).sum()),
            total=len(ids),training_majority=majority,majority_correct=int((correct==majority).sum()),
            majority=majority_scores,labels=list(metric.labels),
            confusion_actual_by_prediction=confusion_matrix(correct,predicted,labels=list(metric.labels)).tolist(),
            two_identical_prediction_exports=ab_run.sha(exported[0]),outer_source_overlap=registration['outer']['families_shared'])
        if baseline:
            prior = rec.read(baseline/'results.json')
            base_csv = baseline/'export_0/rows.csv'
            if ab_run.sha(base_csv)!=prior['two_identical_prediction_exports']:
                raise ValueError('Baseline CSV changed')
            old = np.array(read_export(base_csv,len(ids)))==correct
            new = predicted==correct
            if int(old.sum())!=prior['correct']:
                raise ValueError('Baseline correct count no longer matches on the same rows')
            result['comparison'] = dict(prior_correct=int(old.sum()),newly_correct=int((~old & new).sum()),
                newly_wrong=int((old & ~new).sum()),accuracy_difference_pp=100*(scores['accuracy']-prior['official_metric']['accuracy']),
                macro_f1_difference_pp=100*(scores['macro_f1']-prior['official_metric']['macro_f1']),
                common_candidates=len(common),expanded_candidates=len(menu),changed_production_files=changes,
                new_candidate_positions=[i for i,a in enumerate(trained['actions'],1) if a['candidate_id']==RELATION],
                new_candidate_selected=trained['selected']==RELATION)
        rec.write(output/'results.json',result)
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
    p.add_argument('--compare-baseline',type=Path)
    args=p.parse_args()
    run(args.prepared.resolve(),args.source.resolve(),args.output.resolve(),
        args.compare_baseline.resolve() if args.compare_baseline else None)
