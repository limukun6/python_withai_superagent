"""Freeze a three-modality development pilot before running the official AB CLI."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts import run_b1_ab_acceptance as offline_fixture
from superagent.ab_run import sources, freeze_partitions
from superagent.ab_catalog import capabilities_for
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('bean', 'sms', 'digits')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(command, log, timeout):
    env = {**os.environ, 'SUPERAGENT_OFFLINE':'1', 'PYTHONIOENCODING':'utf-8',
           'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as file:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'], capture_output=True)
            child.wait(timeout=15)
            raise TimeoutError('Pilot subprocess timed out; stopped without retry')
    if code:
        raise RuntimeError('Pilot subprocess failed; see ' + str(log))
    return time.monotonic()-start


def run(args, comparison=False):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    offline_fixture.offline(output)
    start = time.monotonic()
    frozen = sources()
    results = []
    records = []
    try:
        metadata = {t['name']:t for t in read(args.prepared/'tasks.json')}
        for name in NAMES:
            source = args.prepared/name
            meta = metadata[name]
            for fname, digest in meta['files'].items():
                if sha(source/fname) != digest:
                    raise ValueError('Prepared task bytes changed: ' + name + '/' + fname)
            frozen_dir = output/(name+'_partitions')
            frozen_dir.mkdir()
            task = load_development_task(source/'train.zip')
            metric = MetricSpec.for_task(task)
            spec = TaskSpec.from_task(task).record()
            _, allowed = freeze_partitions(task, frozen_dir, 707, metric)
            caps = capabilities_for(spec, train_rows=len(task.y_train))
            if not caps:
                raise ValueError('No executable candidate: '+name)
            records.append(dict(name=name, kind=task.kind, metric=metric.name, task=meta,
                                capabilities=caps, splits=allowed,
                                frozen_partitions={s:read(frozen_dir/(s+'_split.json')) for s in allowed}))
        dependencies = subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True)
        (output/'environment.txt').write_text(dependencies, encoding='utf-8')
        schedule = [('bean','ab-fixed'),('bean','ab-v1'),('sms','ab-v1'),('sms','ab-fixed'),
                    ('digits','ab-fixed'),('digits','ab-v1')] if comparison else [(n,'ab-v1') for n in NAMES]
        save(output/'protocol.json', dict(batch='A11' if comparison else 'A10', source_files=frozen,
            baseline='5ed06c6' if comparison else '07884d0', schedule=schedule,
            driver_sha256=sha(Path(__file__)), plan_sha256=sha(ROOT/('docs/batches/A11_fixed_dynamic_plan.md' if comparison
                else 'docs/batches/A10_real_task_pilot_plan.md')),
            tasks=records, fit_limit_per_path=11, fit_limit_batch=66 if comparison else 33, api_limit=0,
            seed=707, trial_timeout=60, time_limit=300, validate_only=args.validate_only,
            environment_sha256=sha(output/'environment.txt'),
            scope='Paired fixed/dynamic development comparison' if comparison else 'Real development pilot; no paired efficacy comparison; all models frozen before scoring'))
        if args.validate_only:
            save(output/'completion.json',dict(status='validated', actual_classifier_fits=0, api_requests=0,
                source_unchanged=sources()==frozen, elapsed_seconds=time.monotonic()-start))
            return
        for task_name, scheduler in schedule:
            entry = next(r for r in records if r['name']==task_name)
            name = task_name+'_'+scheduler if comparison else task_name
            source = args.prepared/task_name
            directory = output/name
            if sources()!=frozen:
                raise ValueError('Production source changed before launch')
            command = [sys.executable,'-m','superagent','run',str(source/'train.zip'),'--run-dir',str(directory),
                '--scheduler',scheduler,'--portfolio','ab-v1','--fit-limit','11','--seed','707',
                '--trial-timeout','60','--time-limit','300']
            save(output/(name+'_command.json'), command)
            seconds = launch(command, output/(name+'_train.log'), 330)
            outcome = read(directory/'run_outcome.json')
            counts = outcome['classifier_fit_accounting']
            if not (outcome['requested_planner_completed'] and outcome['new_api_requests']==0
                    and outcome['source_unchanged'] and counts['unknown_failed_attempts']==0
                    and 2<=counts['attempted_classifier_fits']==counts['completed_classifier_fits']<=11):
                raise ValueError('Run failed completion or resource requirements: '+name)
            for split, expected in entry['frozen_partitions'].items():
                actual = read(directory/(split+'_split.json'))
                if actual!=expected:
                    raise ValueError('Registered split differs from frozen row IDs: '+name)
            manifest=read(directory/'run_manifest.json')
            if manifest['capabilities']!=entry['capabilities'] or manifest['source_files']!=frozen:
                raise ValueError('Candidate capabilities or production source differ: '+name)
            selection = read(directory/'final_selection.json')
            if sha(directory/selection['model_file']) != selection['model_sha256']:
                raise ValueError('Frozen model digest differs')
            results.append(dict(name=name, task=task_name, scheduler=scheduler, train_process_seconds=seconds, outcome=outcome,
                                selection=selection, actions=read(directory/'actions.json')))
            save(output/'checkpoint.json',results)
            print(json.dumps(dict(task=name,status='model_frozen',fits=counts['attempted_classifier_fits'],
                                  seconds=seconds,selected=outcome['selected_candidate'])), flush=True)
        save(output/'all_models_frozen.json', [dict(name=r['name'],selection=r['selection']) for r in results])
        if comparison:
            for task_name in NAMES:
                pair=[r for r in results if r['task']==task_name]
                candidates=[[a['candidate_id'] for a in r['actions'] if a['action']=='explore'] for r in pair]
                length=min(map(len,candidates))
                if candidates[0][:length]!=candidates[1][:length]:
                    raise ValueError('Common exploration prefix differs: '+task_name)
            save(output/'fairness.json',dict(same_source=True,same_candidates=True,same_actual_partitions=True,
                same_budget_caps=True,same_exploration_prefix=True,all_models_frozen_before_scoring=True))
        for row in results:
            name = row['name']
            directory = output/name
            source = args.prepared/row['task']
            relocated = directory/'relocated'
            relocated.mkdir()
            for fname in ('final_selection.json','task_spec.json',row['selection']['model_file']):
                shutil.copy2(directory/fname, relocated/fname)
            export_seconds = 0
            csvs = []
            for i, model_dir in enumerate((directory,relocated)):
                target = model_dir/'predictions'/'rows.csv'
                export_seconds += launch([sys.executable,'-m','superagent','predict',str(source/'features.zip'),
                    '--run-dir',str(model_dir),'--output',str(target)], output/f'{name}_predict_{i}.log',60)
                csvs.append(target)
            if csvs[0].read_bytes()!=csvs[1].read_bytes():
                raise ValueError('Relocated model predicts different rows: '+name)
            export_seconds += launch([sys.executable,str(ROOT/'scripts/a07_verify_export.py'),
                '--source-root',str(ROOT),'--run-dir',str(directory),'--holdout',str(source/'holdout.joblib'),
                '--predictions',str(csvs[0]),'--output',str(directory/'export_verification.json')],
                output/(name+'_verify.log'),60)
            row.update(export_seconds=export_seconds,verification=read(directory/'export_verification.json'),
                       relocated_prediction_identical=True,prediction_csv_sha256=sha(csvs[0]))
            save(output/'results.json',results)
        attempted = sum(r['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for r in results)
        physical = offline_fixture.trace(output)
        if len(physical)!=attempted or sources()!=frozen:
            raise ValueError('Independent fit count or frozen production source differs')
        save(output/'completion.json',dict(status='passed', tasks_completed=len(results),
            attempted_classifier_fits=attempted, completed_classifier_fits=attempted,
            independent_observed_fits=len(physical),api_requests=0,source_unchanged=True,
            elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(output/'results.json',results)
        save(output/'completion.json',dict(status='failed',error=str(error),api_requests=0,
            independent_observed_fits=len(offline_fixture.trace(output)),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    run(parser.parse_args())
