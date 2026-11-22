"""Two-task, six-arm live development comparison; train-only inputs and delayed validation."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a20_local_c1 import train_only_copy
from superagent.ab_run import sources, registered_capabilities, freeze_partitions
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_training_task, load_development_task
from superagent.metrics import MetricSpec
from superagent.research_budget import research_preflight


ROOT = Path(__file__).resolve().parents[1]
ARMS = {'L0': ('local', 'ab-fixed'), 'L1': ('local', 'ab-v1'),
        'R0': ('raw-history', 'ab-fixed'), 'E0': ('checked-history', 'ab-fixed'),
        'R1': ('raw-history', 'ab-v1'), 'E1': ('checked-history', 'ab-v1')}
ORDER = [(task, arm) for task, arms in [('coupon', ['L1','R0','E1','L0','E0','R1']),
          ('preference', ['E0','L0','R1','E1','R0','L1'])] for arm in arms]
ROUTE_ORDER = [(task, arm) for task, arms in [
    ('coupon',['L1','R0_off','R0_routes','E1_routes','L0','E1_off','E0_off','E0_routes','R1_routes','R1_off']),
    ('preference',['E0_routes','E0_off','L0','R1_off','R1_routes','E1_off','E1_routes','R0_routes','R0_off','L1'])]
    for arm in arms]


class WindowLimitReached(RuntimeError):
    pass


def settings(arm):
    base, _, coverage = arm.partition('_')
    return (*ARMS[base], coverage or 'off')


def plan_path(args):
    name='A32_route_coverage_matrix_plan.md' if args.study=='route-coverage' else 'A29_limited_choice_matrix_plan.md'
    return ROOT/'docs'/'batches'/name


def physical(root):
    path = root / 'physical_fits.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    frozen = sources()
    save(root / 'protocol.json', dict(batch='A29-prepare', source_files=frozen,
        runner_sha256=sha(Path(__file__)), plan_sha256=sha(plan_path(args)), study=args.study,
        original_readiness_sha256=sha(args.readiness / 'readiness.json'), real_api_requests=0, classifier_fits=0))
    entries = []
    originals = {e['task']: e for e in read(args.readiness / 'readiness.json')}
    original_read = zipfile.ZipFile.read
    def train_only(archive, member, *a, **kw):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if any(value in name.casefold() for value in ('test', 'validation', 'val.csv', 'c_val', 'y_val')):
            raise ValueError('A29 prepare attempted to read nontraining member')
        return original_read(archive, member, *a, **kw)
    for taskname in ('coupon', 'preference'):
        old = originals[taskname]
        source = Path(old['source_archive'])
        if sha(source) != old['task_archive_sha256']:
            raise ValueError('Source task archive differs from the previous registration')
        target = root / taskname
        target.mkdir()
        with patch.object(zipfile.ZipFile, 'read', new=train_only):
            copied = train_only_copy(source, target / 'train.zip')
            task = load_training_task(target / 'train.zip')
        spec = TaskSpec.from_task(task).record()
        caps = registered_capabilities(spec, len(task.y_train), 'full')
        if spec != old['task_spec'] or caps != old['capabilities'] or len(caps) <= 10:
            raise ValueError('Task contract/menu changed or no scarce-choice setting: ' + taskname)
        metric = MetricSpec.for_task(task)
        _, allowed = freeze_partitions(task, target, 707, metric)
        for split in allowed:
            if read(target / (split + '_split.json')) != read(args.readiness / taskname / (split + '_split.json')):
                raise ValueError('Actual registered split differs from the established development split')
        entries.append(dict(task=taskname, task_budget_id=old['task_budget_id'], source_archive=str(source),
            source_archive_sha256=sha(source), training_archive=str(target / 'train.zip'),
            training_archive_sha256=sha(target / 'train.zip'), copied_training_members=copied,
            task_spec=spec, capabilities=caps, rows=len(task.y_train), splits=allowed))
    save(root / 'readiness.json', entries)
    (root / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    save(root / 'completion.json', dict(status='prepared', tasks=2, arms=20 if args.study=='route-coverage' else 12, actual_classifier_fits=0,
        real_api_requests=0, source_unchanged=sources() == frozen, elapsed_seconds=time.monotonic() - began))
    print(json.dumps(dict(status='prepared', tasks=[dict(task=e['task'], candidates=len(e['capabilities'])) for e in entries])))


def command(entry, arm, directory, args):
    planner, scheduler, coverage = settings(arm)
    result = [sys.executable, '-m', 'superagent', 'run', entry['training_archive'], '--run-dir', str(directory),
        '--planner', planner, '--scheduler', scheduler, '--portfolio', 'ab-v1', '--catalog-scope', 'full',
        '--fit-limit', '11', '--seed', '707', '--time-limit', '600', '--trial-timeout', '60']
    if planner != 'local':
        result += ['--api-key-file', str(args.api_key_file), '--budget-ledger', str(args.ledger),
            '--task-budget-id', entry['task_budget_id'], '--llm-budget-rmb', '.30',
            '--history-policy', 'incremental-research']
        result += ['--candidate-coverage',coverage]
    return result


def launch(cmd, log, env, timeout):
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as file:
        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            child.wait(timeout=15)
            raise TimeoutError('A29 process exceeded its outer limit; stopped without retry')
    return code, time.monotonic() - start


def evaluation(entry, root, frozen_model_count=12):
    folder = root / 'evaluation' / entry['task']
    folder.mkdir(parents=True)
    if sha(Path(entry['source_archive'])) != entry['source_archive_sha256']:
        raise ValueError('Original task archive changed before post-freeze validation')
    opened = []
    original = zipfile.ZipFile.read
    def no_test(archive, member, *a, **kw):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        if 'test' in name.casefold():
            raise ValueError('A29 attempted to read original formal test')
        opened.append(name)
        return original(archive, member, *a, **kw)
    with patch.object(zipfile.ZipFile, 'read', new=no_test):
        task = load_development_task(entry['source_archive'])
    x, y = task.X_val, task.y_val
    if x is None or y is None:
        raise ValueError('Designated development val is absent')
    with zipfile.ZipFile(folder / 'features.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        if entry['task'] == 'coupon':
            buffer = io.BytesIO()
            np.save(buffer, x)
            archive.writestr('C_test.npy', buffer.getvalue())
        else:
            buffer = io.StringIO(newline='')
            writer = csv.writer(buffer)
            writer.writerow(['headline_A', 'headline_B'])
            writer.writerows(x)
            archive.writestr('test.csv', buffer.getvalue())
    joblib.dump(dict(X=x, y=y, row_ids=list(range(len(y)))), folder / 'holdout.joblib')
    save(folder / 'source_access.json', dict(original_members_opened=opened, formal_test_opened=False,
        scope=f'Original designated val opened only after all {frozen_model_count} models frozen; output test-named file holds val features'))
    return folder


def execute(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    frozen = sources()
    runner_hash = sha(Path(__file__))
    observer = ROOT / 'scripts' / 'a29_worker_observer' / 'sitecustomize.py'
    observer_hash = sha(observer)
    results = []
    order=ROUTE_ORDER if args.study=='route-coverage' else ORDER
    paid_count=sum(settings(arm)[0]!='local' for _,arm in order)
    per_task_allowance=2.40 if args.study=='route-coverage' else 1.20
    deadline=datetime.fromisoformat(args.window_deadline) if args.window_deadline else None
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError('Window deadline must include a timezone')
    base_env = {**os.environ, 'PYTHONIOENCODING':'utf-8', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2',
        'MKL_NUM_THREADS':'2', 'TOKENIZERS_PARALLELISM':'false', 'AB_A29_OBSERVE_FITS':'1',
        'AB_A29_FIT_LOG': str(root / 'physical_fits.jsonl'),
        'PYTHONPATH': os.pathsep.join([str(observer.parent), str(ROOT)])}
    protocol = dict(batch='A32-route-coverage' if args.study=='route-coverage' else 'A29-six-arm-development', source_files=frozen, runner_sha256=runner_hash,
        observer_sha256=observer_hash, plan_sha256=sha(plan_path(args)), study=args.study, window_deadline=args.window_deadline,
        order=order, maximum_classifier_fits=11*len(order), maximum_api_requests=3*paid_count, per_paid_arm_allowance_rmb=.30,
        history_policy='incremental-research', course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False, automatic_retries=False, validate_only=args.validate_only)
    save(root / 'protocol.json', protocol)
    try:
        if read(args.prepared / 'protocol.json')['source_files'] != frozen:
            raise ValueError('Prepared source differs; do not reuse another version as current freeze')
        entries = {entry['task']: entry for entry in read(args.prepared / 'readiness.json')}
        commands = []
        ledger = None
        for entry in entries.values():
            if sha(Path(entry['training_archive'])) != entry['training_archive_sha256']:
                raise ValueError('Prepared train-only archive changed')
        for taskname, arm in order:
            entry = entries[taskname]
            cmd = command(entry, arm, root / (taskname + '_' + arm), args)
            if settings(arm)[0] != 'local':
                parsed = build_parser().parse_args(cmd[3:])
                ledger, _, _ = preflight(parsed)
                research_preflight(ledger, entry['task_budget_id'], per_task_allowance)
            commands.append(dict(task=taskname, arm=arm, command=cmd))
        protocol.update(entries=list(entries.values()), commands=commands, project_usage_before=ledger.project_summary())
        save(root / 'protocol.json', protocol)
        if args.validate_only:
            save(root / 'completion.json', dict(status='validated_not_executed', actual_classifier_fits=0, real_api_requests=0))
            print(json.dumps(dict(status='validated_not_executed', arms=len(commands))))
            return
        for item in commands:
            if deadline is not None and (deadline-datetime.now(timezone.utc)).total_seconds()<660:
                raise WindowLimitReached('Less than 660 seconds remain; no new path or request started')
            taskname, arm, cmd = item['task'], item['arm'], item['command']
            if sources() != frozen or sha(Path(__file__)) != runner_hash or sha(observer) != observer_hash:
                raise ValueError('Frozen production/runner/observer changed during matrix')
            directory = root / (taskname + '_' + arm)
            env = {**base_env, 'AB_A29_RUN_NAME':directory.name,
                   'SUPERAGENT_OFFLINE':'1' if settings(arm)[0] == 'local' else '0'}
            code, seconds = launch(cmd, root / (directory.name + '.log'), env, 660)
            value = read(directory / 'run_outcome.json')
            results.append(dict(task=taskname, arm=arm, outcome=value, process_seconds=seconds))
            save(root / 'checkpoint.json', results)
            counts = value['classifier_fit_accounting']
            traced = sum(row['run_name'] == directory.name for row in physical(root))
            if code or not value['requested_planner_completed']:
                from scripts.status_a29_choice_matrix import completion_failure
                raise RuntimeError('A29 path stopped without retry: ' + directory.name + ': ' + str(completion_failure(value)))
            if not (2 <= traced == counts['attempted_classifier_fits'] == counts['completed_classifier_fits'] <= 11
                    and counts['unknown_failed_attempts'] == 0 and value['new_api_requests'] <= 3):
                raise ValueError('Actual fit/request count differs or exceeded budget')
            entry = entries[taskname]
            manifest = read(directory / 'run_manifest.json')
            if manifest['source_files'] != frozen or manifest['capabilities'] != entry['capabilities']:
                raise ValueError('Path source/menu differs from matrix registration')
            for split in entry['splits']:
                if read(directory / (split + '_split.json')) != read(args.prepared / taskname / (split + '_split.json')):
                    raise ValueError('Actual split differs from matrix registration')
            selection = read(directory / 'final_selection.json')
            if sha(directory / selection['model_file']) != selection['model_sha256']:
                raise ValueError('Frozen model changed')
            results[-1].update(selection=selection, actions=read(directory / 'actions.json'))
            save(root / 'checkpoint.json', results)
            print(json.dumps(dict(task=taskname, arm=arm, status='model_frozen', fits=traced,
                requests=value['new_api_requests'], seconds=seconds, selected=value['selected_candidate'])), flush=True)
        save(root / 'all_models_frozen.json', [dict(task=r['task'], arm=r['arm'], selection=r['selection']) for r in results])
        held = {name: evaluation(entry, root, len(order)) for name, entry in entries.items()}
        for row in results:
            directory = root / (row['task'] + '_' + row['arm'])
            folder = held[row['task']]
            env = {**base_env, 'SUPERAGENT_OFFLINE':'1', 'AB_A29_RUN_NAME':directory.name}
            query = directory / 'predictions' / 'rows.csv'
            commands_export = [
                [sys.executable, '-m', 'superagent', 'predict', str(folder / 'features.zip'), '--run-dir', str(directory), '--output', str(query)],
                [sys.executable, str(ROOT / 'scripts' / 'a07_verify_export.py'), '--source-root', str(ROOT), '--run-dir', str(directory),
                 '--holdout', str(folder / 'holdout.joblib'), '--predictions', str(query), '--output', str(directory / 'verification.json')]]
            export_seconds = 0
            for index, cmd in enumerate(commands_export):
                code, seconds = launch(cmd, root / (directory.name + f'_export_{index}.log'), env, 60)
                if code:
                    raise RuntimeError('Frozen model export/verification failed: ' + directory.name)
                export_seconds += seconds
            row.update(evaluation=read(directory / 'verification.json'), export_seconds=export_seconds)
            save(root / 'results.json', results)
        fits = sum(row['outcome']['classifier_fit_accounting']['attempted_classifier_fits'] for row in results)
        if fits != len(physical(root)) or fits > 11*len(order) or sources() != frozen:
            raise ValueError('Final physical-fit accounting or frozen source differs')
        save(root / 'completion.json', dict(status='completed', paths_completed=len(results), actual_classifier_fits=fits,
            real_api_requests=sum(row['outcome']['new_api_requests'] for row in results),
            known_new_course_cost_rmb=sum(row['outcome']['known_course_cost_rmb'] for row in results),
            project_usage_after=ledger.project_summary(), source_unchanged=True, original_formal_test_opened=False,
            elapsed_seconds=time.monotonic() - began, course_budget_compliance='unverified_history', valid_for_budgeted_final_report=False))
    except Exception as error:
        save(root / 'results.json', results)
        save(root / 'completion.json', dict(status='window_limited' if isinstance(error,WindowLimitReached) else 'stopped', error=str(error), automatic_retry=False,
            completed_path_outcomes=len(results), observed_classifier_fit_entries=len(physical(root)),
            real_api_requests=sum(row['outcome']['new_api_requests'] for row in results),
            source_unchanged=sources() == frozen, elapsed_seconds=time.monotonic() - began))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'execute'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readiness', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--api-key-file', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--study',choices=['six-arm','route-coverage'],default='six-arm')
    parser.add_argument('--window-deadline')
    args = parser.parse_args()
    if args.phase == 'prepare':
        if args.readiness is None: parser.error('prepare needs --readiness')
        prepare(args)
    else:
        if any(getattr(args, name) is None for name in ('prepared', 'ledger', 'api_key_file')):
            parser.error('execute needs --prepared, --ledger, --api-key-file')
        execute(args)
