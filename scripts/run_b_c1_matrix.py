"""B's two six-arm blocks on the official entry, with persistent arm identity and deferred scoring."""
import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.materialize_b_c1_rows import load_original
from scripts.prepare_b_c1 import ARMS, TASKS, ORDER_SALT
from superagent import ab_flash, ab_run
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import content_digest, registered_partitioner


ROOT = rec.ROOT
REFERENCE = ROOT / 'research/bc1_preparation'
PORTABLE = ROOT / 'research/bp_02_materialization/environment_1/results.json'
TASK_BUDGET_IDS = {'headlines':'task2', 'bean':'dry_bean'}


def ordered(task):
    return sorted(ARMS, key=lambda arm: rec.sha(f'{ORDER_SALT}/{task}/{arm}'.encode()))


def options(taskfile, arm, destination):
    planner, allocation = ARMS[arm]
    return ['run',str(taskfile),'--planner',planner,'--history-policy','strict','--scheduler','ab-fixed' if allocation=='fixed' else 'ab-v1',
            '--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11','--seed','707','--time-limit','1200','--trial-timeout','300',
            '--run-dir',str(destination)]


def freeze_inputs(portable):
    old, packets = rec.read(REFERENCE/'results.json'), rec.read(PORTABLE)
    tasks = {}
    for name in TASKS:
        folder, expected = portable/name, packets[name]['canonical_files']
        train, inner = folder/'optimizer/train.zip', folder/'optimizer/inner_splits.json'
        for file in (train, inner, folder/'delivery/features.zip'):
            if ab_run.sha(file) != expected[file.name]:
                raise ValueError('Registered input changed: '+name+'/'+file.name)
        data = load_development_task(train)
        catalog = ab_run.registered_capabilities(TaskSpec.from_task(data).record(), len(data.y_train), 'full')
        reference = old['tasks'][name]
        if (data.X_val is not None or data.y_val is not None or list(catalog) != reference['candidate_ids'] or
                rec.digest(catalog) != reference['catalog_sha256'] or len(data.y_train) != packets[name]['training_rows']):
            raise ValueError('Training packet or shared method catalogue differs on '+name)
        registered_partitioner(data,inner,expected['inner_splits.json'])
        splits = {p['id']:{k:p[k] for k in ('id','seed','row_ids_sha256','label_sha256')}
                  for p in rec.read(inner)['splits']}
        arms = ordered(name)
        if arms != [b['arm'] for b in old['scheduled_blocks'][name]]:
            raise ValueError('Preregistered order changed on '+name)
        for arm in arms:
            parsed = build_parser().parse_args(options(train,arm,Path('unstarted')))
            if ab_run.validate_options(parsed) != 11:
                raise ValueError('Formal entry did not accept the registered cap')
        tasks[name] = dict(order=arms,files=expected,catalog_scope='full',history_policy='strict',task_budget_id=TASK_BUDGET_IDS[name],
                           catalog_sha256=rec.digest(catalog),anchor=next(iter(catalog)),
                           splits=splits,held_rows=packets[name]['held_rows'],held_content=packets[name]['held_content_sha256'])
    helpers = ('scripts/run_b_c1_matrix.py','tools/ab_registered_rows.py','tools/ab_registered_run.py')
    return dict(batch='BC1',source=rec.digest(ab_run.sources()),runtime=pilot.environment(),tasks=tasks,
                research_files={p:ab_run.sha(ROOT/p) for p in helpers},fit_limit=11,wall_seconds=1200,trial_seconds=300,
                local_request_cap=0,flash_request_cap=3)


def verify_strict_history(manifest, outcome, planner):
    required = 'no_new_llm_calls' if planner=='local' else 'verified_local_ledger'
    if (manifest.get('history_policy')!='strict' or outcome.get('history_policy')!='strict'
            or outcome.get('course_budget_compliance')!=required or outcome.get('valid_for_budgeted_final_report') is False):
        raise ValueError('Run did not use the registered strict budget history policy')
    if planner!='local':
        after=outcome.get('project_usage_after_run') or {}
        total=after.get('complete_course_cost_rmb')
        if (after.get('unresolved_requests')!=0 or after.get('pending_requests')!=0 or
                type(total) not in (int,float) or not math.isfinite(total) or total<0
                or outcome.get('complete_project_course_cost_rmb')!=total):
            raise ValueError('The strict project budget receipt is missing or inconsistent')


def verify_run(run, source, task, arm, *, funding=None):
    manifest, result = rec.read(run/'run_manifest.json'), rec.read(run/'run_outcome.json')
    fits, chosen = rec.read(run/'classifier_fits.json'), rec.read(run/'final_selection.json')
    planner, allocation = ARMS[arm]
    scheduler = 'ab-fixed' if allocation == 'fixed' else 'ab-v1'
    cap = 0 if planner == 'local' else 3
    if not (manifest['entrypoint']=='superagent run' and rec.digest(manifest['source_files'])==source and
            ('catalog_scope' not in task or manifest.get('catalog_scope')==task['catalog_scope']) and
            manifest['task_archive_sha256']==task['files']['train.zip'] and rec.digest(manifest['capabilities'])==task['catalog_sha256'] and
            manifest['fit_limit']==11 and manifest['time_limit_seconds']==1200 and manifest['trial_timeout_seconds']==300 and
            manifest['seed']==707 and manifest['planner_mode']==result['planner_mode']==planner and
            manifest['scheduler']==result['scheduler']==scheduler and manifest['llm_request_limit']==cap):
        raise ValueError('Run identity, catalogue or budgets differ from the registered arm')
    if 'history_policy' in task:
        if task['history_policy']!='strict':
            raise ValueError('A formal C1 task requires the registered strict budget history policy')
        verify_strict_history(manifest,result,planner)
    if not (result['status']=='completed' and result['requested_planner_completed'] and result['full_refit_completed'] and
            result['source_unchanged'] and not result['formal_test_accessed'] and result['validation_score'] is None and
            result['unresolved_calls']==0 and not result['automatic_retries']):
        raise ValueError('Incomplete or unaccounted official arm')
    elapsed = result['elapsed_seconds']
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:
        raise ValueError('Actual training and selection time is missing or invalid')
    tally = result['classifier_fit_accounting']
    if (any(tally[k]!=fits[k] for k in ('attempted_classifier_fits','completed_classifier_fits','unknown_failed_attempts')) or
            fits['unknown_failed_attempts'] or not 1 < fits['attempted_classifier_fits']==fits['completed_classifier_fits']<=11 or
            any(sum(row[k] for row in fits['workers'])!=fits[k] for k in ('attempted_classifier_fits','completed_classifier_fits'))):
        raise ValueError('Physical worker fits disagree')
    for split, expected in task['splits'].items():
        actual = rec.read(run/(split+'_split.json'))
        if (any(actual[k]!=expected[k] for k in ('id','seed','row_ids_sha256','label_sha256')) or
                actual['evaluation_payload_sha256']!=ab_run.sha(run/(split+'.joblib'))):
            raise ValueError('Actual partition identity differs: '+split)
    first = rec.read(run/'actions.json')[0]
    if first['action']!='explore' or first['candidate_id']!=task['anchor']:
        raise ValueError('Run did not start from the shared anchor')
    usage = [json.loads(line) for line in (run/'llm_usage.jsonl').read_text().splitlines()]
    calls = [row for row in usage if row['event']=='llm_call']
    if (any(row.get('usage_known') is False for row in usage) or not len(calls)==result['new_api_requests']<=cap or
            not math.isclose(sum(row['course_cost_rmb'] for row in calls),result['complete_course_cost_rmb'],abs_tol=1e-9)):
        raise ValueError('Measured API use differs from the outcome')
    if planner!='local':
        before, after = manifest['task_budget_before_run'],result['task_budget_after_run']
        if (funding is None or not after['accounting_complete'] or after['limit_exceeded'] or
                before['task_budget_id']!=funding['task_id'] or after['task_budget_id']!=funding['task_id'] or
                manifest['incremental_run_allowance_rmb']!=funding['allowance'] or
                not all(c.get('persistent_settlement_complete') for c in calls)):
            raise ValueError('Cumulative task identity or accounting is incomplete')
    if not (chosen['refit'] and not chosen['test_accessed'] and chosen['family']==result['selected_candidate'] and
            ab_run.sha(run/chosen['model_file'])==chosen['model_sha256'] and ab_run.sha(run/'task_spec.json')==chosen['task_spec_sha256']):
        raise ValueError('Delivered model or its schema differs from the frozen selection')
    return dict(model_sha256=chosen['model_sha256'],candidate=chosen['family'],fits=fits['completed_classifier_fits'],
                requests=len(calls),course_rmb=result['complete_course_cost_rmb'],elapsed_seconds=elapsed)


def claim_or_resume(folder, intent):
    folder.parent.mkdir(parents=True,exist_ok=True)
    try:
        folder.mkdir()
    except FileExistsError:
        if not (folder/'intent.json').is_file() or rec.read(folder/'intent.json')!=intent:
            raise ValueError('Existing arm has no matching launch intent')
        if not (folder/'official/run_outcome.json').is_file():
            raise ValueError('Arm already started and has no final outcome; no new process launched')
        return False
    rec.write(folder/'intent.json',intent)
    return True


def location(matrix, task, arm):
    return matrix/'arms'/task/arm


def frozen_models(matrix, protocol):
    records = {}
    identity = rec.digest(protocol)
    for name, task in protocol['tasks'].items():
        records[name] = {}
        for arm in task['order']:
            folder = location(matrix,name,arm)
            if not (folder/'official/run_outcome.json').is_file():
                raise ValueError('Scoring requires every frozen model; missing '+name+'/'+arm)
            if rec.read(folder/'intent.json')!=dict(protocol=identity,task=name,arm=arm):
                raise ValueError('Arm provenance differs from the frozen matrix')
            funding = None
            if ARMS[arm][0]!='local':
                finance = rec.read(matrix/'budget_context.json')
                if finance['task_ids'][name]!=task['task_budget_id']:
                    raise ValueError('Budget context differs from the registered cumulative task ID')
                funding = dict(task_id=finance['task_ids'][name],allowance=finance['per_arm_rmb'])
            records[name][arm] = verify_run(folder/'official',protocol['source'],task,arm,funding=funding)
    return records


def run_matrix(args, protocol):
    ids = {'headlines':args.headlines_id,'bean':args.bean_id}
    if ids!={name:task['task_budget_id'] for name,task in protocol['tasks'].items()} or ids!=TASK_BUDGET_IDS:
        raise ValueError('Use the registered cumulative task IDs: headlines=task2 and bean=dry_bean')
    finance = dict(ledger=str(args.ledger.resolve()),task_ids=ids,per_arm_rmb=args.allowance)
    context = args.matrix/'budget_context.json'
    if context.exists() and rec.read(context)!=finance:
        raise ValueError('Budget ledger, task IDs or per-arm allowance changed')
    def paid(name):
        return ['--api-key-file',str(args.api_key_file.resolve()),'--budget-ledger',finance['ledger'],
                '--task-budget-id',ids[name],'--llm-budget-rmb',str(args.allowance)]
    for name, task in protocol['tasks'].items():
        remaining = [a for a in task['order'] if ARMS[a][0]!='local' and
                     not (location(args.matrix,name,a)/'official/run_outcome.json').is_file()]
        if remaining:
            probe = build_parser().parse_args(options('unopened',remaining[0],Path('unstarted'))+paid(name))
            ledger,allowance,_ = ab_flash.preflight(probe)
            ledger.preflight(ids[name],allowance*len(remaining))
    if not context.exists():
        with context.open('x',encoding='utf-8') as f:
            json.dump(finance,f)
    identity = rec.digest(protocol)
    for name, task in protocol['tasks'].items():
        folder = args.portable/name/'optimizer'
        for arm in task['order']:
            base = location(args.matrix,name,arm)
            actual = base/'official'
            if claim_or_resume(base,dict(protocol=identity,task=name,arm=arm)):
                command = options(folder/'train.zip',arm,actual)
                if ARMS[arm][0]!='local':
                    command += paid(name)
                with (base/'process.log').open('x') as log:
                    subprocess.run([sys.executable,'-m','tools.ab_registered_run','--split-manifest',str(folder/'inner_splits.json'),
                        '--split-sha256',task['files']['inner_splits.json'],'--',*command],cwd=ROOT,stdout=log,
                        stderr=subprocess.STDOUT,check=True,timeout=protocol['wall_seconds']+90)
            verify_run(actual,protocol['source'],task,arm,funding=dict(task_id=ids[name],allowance=args.allowance))
    rec.write(args.matrix/'all_models_frozen.json',frozen_models(args.matrix,protocol))


def read_export(path, expected_rows):
    with path.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    manifest = rec.read(path.parent/'prediction_manifest.json')
    if (manifest['test_labels_read'] is not False or manifest['prediction_count']!=expected_rows or
            [row['row_id'] for row in rows]!=list(map(str,range(expected_rows)))):
        raise ValueError('Exported row count, identities or label access differ')
    return [row['prediction'] for row in rows]


def grade_task(name, portable, task, original_packet, predictions):
    reference = rec.read(REFERENCE/'results.json')['tasks'][name]
    typed = load_development_task(portable/'optimizer/train.zip')
    metric = MetricSpec.for_task(typed)
    original,_ = load_original(name,original_packet,reference['source'])
    identity = gzip.decompress((REFERENCE/name/'row_ids.json.gz').read_bytes())
    if rec.sha(identity)!=reference['row_ids_sha256']:
        raise ValueError('Source indices differ from registration')
    truth = original.y_train[json.loads(identity)['outer']['held']]
    query = load_prediction_features(portable/'delivery/features.zip',typed.kind,typed.spec)
    if content_digest(query,truth)!=task['held_content']:
        raise ValueError('Outer sources, labels or row order differ')
    mapping = {str(label):label for label in metric.labels}
    return {arm:metric.evaluate(truth,[mapping[p] for p in column]) for arm,column in predictions.items()}


def score(args, protocol):
    models = frozen_models(args.matrix,protocol)
    if rec.read(args.matrix/'all_models_frozen.json')!=models:
        raise ValueError('Frozen model set changed')
    args.output.mkdir(parents=True,exist_ok=False)
    exported, csv_hashes = {}, {}
    for name, task in protocol['tasks'].items():
        exported[name],csv_hashes[name] = {},{}
        for arm in task['order']:
            target = args.output/name/arm/'rows.csv'
            subprocess.run([sys.executable,'-m','superagent','predict',str(args.portable/name/'delivery/features.zip'),
                    '--run-dir',str(location(args.matrix,name,arm)/'official'),'--output',str(target)],cwd=ROOT,check=True)
            exported[name][arm] = read_export(target,task['held_rows'])
            csv_hashes[name][arm] = ab_run.sha(target)
    scores = {name:grade_task(name,args.portable/name,task,getattr(args,name),exported[name])
              for name,task in protocol['tasks'].items()}
    rec.write(args.output/'results.json',dict(protocol=rec.digest(protocol),models=models,prediction_sha256=csv_hashes,scores=scores))
    rec.write(args.output/'completion.json',dict(status='scored',paths=12,training_or_model_api_requests=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    for mode in ('prepare','run','score'):
        p = modes.add_parser(mode)
        for flag in ('matrix','portable','weights'):
            p.add_argument('--'+flag,type=Path,required=True)
        if mode=='run':
            for flag in ('api-key-file','ledger'):
                p.add_argument('--'+flag,type=Path,required=True)
            for name,identifier in TASK_BUDGET_IDS.items():
                p.add_argument('--'+name+'-id',default=identifier,choices=[identifier])
            p.add_argument('--allowance',type=float,default=.30)
        if mode=='score':
            for flag in ('headlines','bean','output'):
                p.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    args.portable,args.matrix = args.portable.resolve(),args.matrix.resolve()
    os.environ['SUPERAGENT_MINILM_WEIGHTS'] = str(args.weights.resolve())
    protocol = freeze_inputs(args.portable)
    if args.mode=='prepare':
        args.matrix.mkdir(parents=True,exist_ok=False)
        rec.write(args.matrix/'protocol.json',protocol)
        return
    if rec.read(args.matrix/'protocol.json')!=protocol:
        raise ValueError('Production, environment, runner or inputs differ from the registered matrix')
    if args.mode=='run':
        run_matrix(args,protocol)
    else:
        score(args,protocol)


if __name__=='__main__':
    main()
