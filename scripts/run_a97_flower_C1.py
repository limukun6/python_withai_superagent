"""Known non-identity flower cohort, six whole-loop arms, explicit reuse, one real ledger."""
import argparse
import os
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a29_choice_matrix import launch, physical, ARMS
from superagent.ab_run import sources, registered_capabilities
from superagent.ab_flash import preflight
from superagent.cli import build_parser
from superagent.research_budget import research_preflight
from tools.unattended_runtime import prevent_idle_sleep

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['L0', 'R0', 'E1', 'E0', 'R1']
SCOPE = 'tensorflow_flowers_b_cohort'


def main(a):
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    frozen = sources()
    results = []
    ledger = None
    before = None
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS='',
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets / 'mobilenet_v3_small-047dcff4.pth'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SUPERAGENT_OFFLINE='0')
    helpers = [Path(__file__), ROOT / 'scripts/a29_worker_observer/sitecustomize.py',
        ROOT / 'scripts/run_a29_choice_matrix.py', ROOT / 'scripts/campus_direct.py',
        ROOT / 'scripts/a38_predict_guard.py', ROOT / 'scripts/a07_verify_export.py']
    immutable = {str(p.relative_to(ROOT)): sha(p) for p in helpers}
    protocol = dict(batch='A97', source_files=frozen, script_hashes=immutable,
        plan_sha256=sha(ROOT / 'docs/batches/A97_flower_six_arm_C1_plan.md'),
        order=ORDER, maximum_new_classifier_fits=55, reused_L1_fits=10,
        maximum_new_API_requests=12, starting_A_fits=1385, research_allowance_rmb=1.20,
        history_policy='incremental-research', course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False, reference_L1_already_scored=True,
        environment=read(a.assets / 'environment.json'), assets=read(a.assets / 'assets.json'))
    save(root / 'protocol.json', protocol)
    try:
        if read(a.reference / 'completion.json')['status'] != 'completed':
            raise ValueError('Require completed reference')
        reference_protocol = read(a.reference / 'protocol.json')
        if reference_protocol['source_files'] != frozen:
            raise ValueError('Q source differs')
        registration = read(a.data / 'registration.json')
        if reference_protocol['registration'] != registration:
            raise ValueError('Reference input registration differs')
        split_files = list((a.reference / 'inputs').glob('*_split.json'))
        if {p.name for p in split_files} != {'primary_split.json', 'confirmation_1_split.json', 'confirmation_2_split.json'}:
            raise ValueError('Require exactly three registered partitions')
        if read(a.data / 'completion.json')['status'] != 'prepared':
            raise ValueError('Require prepared input')
        if sha(a.data / 'train.zip') != registration['training_package_sha256']:
            raise ValueError('Training input changed')
        reused = a.reference / 'full'
        manifest = read(reused / 'run_manifest.json')
        caps = registered_capabilities(manifest['task_contract'], 800, 'full')
        if len(caps) != 9 or caps != manifest['capabilities'] or manifest['source_files'] != frozen:
            raise ValueError('L1 full catalogue/source differs')
        if (manifest['fit_limit'], manifest['time_limit_seconds'], manifest['trial_timeout_seconds']) != (11, 1200, 300):
            raise ValueError('L1 budgets differ')
        reference_row = next(r for r in read(a.reference / 'results.json') if r['arm'] == 'full')
        outcome = read(reused / 'run_outcome.json')
        if reference_row['actual_classifier_fits'] != 10 or not outcome['requested_planner_completed']:
            raise ValueError('L1 fit accounting/completion differs')
        selection = read(reused / 'final_selection.json')
        if sha(reused / selection['model_file']) != selection['model_sha256']:
            raise ValueError('L1 frozen model changed')
        results.append(dict(arm='L1', run_dir=str(reused), reused=True, new_classifier_fits=0,
            reused_classifier_fits=10, outcome=outcome, selection=selection, process_seconds=None,
            reference_process_seconds=reference_row['process_seconds']))
        commands = {}
        for arm in ORDER:
            planner, scheduler = ARMS[arm]
            command = ['run', str(a.data / 'train.zip'), '--run-dir', str(root / arm),
                '--planner', planner, '--scheduler', scheduler, '--portfolio', 'ab-v1',
                '--catalog-scope', 'full', '--fit-limit', '11', '--seed', '707',
                '--time-limit', '1200', '--trial-timeout', '300', '--partial-delivery', 'checkpoint']
            if planner != 'local':
                command += ['--api-key-file', str(a.api_key_file), '--budget-ledger', str(a.ledger),
                    '--task-budget-id', SCOPE, '--llm-budget-rmb', '.30',
                    '--history-policy', 'incremental-research', '--proposal-contract', 'calibrated-v2']
                ledger, _, _ = preflight(build_parser().parse_args(command))
                research_preflight(ledger, SCOPE, 1.20)
            commands[arm] = command
        before = ledger.project_summary()
        protocol.update(commands=commands, registration=registration, capabilities=caps,
            project_usage_before=before, reference_model_sha256=selection['model_sha256'])
        save(root / 'protocol.json', protocol)
        if a.validate_only:
            save(root / 'completion.json', dict(status='validated', actual_classifier_fits=0,
                real_api_requests=0, elapsed_seconds=time.monotonic()-began))
            print('Validated five new paths plus one explicit reused path; zero fits/API', flush=True)
            return
        env = {**os.environ, 'AB_A29_OBSERVE_FITS': '1', 'AB_A29_FIT_LOG': str(root / 'physical_fits.jsonl'),
            'PYTHONPATH': os.pathsep.join([str(ROOT / 'scripts/a29_worker_observer'), str(ROOT)]),
            'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        for arm in ORDER:
            if sources() != frozen or any(sha(ROOT / p) != digest for p, digest in immutable.items()):
                raise ValueError('Frozen source/helper changed')
            if ledger.project_summary()['requests'] != before['requests'] + sum(r['outcome']['new_api_requests'] for r in results):
                raise ValueError('Unexpected concurrent ledger writer')
            command = commands[arm]
            paid = ARMS[arm][0] != 'local'
            if paid:
                preflight(build_parser().parse_args(command))
            prefix = [sys.executable, '-m', 'scripts.campus_direct', '--ip', '10.121.10.250', 'superagent'] if paid else [sys.executable, '-m', 'superagent']
            code, seconds = launch([*prefix, *command], root / (arm+'.log'), {**env, 'AB_A29_RUN_NAME': arm}, 1260)
            directory = root / arm
            if not (directory / 'run_outcome.json').exists():
                raise RuntimeError('Official path failed before outcome: '+arm)
            outcome = read(directory / 'run_outcome.json')
            count = sum(r['run_name'] == arm for r in physical(root))
            results.append(dict(arm=arm, run_dir=str(directory), reused=False, new_classifier_fits=count,
                reused_classifier_fits=0, outcome=outcome, process_seconds=seconds))
            save(root / 'checkpoint.json', results)
            if code or not outcome['requested_planner_completed']:
                raise RuntimeError('Path stopped, no retry: '+arm+': '+str(outcome.get('error')))
            accounting = outcome['classifier_fit_accounting']
            active = read(directory / 'run_manifest.json')
            if not 2 <= count == accounting['attempted_classifier_fits'] == accounting['completed_classifier_fits'] <= 11:
                raise ValueError('Actual fit count differs')
            if active['source_files'] != frozen or active['capabilities'] != caps:
                raise ValueError('Source/menu differs')
            if outcome['new_api_requests'] > 3 or (not paid and outcome['new_api_requests']):
                raise ValueError('Request allowance differs')
            for split_file in split_files:
                if read(directory / split_file.name) != read(split_file):
                    raise ValueError('Actual partition differs')
            selection = read(directory / 'final_selection.json')
            if sha(directory / selection['model_file']) != selection['model_sha256']:
                raise ValueError('Frozen model changed')
            results[-1]['selection'] = selection
            save(root / 'checkpoint.json', results)
            print(dict(arm=arm, status='model_frozen', fits=count, requests=outcome['new_api_requests'], seconds=seconds), flush=True)
        save(root / 'all_models_frozen.json', results)
        if sha(a.data / 'features.zip') != registration['feature_package_sha256'] or sha(a.data / 'holdout.joblib') != registration['holdout_sha256']:
            raise ValueError('Development holdout changed')
        for row in results:
            directory = Path(row['run_dir'])
            destination = root / 'predictions' / row['arm'] / 'rows.csv'
            commands = [[sys.executable, '-m', 'scripts.a38_predict_guard', str(a.data / 'features.zip'),
                '--run-dir', str(directory), '--output', str(destination)],
                [sys.executable, str(ROOT / 'scripts/a07_verify_export.py'), '--source-root', str(ROOT),
                 '--run-dir', str(directory), '--holdout', str(a.data / 'holdout.joblib'),
                 '--predictions', str(destination), '--output', str(root / (row['arm']+'_verification.json'))]]
            for i, command in enumerate(commands):
                code, _ = launch(command, root / (row['arm']+f'_export{i}.log'), {**env, 'AB_A29_RUN_NAME': 'inference_only'}, 300)
                if code:
                    raise RuntimeError('Post-freeze prediction failed: '+row['arm'])
            row['evaluation'] = read(root / (row['arm']+'_verification.json'))
            save(root / 'results.json', results)
        if len(physical(root)) != sum(r['new_classifier_fits'] for r in results):
            raise ValueError('Unexpected inference fit')
        after = ledger.project_summary()
        requests = sum(r['outcome']['new_api_requests'] for r in results)
        if after['requests'] != before['requests']+requests or after['unresolved_requests'] != before['unresolved_requests'] or after['pending_requests']:
            raise ValueError('Unexpected ledger change or unresolved request')
        save(root / 'completion.json', dict(status='completed', actual_classifier_fits=len(physical(root)),
            reused_classifier_fits=10, A_cumulative_fits_after=1385+len(physical(root)), real_api_requests=requests,
            elapsed_seconds=time.monotonic()-began, source_unchanged=sources() == frozen,
            project_usage_after=after, course_budget_compliance='unverified_history',
            valid_for_budgeted_final_report=False, formal_test_or_H_opened=False))
    except Exception as error:
        after = ledger.project_summary() if ledger else None
        save(root / 'completion.json', dict(status='stopped', error=str(error),
            actual_classifier_fits=len(physical(root)), A_cumulative_fits_after=1385+len(physical(root)),
            real_api_requests=after['requests']-before['requests'] if before and after else 0,
            reused_classifier_fits=10, automatic_retry=False, elapsed_seconds=time.monotonic()-began,
            project_usage_after=after, formal_test_or_H_opened=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'reference', 'assets', 'ledger', 'api-key-file', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.validate_only:
        main(a)
    else:
        with prevent_idle_sleep():
            main(a)
