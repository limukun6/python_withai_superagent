"""Exercise actual registered-row CLI entry up to, but never into, H2 training."""
import argparse
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'L0': ('local', 'ab-fixed'), 'L1': ('local', 'ab-v1'),
        'R0': ('raw-history', 'ab-fixed'), 'R1': ('raw-history', 'ab-v1'),
        'E0': ('checked-history', 'ab-fixed'), 'E1': ('checked-history', 'ab-v1')}


class BeforeTraining(BaseException):
    """Intentional interception outside the production error/recovery handler."""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('inputs', 'assets', 'output'):
        p.add_argument('--'+key, required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    from scripts import a38_predict_guard  # Known classifier fits and all network forbidden.
    from superagent import ab_run
    from superagent.cli import build_parser
    from tools import ab_registered_run
    source = ab_run.sources()
    save(a.output/'protocol.json', dict(batch='A110', source_files=source,
         runner_sha256=sha(Path(__file__)), plan_sha256=sha(ROOT/'docs/batches/A110_H2_official_entry_plan.md'),
         input_receipt_sha256=sha(a.inputs/'completion.json'),
         registration_sha256=sha(ROOT/'research/bh_03_training_inputs/results.json'),
         actual_classifier_fits=0, real_API_requests=0, holdout_image_access_allowed=False,
         final_H_freeze=False, flash_handler_execution_allowed=False))
    try:
        if read(a.inputs/'completion.json')['status'] != 'prepared':
            raise ValueError('H2 training inputs are not prepared')
        os.environ.update(SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        expected = read(ROOT/'research/bh_03_training_inputs/results.json')
        tasks = {}
        original_freeze = ab_run.freeze_partitions
        for seed, registration in expected.items():
            folder = a.inputs/'training'/seed
            if (sha(folder/'train.zip') != registration['train_archive_sha256'] or
                    sha(folder/'inner_splits.json') != registration['inner_manifest_sha256']):
                raise ValueError('Registered input identity differs')
            commands, probes = {}, {}
            for arm, (planner, scheduler) in ARMS.items():
                directory = a.output/(seed+'_'+arm)
                command = ['run', str(folder/'train.zip'), '--run-dir', str(directory), '--planner', planner,
                           '--scheduler', scheduler, '--portfolio', 'ab-v1', '--catalog-scope', 'full',
                           '--fit-limit', '11', '--seed', '707', '--time-limit', '1200',
                           '--trial-timeout', '300', '--partial-delivery', 'checkpoint']
                if planner != 'local':
                    command += ['--model', 'DeepSeek-V4.1-Flash', '--api-key-file', 'LOCAL_KEY_PATH',
                                '--budget-ledger', 'EXISTING_SHARED_LEDGER_PATH', '--task-budget-id', 'h2_image',
                                '--llm-budget-rmb', '.30', '--history-policy', 'incremental-research',
                                '--proposal-contract', 'calibrated-v2']
                parsed = build_parser().parse_args(command)
                if ab_run.validate_options(parsed) != 11:
                    raise ValueError('Incorrect fit limit')
                commands[arm] = [sys.executable, '-m', 'tools.ab_registered_run', '--split-manifest',
                                 str(folder/'inner_splits.json'), '--split-sha256',
                                 registration['inner_manifest_sha256'], '--', *command]
                if planner != 'local':
                    probes[arm] = dict(stage='parameters_validated_only', real_flash_preflight=False)
                    continue

                def stop_at_partitions(task, output, inner_seed, metric):
                    if task.X_val is not None or task.y_val is not None:
                        raise ValueError('Outer validation reached the runner')
                    trusted, actual = original_freeze(task, output, inner_seed, metric)
                    expected_allowed = {key: {k: v for k, v in row.items()
                                              if k not in ('train_rows', 'internal_eval_rows')}
                                        for key, row in registration['splits'].items()}
                    if actual != expected_allowed:
                        raise ValueError('Actual official CLI partitions differ')
                    for split in actual:
                        record = read(output/(split+'_split.json'))
                        if len(record['train_indices']) != 2000 or len(record['search_indices']) != 500:
                            raise ValueError('Official CLI row counts differ')
                    raise BeforeTraining()

                with patch.object(ab_run, 'freeze_partitions', stop_at_partitions):
                    try:
                        ab_registered_run.run(folder/'inner_splits.json', registration['inner_manifest_sha256'], command)
                    except BeforeTraining:
                        pass
                    else:
                        raise RuntimeError('CLI did not reach expected training boundary')
                manifest = read(directory/'run_manifest.json')
                if len(manifest['capabilities']) != 9 or manifest['source_files'] != source:
                    raise ValueError('Unexpected candidate directory/source')
                if any(directory.glob('trial_*.joblib')) or (directory/'final_selection.json').exists():
                    raise ValueError('Unexpected model produced')
                probes[arm] = dict(stage='official_CLI_stopped_before_executor',
                                  actual_partitions_match_B=True, candidates=9, training_rows=2500,
                                  manifest_sha256=sha(directory/'run_manifest.json'))
                print(dict(seed=seed, arm=arm, stage='before_training', partitions_match=True), flush=True)
            tasks[seed] = dict(commands=commands, probes=probes)
        if ab_run.sources() != source:
            raise ValueError('Production source changed')
        save(a.output/'results.json', dict(tasks=tasks, readiness='training_entry_checked_not_final_H_freeze',
             required_before_formal_H=['Task2/Bean current five-task C1 evidence and F* selection',
                                      'shared frozen source/prompt/catalogue/resources and registered policy',
                                      'confirm B H2 run ownership to prevent duplicated work',
                                      'actual shared ledger sole-writer window and paid allowance'],
             no_H_performance_observed=True))
        save(a.output/'completion.json', dict(status='entry_checked_awaiting_formal_freeze',
             actual_classifier_fits=0, real_API_requests=0, model_predictions=0,
             local_CLI_training_boundaries_checked=4, flash_parameter_sets_checked=8,
             holdout_image_members_opened=0, source_unchanged=True,
             A_cumulative_fits_after=1501, elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json', dict(status='stopped', error=str(error),
             actual_classifier_fits=0, real_API_requests=0, elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
