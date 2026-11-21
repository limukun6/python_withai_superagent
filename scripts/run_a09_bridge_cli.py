"""Offline acceptance through the official CLI. Never overwrites an earlier run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_b1_ab_acceptance as fixture
from superagent.ab_run import sources


def run(directory, existing=None):
    directory.mkdir(parents=True, exist_ok=False)
    fixture.offline(directory)
    start = time.monotonic()
    results = {}
    frozen = sources()
    fixture.write(directory / 'protocol.json', dict(batch='A09', b_handoff='a787a287147288ce0e853bcdce18a3e49b0fc210',
        source_files=frozen, driver_sha256=fixture.sha(Path(__file__).read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT / 'docs/batches/A09_B_bridge_cli_plan.md').read_bytes()),
        input_scope='same anonymous synthetic numeric fixture as B1-04',
        data_seed=932, query_seed=975, split_seeds=[41,142,243], training_rows=256, query_rows=31,
        fit_limit_including_refit=11, real_api_request_limit=0, trial_timeout_seconds=65, time_limit_seconds=240))
    try:
        if existing is not None:
            existing = existing.resolve()
            original = json.loads((existing / 'protocol.json').read_text(encoding='utf-8'))
            if original['source_files'] != frozen:
                raise ValueError('Cannot verify reused run under changed production source')
            output = existing / 'official_run'
            import numpy as np
            query = np.random.default_rng(975).normal(size=(31, 8))
            origin = existing
            fixture.write(directory / 'reuse.json', dict(source=str(existing),
                source_protocol_sha256=fixture.sha((existing / 'protocol.json').read_bytes()),
                note='Revalidate existing completed official run; no additional training'))
        else:
            _, query = fixture.packages(directory)
            output = directory / 'official_run'
            origin = directory
        command = [sys.executable, '-m', 'superagent', 'run', str(directory / 'train.zip'),
            '--run-dir', str(output), '--scheduler', 'ab-v1', '--portfolio', 'ab-v1',
            '--fit-limit', '11', '--seed', '41', '--time-limit', '240', '--trial-timeout', '65']
        fixture.write(directory / 'command.json', command)
        completed = None
        if existing is None:
            completed = subprocess.run(command, cwd=fixture.ROOT, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace', timeout=300)
            (directory / 'cli_stdout.log').write_text(completed.stdout, encoding='utf-8')
            (directory / 'cli_stderr.log').write_text(completed.stderr, encoding='utf-8')
        outcome = json.loads((output / 'run_outcome.json').read_text(encoding='utf-8'))
        results['outcome'] = outcome
        if (completed is not None and completed.returncode) or not outcome['requested_planner_completed']:
            raise RuntimeError('Official CLI failed: ' + str(outcome.get('error')))
        trace = fixture.trace(origin)
        fits = outcome['classifier_fit_accounting']
        if not (2 <= len(trace) <= 11 and len(trace) == fits['attempted_classifier_fits'] == fits['completed_classifier_fits']):
            raise ValueError('Independent actual fit observation disagrees with official bounded path')
        if fits['unknown_failed_attempts'] or outcome['new_api_requests']:
            raise ValueError('Acceptance requires known fits and zero API requests')
        counts = outcome['action_counts']
        if not (counts.get('explore',0) > 0 and counts.get('confirm',0) > 0 and counts.get('stop') == 1
                and outcome['full_refit_completed'] and len(trace) == counts['explore'] + counts['confirm'] + 1):
            raise ValueError('Anonymous path failed three-action coverage or final refit accounting')
        # The production predictor receives an unlabeled query package in two new processes.
        import shutil
        shutil.copy2(origin / 'test.zip', output / 'test.zip')
        prediction = fixture.predict_twice(output, query)
        results.update(physical_fit_calls=trace, prediction=prediction,
                       actions=json.loads((output / 'actions.json').read_text(encoding='utf-8')),
                       source_unchanged=sources()==frozen,
                       elapsed_seconds=time.monotonic()-start)
        if not results['source_unchanged']:
            raise ValueError('Production source changed during acceptance')
        fixture.write(directory / 'results.json', results)
        fixture.write(directory / 'completion.json', dict(status='passed', official_run_cli_exercised=True,
            actual_classifier_fits=0 if existing else len(trace), reused_path_classifier_fits=len(trace) if existing else 0,
            action_counts=outcome['action_counts'], api_requests=0,
            fresh_prediction_processes=2, matching_rows=31, elapsed_seconds=results['elapsed_seconds'],
            scope='synthetic workflow acceptance; no real-task efficacy claim'))
    except Exception as error:
        fixture.write(directory / 'results.json', results)
        fixture.write(directory / 'completion.json', dict(status='failed', error=str(error),
            actual_classifier_fits=len(fixture.trace(directory)), api_requests=0, elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--verify-existing', type=Path)
    args = parser.parse_args()
    run(args.directory.resolve(), args.verify_existing)
