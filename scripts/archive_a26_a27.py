"""Archive method acceptance, one Flash failure, and a transparent historical fit-count correction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from superagent.ab_catalog import capabilities_for
from superagent.task_budget import TaskBudgetLedger


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = args.runs_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    folders = ['a26_symmetric_checks', 'a26_symmetric_checks_fixed', 'a26_symmetric_checks_final',
        'a26_symmetric_cli', 'a26_observer_validation', 'a27_flash_preflight', 'a27_flash_preflight_fixed',
        'a27_flash_once', 'a27_readonly_network', 'a27_no_dispatch_review']
    allowed = {'protocol.json', 'completion.json', 'results.json', 'fit_observations.json', 'tests.log',
        'physical_fits.jsonl', 'command.json', 'proof.json', 'budget_before.json', 'events.jsonl', 'llm_usage.jsonl'}
    for folder in folders:
        target = output / folder
        target.mkdir()
        for source in (runs / folder).iterdir():
            if source.is_file() and source.name in allowed:
                shutil.copy2(source, target / source.name)
                inventory.append(dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    original = runs / 'a26_symmetric_cli' / 'official_run'
    target = output / 'a26_symmetric_cli' / 'official_run'
    target.mkdir()
    for name in ('run_manifest.json', 'run_outcome.json', 'classifier_fits.json', 'actions.json', 'history.json', 'final_selection.json'):
        shutil.copy2(original / name, target / name)
    # Registration contains a frozen TaskSpec; do not open the H archives or train on H data.
    registration = read(runs / 'a13_holdout_packages' / 'registration.json')
    H = {}
    for name, record in registration.items():
        caps = capabilities_for(record['task_spec'], train_rows=record['rows'])
        H[name] = dict(task_archive_sha256=record['task_archive_sha256'], rows=record['rows'],
            capabilities=caps, executable_candidates=len(caps), H_data_opened=False, model_evaluated=False, score=None)
    save(output / 'H_capability_update.json', H)
    correction = []
    test_names = ['image_shape_and_duplicate_grouping_are_preserved',
        'numeric_multiclass_pipeline_preserves_labels_and_saved_predictions',
        'mixed_schema_survives_slicing_missing_numeric_and_unseen_category',
        'single_text_has_its_own_multiclass_route_and_normalized_duplicate_groups']
    for directory in sorted(runs.glob('a*')):
        protocol, completion, log = (directory / n for n in ('protocol.json', 'completion.json', 'tests.log'))
        if not all(p.is_file() for p in (protocol, completion, log)):
            continue
        config, result = read(protocol), read(completion)
        if config.get('batch') != 'A09-interface-regression' or result.get('actual_classifier_fits') != 0:
            continue
        if 'tests.test_general_schema' not in config.get('suites', []):
            continue
        text = log.read_text(encoding='utf-8')
        matched = [name for name in test_names if f'test_{name} (tests.test_general_schema.SchemaTests.test_{name}) ... ok' in text]
        if len(matched) != 4:
            raise ValueError('Cannot reconstruct all four original fit paths: ' + str(directory))
        correction.append(dict(directory=directory.name, originally_reported_fits=0, reconstructed_fits=4,
            completed_fit_test_paths=matched, original_files_modified=False,
            protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
            tests_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            evidence_basis='Successful original test paths and unchanged fitting test code; not contemporaneous physical instrumentation'))
    recovered = sum(row['reconstructed_fits'] for row in correction)
    actual_new = sum(read(runs / folder / 'completion.json')['actual_classifier_fits']
        for folder in folders if folder.startswith('a26_'))
    save(output / 'fit_count_correction.json', dict(historical=correction, historical_added_fits=recovered,
        current_A26_actual_fits=actual_new, previous_window_reported_fits=383,
        corrected_tracked_window_fits=383 + recovered + actual_new,
        scope='A window only; earlier project and teammate use not claimed fully audited', original_logs_modified=False))
    save(output / 'current_usage.json', TaskBudgetLedger(runs / 'a23_real_local_history' / 'ledger.sqlite').project_summary())
    save(output / 'archive_manifest.json', dict(files=inventory, contains_raw_data_models_credentials_sqlite=False,
        new_api_requests=0, new_classifier_fits=0))
    print(json.dumps(dict(status='archived', files=len(inventory), historical_added_fits=recovered,
        A26_actual_fits=actual_new, H_candidates={k: v['executable_candidates'] for k, v in H.items()})))
