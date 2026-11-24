"""No-fit runtime/menu candidate snapshot; H registration metadata only, no H samples."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'L0':('local','ab-fixed'), 'L1':('local','ab-v1'), 'R0':('raw-history','ab-fixed'),
        'R1':('raw-history','ab-v1'), 'E0':('checked-history','ab-fixed'), 'E1':('checked-history','ab-v1')}

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('assets','output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False); start=time.monotonic()
    from scripts import a38_predict_guard  # Reject known classifier fit and all network operations.
    from superagent.ab_run import sources, registered_capabilities
    from superagent.ab_sentence import SentenceFeatures
    from superagent.ab_vision import FrozenVisionFeatures
    from superagent.external_assets import model_resources, verify_resources
    from superagent.cli import build_parser
    from superagent.scheduler_bridge import fingerprint
    frozen = sources()
    save(a.output/'protocol.json', dict(batch='A101', runner_sha256=sha(Path(__file__)), source_files=frozen,
        plan_sha256=sha(ROOT/'docs/batches/A100_A101_evidence_freeze_plan.md'),
        actual_classifier_fits=0, real_API_requests=0, H_registration_metadata_only=True,
        formal_H_version_frozen=False))
    try:
        if read(a.assets/'completion.json')['status'] != 'ready': raise ValueError('Assets not ready')
        os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
            SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'))
        resources, packages = model_resources([SentenceFeatures('single_text'), FrozenVisionFeatures()])
        verify_resources(resources)
        if len(resources) != 2: raise ValueError('Require two validated encoder resource sets')
        registered = read(ROOT/'research/a94_portable_joint_inputs/results.json')
        tasks = {k:dict(spec=v['task_spec'], rows=v['training_rows'], role='known_development') for k,v in registered.items()}
        catalogue = read(ROOT/'research/a93_teacher_encoder_clarification/catalogue/results.json')
        tasks['sms'] = dict(spec=catalogue['sms']['task_spec'], rows=catalogue['sms']['full_training_rows'], role='known_development')
        photo = read(ROOT/'research/a96_flower_Q/full/run_manifest.json')
        tasks['flowers'] = dict(spec=photo['task_contract'], rows=800, role='known_development')
        h = read(ROOT/'research/a12_a16_day/a13_holdout_packages/registration.json')
        for key, row in h.items():
            tasks[key] = dict(spec=row['task_spec'], rows=row['rows'], role='sealed_metadata_only',
                archive_sha256=row['task_archive_sha256'], registered_partition_ids=list(row['partitions']))
        result = {}
        for name, row in tasks.items():
            enabled = registered_capabilities(row['spec'], row['rows'], 'full')
            os.environ.update(SUPERAGENT_MINILM_WEIGHTS='', SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS='')
            classic = registered_capabilities(row['spec'], row['rows'], 'full')
            os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),
                SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth'))
            if not enabled or not classic: raise ValueError('Missing runnable classic route: '+name)
            result[name] = dict(**row, full_capabilities=enabled, classic_capabilities=classic,
                full_menu_sha256=fingerprint(enabled), classic_menu_sha256=fingerprint(classic),
                full_candidates=len(enabled), classic_candidates=len(classic),
                enabled_encoders=[k for k,v in enabled.items() if v['resource']=='local_frozen_encoder_cpu'])
        commands = {}
        for arm,(planner,scheduler) in ARMS.items():
            command=['run','TASK_TRAIN_ONLY.zip','--planner',planner,'--scheduler',scheduler,
                '--portfolio','ab-v1','--catalog-scope','full','--fit-limit','11', '--seed','707',
                '--time-limit','1200','--trial-timeout','300','--partial-delivery','checkpoint']
            if planner!='local': command += ['--model','DeepSeek-V4.1-Flash','--proposal-contract','calibrated-v2',
                '--history-policy','incremental-research','--task-budget-id','TASK_SCOPE',
                '--api-key-file','LOCAL_KEY_PATH','--budget-ledger','EXISTING_LEDGER_PATH','--llm-budget-rmb','.30']
            parsed=build_parser().parse_args(command)
            if parsed.fit_limit!=11: raise ValueError('Fit limit changed')
            commands[arm]=command
        save(a.output/'results.json', dict(tasks=result, parsed_command_templates=commands, resources=resources,
            python=sys.version, executable=sys.executable,
            packages={name:importlib.metadata.version(name) for name in sorted(set(packages)|{'numpy','scipy','scikit-learn','joblib'})},
            readiness='Candidate runtime and legal menus only; final strategy/H schedule still require explicit freeze',
            actual_deployment_strategy_selected=False))
        if sources()!=frozen: raise ValueError('Source changed')
        save(a.output/'completion.json', dict(status='passed', tasks=len(result), parsed_arms=len(commands),
            verified_resource_sets=len(resources), actual_classifier_fits=0, real_API_requests=0,
            elapsed_seconds=time.monotonic()-start, A_cumulative_fits_after=1438,
            H_samples_or_formal_test_opened=False, formal_H_version_frozen=False))
        print({k:dict(full=v['full_candidates'], classic=v['classic_candidates'], encoders=len(v['enabled_encoders'])) for k,v in result.items()})
    except Exception as e:
        save(a.output/'completion.json', dict(status='stopped', error=str(e), actual_classifier_fits=0,
            real_API_requests=0, elapsed_seconds=time.monotonic()-start)); raise
