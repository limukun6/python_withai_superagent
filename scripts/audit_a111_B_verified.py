"""No-fit intake of B's published native preference and component research delivery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import read, sha

ROOT = Path(__file__).resolve().parents[1]

MENU_PROBE = '''
import json,sys
from pathlib import Path
from scripts import a38_predict_guard
from superagent.ab_run import registered_capabilities
tasks=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
result={name:{scope:registered_capabilities(row['spec'],row['rows'],scope)
              for scope in ('baseline','full')} for name,row in tasks.items()}
print(json.dumps(result,sort_keys=True))
'''

TEST_PROBE = '''
from scripts import a38_predict_guard
import unittest
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_ab_deep_preference_catalog')
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source = hashes(ROOT)
    target_source = hashes(a.target)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.target, text=True).strip()
    save(a.output/'protocol.json', dict(batch='A111', target_commit=commit, source_files=source,
         target_source_files=target_source, runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A111_B_verified_intake_plan.md'),
         actual_classifier_fits=0, real_API_requests=0, H_performance_access_allowed=False))
    try:
        if commit != 'c6584776be99a1eaee59b881047566219651edbd':
            raise ValueError('Unexpected B delivery commit')
        env = {**os.environ, 'SUPERAGENT_MINILM_WEIGHTS': str(a.assets/'minilm'),
               'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS': str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
               'SUPERAGENT_DEEP_MINILM_WEIGHTS': '', 'SUPERAGENT_AB_PREFERENCE_CATALOG': 'standard',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8'}
        env.pop('PYTHONPATH', None)
        test = subprocess.run([sys.executable, '-c', TEST_PROBE], cwd=a.target, env=env,
                              capture_output=True, text=True, encoding='utf-8', timeout=120)
        (a.output/'unit_tests.log').write_text(test.stdout+test.stderr, encoding='utf-8')
        if test.returncode:
            raise ValueError('B no-fit scope tests failed; see log')
        tasks = {k: dict(spec=v['task_spec'], rows=v['training_rows'])
                 for k, v in read(ROOT/'research/a94_portable_joint_inputs/results.json').items()}
        tasks['sms'] = dict(spec=read(ROOT/'research/a93_teacher_encoder_clarification/catalogue/results.json')['sms']['task_spec'], rows=3302)
        tasks['flowers'] = dict(spec=read(ROOT/'research/a96_flower_Q/full/run_manifest.json')['task_contract'], rows=800)
        for seed, row in read(ROOT/'research/a109_H2_canonical_inputs/results.json')['menus'].items():
            tasks['H2_'+seed] = dict(spec=row['task_spec'], rows=row['rows'])
        save(a.output/'task_metadata.json', tasks)
        menus = {}
        for name, folder in (('A', ROOT), ('B', a.target)):
            probe = subprocess.run([sys.executable, '-c', MENU_PROBE, str(a.output/'task_metadata.json')],
                                   cwd=folder, env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
            if probe.returncode:
                raise ValueError(name+' metadata probe failed: '+probe.stderr[-2000:])
            menus[name] = json.loads(probe.stdout)
        if menus['A'] != menus['B']:
            raise ValueError('Default candidate metadata changed')
        archive = a.target/'research/b_local_components_1123/runtime-source.zip'
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
        if any(Path(n).suffix.lower() not in ('.py', '.json') for n in names):
            raise ValueError('Unexpected payload in archived research runtime')
        base = a.target/'research/b_accuracy_frontier_1123'
        pair = read(base/'pair_results.json')
        usage = read(base/'pair_usage.json')
        if pair['correct'] != {'deep': 1224, 'deep-edit': 1224} or pair['actual_total_fits'] != 20:
            raise ValueError('Native pair receipts differ from handoff')
        if any(usage[arm]['attempted'] != 10 or usage[arm]['model_api_calls'] != 0 for arm in ('deep', 'deep-edit')):
            raise ValueError('Native pair accounting differs')
        component = read(a.target/'research/b_local_components_1123/results.json')
        exports = read(base/'pair_exports_frozen.json')
        if len(set(exports['predictions'].values())) != 1:
            raise ValueError('Export receipt prediction identities differ')
        if source != hashes(ROOT) or target_source != hashes(a.target):
            raise ValueError('Reviewed production source changed')
        save(a.output/'results.json', dict(target_commit=commit, default_menus_identical=menus['A'],
             native_pair=pair, native_usage=usage, native_export_receipt=exports,
             component_pair=component, runtime_archive=dict(sha256=sha(archive), files=names, executed=False),
             new_methods_integrated_into_A=False, local_deep_assets_verified=False,
             reported_9ed_case_annotation_package_received=False,
             current_C1_Task2_Bean_paths_added=0, F_star_selected=False,
             evidence_scope='Published receipts checked; only no-fit scope tests and default menu comparison rerun locally'))
        save(a.output/'completion.json', dict(status='reviewed_no_production_merge', actual_classifier_fits=0,
             real_API_requests=0, local_scope_tests_passed=2, default_task_menus_compared=len(tasks),
             default_menu_variants_compared=2*len(tasks), source_unchanged=True, target_source_unchanged=True,
             A_cumulative_fits_after=1501, elapsed_seconds=time.monotonic()-start))
        print('B native receipts checked; 2 scope tests and 7 default task menus passed; zero fits/API', flush=True)
    except Exception as error:
        save(a.output/'completion.json', dict(status='stopped', error=str(error), actual_classifier_fits=0,
             real_API_requests=0, elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
