"""Run meaningful no-fit native integration checks and compare existing default menus."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.a_batch_common import hashes, save
from scripts.audit_a111_B_verified import MENU_PROBE
from scripts.run_a10_real_task_pilot import sha

ROOT = Path(__file__).resolve().parents[1]
TESTS = ['tests.test_a112_native_scope', 'tests.test_ab_text_kernel',
         'tests.test_external_assets', 'tests.test_ab_vision']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('target', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    frozen = hashes(a.target)
    save(a.output/'protocol.json', dict(batch='A112-check', target_source_files=frozen,
         runner_sha256=sha(Path(__file__)), test_modules=TESTS,
         plan_sha256=sha(ROOT/'docs/batches/A112_A114_native_and_H1_queue_plan.md'),
         actual_classifier_fits=0, real_API_requests=0))
    try:
        env = {**os.environ, 'SUPERAGENT_MINILM_WEIGHTS': str(a.assets/'minilm'),
               'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS': str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
               'SUPERAGENT_DEEP_MINILM_WEIGHTS': '', 'SUPERAGENT_AB_PREFERENCE_CATALOG': 'standard',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8'}
        env.pop('PYTHONPATH', None)
        code = ('from scripts import a38_predict_guard\nimport unittest\n'
                'suite=unittest.defaultTestLoader.loadTestsFromNames('+repr(TESTS)+')\n'
                'result=unittest.TextTestRunner(verbosity=2).run(suite)\n'
                'raise SystemExit(0 if result.wasSuccessful() else 1)')
        tests = subprocess.run([sys.executable, '-c', code], cwd=a.target, env=env,
                               capture_output=True, text=True, encoding='utf-8', timeout=180)
        (a.output/'tests.log').write_text(tests.stdout+tests.stderr, encoding='utf-8')
        if tests.returncode:
            raise ValueError('Native tests failed; see tests.log')
        probe = subprocess.run([sys.executable, '-c', MENU_PROBE,
                                str(ROOT/'research/a111_B_verified_intake/task_metadata.json')],
                               cwd=a.target, env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
        if probe.returncode:
            raise ValueError('Default menu probe failed: '+probe.stderr[-2000:])
        menus = json.loads(probe.stdout)
        expected = json.loads((ROOT/'research/a111_B_verified_intake/results.json').read_text(encoding='utf-8'))['default_menus_identical']
        if menus != expected:
            raise ValueError('Default menu changed')
        if hashes(a.target) != frozen:
            raise ValueError('Source changed during tests')
        save(a.output/'results.json', dict(default_menus=menus, native_catalog_modes=['standard','deep','deep-edit'],
             component_controller_integrated=False, default_menus_preserved=True,
             actual_deep_asset_inference_tested=False))
        save(a.output/'completion.json', dict(status='passed', actual_classifier_fits=0, real_API_requests=0,
             source_unchanged=True, default_tasks=len(menus), no_fit_test_modules=len(TESTS),
             A_cumulative_fits_after=1501, elapsed_seconds=time.monotonic()-start))
        print('Native integration no-fit tests and seven default menus passed', flush=True)
    except Exception as error:
        save(a.output/'completion.json', dict(status='stopped', error=str(error), actual_classifier_fits=0,
             real_API_requests=0, elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
