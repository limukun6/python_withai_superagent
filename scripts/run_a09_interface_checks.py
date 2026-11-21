"""Persist no-training interface regression for the A09 handoff."""
import argparse
import contextlib
from pathlib import Path
import unittest

from scripts.a_batch_common import Journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--include-fixed', action='store_true')
    parser.add_argument('--include-flash', action='store_true')
    parser.add_argument('--include-schema', action='store_true')
    args = parser.parse_args()
    suites = ['tests.test_ab_official_cli', 'tests.test_ab_bridge_adapter', 'tests.test_ab_scheduler',
              'tests.test_ab_comparison', 'tests.test_ab_confirmation_budget',
              'tests.test_unified_cli.UnifiedCLITests']
    if args.include_fixed:
        suites.append('tests.test_ab_fixed_schedule')
    if args.include_flash:
        suites.append('tests.test_ab_flash')
    if args.include_schema:
        suites.extend(['tests.test_symmetric_input','tests.test_general_schema','tests.test_native_schema','tests.test_image_files'])
    journal = Journal(args.directory, Path(__file__).resolve().parents[1],
                      dict(batch='A09-interface-regression', suites=suites,
                           scope='Mocks or pre-fit refusals only; no real training or API'))
    with (journal.out / 'tests.log').open('w', encoding='utf-8') as file:
        with contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
            result = unittest.TextTestRunner(stream=file, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromNames(suites))
    journal.finish([], tests=result.testsRun, all_tests_passed=result.wasSuccessful(),
                   failures=len(result.failures), errors=len(result.errors), actual_classifier_fits=0,
                   preprocessing_only_fit_calls=1 if args.include_schema else 0)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
