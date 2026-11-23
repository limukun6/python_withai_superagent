import os
import unittest
from unittest.mock import patch

from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.cli import _run,build_parser


class PreferenceCatalogScopeTests(unittest.TestCase):
    def options(self,*extra):
        return build_parser().parse_args(['run','anonymous.zip','--scheduler','ab-v1',
            '--portfolio','ab-v1','--preference-catalog','deep-edit',*extra])

    def test_worker_catalog_scope_restored_after_success_or_failure(self):
        def check_scope(args):
            self.assertEqual(os.environ[PREFERENCE_CATALOG_ENV],'deep-edit')
            return 17

        with patch.dict(os.environ,{PREFERENCE_CATALOG_ENV:'deep'}):
            with patch('superagent.ab_run.command',side_effect=check_scope):
                self.assertEqual(_run(self.options()),17)
            self.assertEqual(os.environ[PREFERENCE_CATALOG_ENV],'deep')
            with patch('superagent.ab_run.command',side_effect=ValueError('worker error')):
                with self.assertRaisesRegex(ValueError,'worker error'):
                    _run(self.options())
            self.assertEqual(os.environ[PREFERENCE_CATALOG_ENV],'deep')

    def test_distinct_component_executor_cannot_silently_override_the_shared_menu(self):
        with patch('superagent.ab_run.command') as run:
            with self.assertRaisesRegex(ValueError,'standard A/B executor'):
                _run(self.options('--preference-components'))
            run.assert_not_called()
