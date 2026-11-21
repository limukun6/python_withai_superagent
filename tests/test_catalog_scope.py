"""No training: explicit candidate-space projection for same-version Q controls."""
import unittest
from unittest.mock import patch
from superagent.ab_run import registered_capabilities
from superagent.ab_catalog import capabilities_for
from superagent.ab_methods import IMAGE
from superagent.cli import build_parser,_run
from tests.test_ab_scheduler import recorded


class CatalogScopeTests(unittest.TestCase):
    def spec(self,kind):
        spec=recorded()['task'];spec.update(kind=kind,input_schema={})
        return spec

    def test_default_full_is_exact_and_baseline_only_excludes_ports(self):
        spec=self.spec('image');full=registered_capabilities(spec,320)
        self.assertEqual(full,capabilities_for(spec,train_rows=320))
        baseline=registered_capabilities(spec,320,'baseline')
        self.assertEqual(set(full)-set(baseline),{IMAGE})
        self.assertTrue(all(v==full[k] for k,v in baseline.items()))
        self.assertEqual(next(iter(full)),next(iter(baseline)))

    def test_unchanged_spaces_are_identical_and_invalid_scope_rejected(self):
        spec=self.spec('numeric_tabular')
        self.assertEqual(registered_capabilities(spec,8710),registered_capabilities(spec,8710,'baseline'))
        with self.assertRaises(ValueError):registered_capabilities(spec,100,'arbitrary')

    def test_legacy_entry_never_silently_ignores_scope(self):
        args=build_parser().parse_args(['run','missing.zip','--catalog-scope','baseline'])
        with patch('superagent.cli.load_development_task') as loader,self.assertRaises(ValueError):_run(args)
        loader.assert_not_called()
