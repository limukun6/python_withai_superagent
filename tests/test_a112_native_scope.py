import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.pipeline import Pipeline

from superagent.ab_catalog import capabilities_for
from superagent.ab_methods import PREFERENCE_CATALOG_ENV, TEXT_DEEP, TEXT_EDIT
from superagent.ab_sentence import DeepPreferenceFeatures
from superagent.cli import _run, build_parser
from superagent.external_assets import model_resources
from tests.test_ab_text_kernel import example_embeddings


class NativeScopeTests(unittest.TestCase):
    def args(self, mode='standard', scheduler='ab-v1'):
        return build_parser().parse_args(['run', 'anonymous.zip', '--scheduler', scheduler,
                                         '--portfolio', 'ab-v1', '--preference-catalog', mode])

    def test_explicit_standard_overrides_ambient_scope_and_restores_it(self):
        for ambient in ('deep', 'deep-edit'):
            with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: ambient}):
                def check(args):
                    self.assertEqual(os.environ[PREFERENCE_CATALOG_ENV], 'standard')
                    return 17
                with patch('superagent.ab_run.command', side_effect=check):
                    self.assertEqual(_run(self.args()), 17)
                self.assertEqual(os.environ[PREFERENCE_CATALOG_ENV], ambient)

    def test_failure_and_absent_previous_scope_restore_correctly(self):
        with patch.dict(os.environ, {}, clear=True):
            for mode in ('standard', 'deep', 'deep-edit'):
                def fail(args):
                    self.assertEqual(os.environ[PREFERENCE_CATALOG_ENV], mode)
                    raise ValueError('controlled stop')
                with patch('superagent.ab_run.command', side_effect=fail):
                    with self.assertRaisesRegex(ValueError, 'controlled stop'):
                        _run(self.args(mode))
                self.assertNotIn(PREFERENCE_CATALOG_ENV, os.environ)

    def test_expansion_rejects_incompatible_executor(self):
        with self.assertRaisesRegex(ValueError, 'shared A/B executor'):
            _run(self.args('deep', 'fixed'))

    def test_catalogue_additions_require_directional_semantics_and_assets(self):
        def names(semantics, mode, ready=True):
            spec = dict(contract_version=1, objective='classification', kind='pair_text',
                        metric=dict(name='accuracy', labels=[0, 1], direction='maximize'),
                        input_schema=dict(modality='pair_text', pair_semantics=semantics,
                                          pair_label_one='left' if semantics=='preference' else 'equivalent'))
            with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: mode}), \
                    patch('superagent.ab_sentence.verified_directory', return_value=Path('fixture') if ready else None):
                return set(capabilities_for(spec, train_rows=40))
        base = names('preference', 'standard')
        self.assertEqual(names('preference', 'deep') - base, {TEXT_DEEP})
        self.assertEqual(names('preference', 'deep-edit') - base, {TEXT_DEEP, TEXT_EDIT})
        self.assertFalse({TEXT_DEEP, TEXT_EDIT} & names('preference', 'deep-edit', False))
        self.assertFalse({TEXT_DEEP, TEXT_EDIT} & names('symmetric', 'deep-edit'))

    def test_deep_resource_is_identified_from_actual_feature_object(self):
        resources, packages = model_resources(Pipeline([('feature', DeepPreferenceFeatures('pair_text'))]))
        self.assertEqual([r['name'] for r in resources], ['minilm_l12'])
        self.assertEqual(resources[0]['environment_variable'], 'SUPERAGENT_DEEP_MINILM_WEIGHTS')
        self.assertIn('transformers', packages)

    def test_both_native_feature_modes_flip_under_swap_without_classifier_fit(self):
        rows = [('a quiet blue bridge near the long river at dawn', 'a bright green bridge near the long river at dawn'),
                ('another small blue bridge near the old river at dusk', 'another large green bridge near the old river at dusk')]*2
        def encode(texts, **kwargs):
            self.assertTrue(kwargs['deep'])
            return example_embeddings(texts)
        with patch('superagent.ab_sentence.encode', side_effect=encode):
            for edits in (False, True):
                model = DeepPreferenceFeatures('pair_text', edit_context=edits).fit(rows)
                forward = model.transform(rows)
                reverse = model.transform([(b, a) for a, b in rows])
                np.testing.assert_allclose((forward+reverse).toarray(), 0, atol=1e-6)
