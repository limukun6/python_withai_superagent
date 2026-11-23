import unittest
from unittest.mock import patch

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from superagent.ab_catalog import capabilities_for
from superagent.ab_run import registered_capabilities
from superagent.numeric_kernel import NAME,NumericKernelFeatures,estimated_workspace
from superagent.cli import build_parser,_run


def specification(kind='numeric_tabular',metric='accuracy',labels=(0,1)):
    return dict(contract_version=1,objective='classification',kind=kind,input_schema={},
                metric=dict(name=metric,labels=list(labels),direction='maximize'))


class NumericKernelTests(unittest.TestCase):
    def test_opt_in_scope_and_probability_fit_contract(self):
        for metric,labels in [('accuracy',(0,1)),('macro_f1',('a','b','c')),('roc_auc',(0,1))]:
            spec=specification(metric=metric,labels=labels)
            old=registered_capabilities(spec,240)
            full=registered_capabilities(spec,240,numeric_kernel_width=3)
            self.assertEqual(set(full)-set(old),{NAME})
            self.assertEqual({n:v for n,v in full.items() if n!=NAME},old)
            self.assertNotIn(NAME,registered_capabilities(spec,240,'baseline',numeric_kernel_width=3))
            item=full[NAME]
            self.assertTrue(item['supports_probabilities'])
            self.assertEqual((item['classifier_fits'],item['refit_classifier_fits']),(1,1))

    def test_non_numeric_missing_dimensions_and_memory_are_not_offered(self):
        for kind in ('categorical_tabular','mixed_tabular','single_text','image'):
            self.assertNotIn(NAME,capabilities_for(specification(kind),train_rows=240,numeric_kernel_width=3))
        spec=specification();needed=estimated_workspace(240,3)
        self.assertNotIn(NAME,capabilities_for(spec,numeric_kernel_width=3))
        self.assertNotIn(NAME,capabilities_for(spec,train_rows=240))
        self.assertNotIn(NAME,capabilities_for(spec,train_rows=240,numeric_kernel_width=3,kernel_memory_bytes=needed-1))
        self.assertIn(NAME,capabilities_for(spec,train_rows=240,numeric_kernel_width=3,kernel_memory_bytes=needed))
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):capabilities_for(spec,train_rows=240,numeric_kernel_width=value)

    def test_training_only_label_independent_order_stable_and_transform_frozen(self):
        rng=np.random.default_rng(13724);x=rng.normal(size=(410,3))
        first=NumericKernelFeatures().fit(x,np.zeros(len(x)))
        second=NumericKernelFeatures().fit(x[::-1],np.ones(len(x)))
        self.assertEqual(first.median_squared_distance_,second.median_squared_distance_)
        np.testing.assert_array_equal(first.map_.random_weights_,second.map_.random_weights_)
        self.assertAlmostEqual(np.exp(-first.map_.gamma*first.median_squared_distance_),.5)
        weights=first.map_.random_weights_.copy();bandwidth=first.median_squared_distance_
        with patch.object(NumericKernelFeatures,'fit',side_effect=AssertionError('prediction refit')):
            transformed=first.transform(np.array([[9.,8.,7.]]))
        self.assertEqual(transformed.shape,(1,515))
        np.testing.assert_array_equal(transformed[0,:3],[9.,8.,7.])
        np.testing.assert_array_equal(weights,first.map_.random_weights_)
        self.assertEqual(first.median_squared_distance_,bandwidth)

    def test_constant_missing_columns_and_wrong_input_width(self):
        x=np.array([[1.,np.nan,2.],[1.,np.nan,3.],[1.,np.nan,np.nan]])
        pipeline=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                           ('scale',StandardScaler()),('kernel',NumericKernelFeatures())])
        values=pipeline.fit_transform(x)
        self.assertTrue(np.isfinite(values).all())
        constant=NumericKernelFeatures().fit(np.zeros((4,2)))
        self.assertEqual(constant.median_squared_distance_,0.)
        self.assertTrue(np.isfinite(constant.transform(np.ones((1,2)))).all())
        with self.assertRaises(ValueError):constant.transform(np.ones((1,3)))

    def test_cli_is_explicit_and_checks_executor_before_data_access(self):
        self.assertFalse(build_parser().parse_args(['run','missing.zip']).numeric_kernel)
        for scheduler in ('fixed','ab-complement','ab-adaptive-complement'):
            args=build_parser().parse_args(['run','missing.zip','--numeric-kernel','--scheduler',scheduler])
            with patch('superagent.cli.load_development_task') as load:
                with self.assertRaisesRegex(ValueError,'shared A/B executor'):_run(args)
                load.assert_not_called()


if __name__=='__main__':unittest.main()
