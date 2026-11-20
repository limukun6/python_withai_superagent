import unittest
import numpy as np

from superagent.distribution_features import DistributionClassifier
from superagent.evidence_agent import route_of
from superagent.method_choice import COOCCUR, MARGINAL, PERMUTED_RESIDUAL, RESIDUAL, RESIDUAL_DIRECT, select_action


class DistributionTests(unittest.TestCase):
    def test_conditional_and_shuffled_lookups_and_unknown_rows(self):
        rows=np.array([['a','b','I'],['b','c','I'],['c','a','II'],['a','a','III'],['b','a','III']]*24)
        labels=np.array([0,1,1,0,1]*24)
        defaults=dict(backbone='logistic_C0.1',backbone_params={'C':.1},prior_count=20.)
        fit=lambda enc:DistributionClassifier(dict(defaults,representation=enc)).fit(rows,labels)
        actual=fit('conditional');null=fit('permuted');freq=fit('frequency')
        self.assertEqual(set(actual.lookups),{0,1,2})
        self.assertEqual(set(actual.lookups),set(null.lookups))
        for c,lookup in actual.lookups.items():
            values=np.stack(list(lookup.values()))
            swapped=np.stack([null.lookups[c][k] for k in lookup])
            np.testing.assert_array_equal(np.sort(values,axis=0),np.sort(swapped,axis=0))
            weights=np.array([(rows[:,c]==level).mean() for level in lookup])
            np.testing.assert_allclose(weights@values,0.,atol=1e-6)
            self.assertEqual(len(next(iter(freq.lookups[c].values()))),1)
        new=np.array([['unseen','unseen','unseen']])
        self.assertFalse(actual.features(new).toarray().any())
        np.testing.assert_allclose(actual.predict_proba(new).sum(axis=1),1.)

    def test_switches_after_paired_evidence(self):
        ev=dict(kind='categorical_tabular',semantic='q',shared_errors=50,required_shared_errors=10,short_latin_share=1.,
                low_support_shared_errors=20,primary_metric='accuracy',corrected=45,lost=20)
        methods={RESIDUAL,RESIDUAL_DIRECT,PERMUTED_RESIDUAL,COOCCUR,MARGINAL}
        self.assertEqual(select_action(ev,methods,'distribution',[])[0],RESIDUAL)
        self.assertEqual(select_action(ev,methods,'distribution',[RESIDUAL])[0],RESIDUAL_DIRECT)
        self.assertEqual(select_action(ev|dict(corrected=18),methods,'distribution',[RESIDUAL])[0],MARGINAL)
        self.assertEqual(select_action(ev|dict(corrected=5),methods,'distribution',[RESIDUAL])[0],COOCCUR)
        self.assertEqual(select_action(ev,methods,'distribution-control',[])[0],PERMUTED_RESIDUAL)
        for method in methods:self.assertTrue(route_of(method))
