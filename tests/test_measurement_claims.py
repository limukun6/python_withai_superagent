import unittest
from superagent.measurement_claims import check_measurement_claim


class MeasurementClaimTests(unittest.TestCase):
    def setUp(self):
        self.history=[dict(trial=1,status='completed',primary_metric='macro_f1',search_score=.65,
            search_accuracy=.7,train_accuracy=.95,search_metrics={'macro_f1':.65,'accuracy':.7}),
            dict(trial=3,status='failed',search_accuracy=.99)]

    def claim(self,**updates):
        return dict(trial=1,split='search',metric='macro_f1',value=.65)|updates

    def test_named_primary_metric_and_accuracy_are_distinct(self):
        self.assertEqual(check_measurement_claim(self.claim(),self.history)['status'],'confirmed')
        self.assertEqual(check_measurement_claim(self.claim(value=.7),self.history)['status'],'contradicted')
        result=check_measurement_claim(self.claim(metric='accuracy',value=.7),self.history)
        self.assertEqual(result['status'],'confirmed')
        self.assertEqual(result['evidence_id'],'trial:1/search/accuracy')

    def test_wrong_split_and_missing_measurement_are_not_confirmed(self):
        r=check_measurement_claim(self.claim(split='train'),self.history)
        self.assertEqual(r['status'],'unverifiable')
        self.assertEqual(check_measurement_claim(self.claim(metric='roc_auc'),self.history)['status'],'unverifiable')

    def test_trial_ids_are_not_inferred_from_history_length(self):
        self.assertEqual(check_measurement_claim(self.claim(trial=2),self.history)['status'],'contradicted')
        self.assertEqual(check_measurement_claim(self.claim(trial=0),self.history)['status'],'contradicted')
        self.assertEqual(check_measurement_claim(self.claim(trial=3),self.history)['status'],'unverifiable')

    def test_duplicate_identity_and_nonfinite_observation_stay_unknown(self):
        self.assertEqual(check_measurement_claim(self.claim(),self.history*2)['status'],'unverifiable')
        self.history[0]['search_metrics']['macro_f1']=float('nan')
        self.assertEqual(check_measurement_claim(self.claim(),self.history)['status'],'unverifiable')

    def test_malformed_claims_are_rejected(self):
        for bad in ({},self.claim(value=float('nan')),self.claim(trial=True),self.claim(split=[])):
            with self.assertRaises(ValueError):check_measurement_claim(bad,self.history)

    def test_named_fallback_requires_primary_metric_identity(self):
        row=dict(trial=1,status='completed',primary_metric='accuracy',search_score=.8)
        self.assertEqual(check_measurement_claim(self.claim(),[row])['status'],'unverifiable')

    def test_tolerance_is_in_score_units(self):
        self.assertEqual(check_measurement_claim(self.claim(value=.65005),self.history)['status'],'confirmed')
        self.assertEqual(check_measurement_claim(self.claim(value=.651),self.history)['status'],'contradicted')

if __name__=='__main__':unittest.main()
