import unittest
from superagent.prose_claims import check_prose
from superagent.estimator_facts import verify_configuration_claim


class ReviewReferentTests(unittest.TestCase):
    def status(self,text,field='review',family='logistic'):
        result=check_prose({field:text},{'classifier':family},[])
        return [item['status'] for item in result['items'] if item['kind']=='classifier']

    def test_historical_unbound_family_is_not_current_candidate(self):
        self.assertEqual(self.status('Extra trees leaf2 had strong training performance.'),['unverifiable'])

    def test_matching_unbound_family_is_not_silently_confirmed(self):
        self.assertEqual(self.status('Logistic C=3 was tested earlier.'),['unverifiable'])

    def test_explicit_selected_model_conflict_is_still_detected(self):
        self.assertEqual(self.status('The selected classifier is an SVM.'),['contradicted'])

    def test_explicit_selected_model_match_can_be_confirmed(self):
        self.assertEqual(self.status('The current model is logistic regression.'),['confirmed'])

    def test_reason_does_not_acquire_a_blanket_exemption(self):
        self.assertEqual(self.status('This is an SVM.',field='reason'),['contradicted'])

    def test_structured_configuration_claim_remains_authoritative(self):
        self.assertEqual(verify_configuration_claim({'configuration_claim':{'classifier':'extra_trees'}},{'family':'logistic'})['status'],'contradicted')
