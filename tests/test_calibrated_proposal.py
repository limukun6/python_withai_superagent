"""Calibration accepts uncertainty while keeping configuration/split/evidence guards."""
from copy import deepcopy
import unittest

from superagent.ab_flash import context_for,validate_proposal
from superagent.hypothesis_agent import check_expectation
from tests.test_ab_flash import fixture_state,answer_for


class CalibratedTests(unittest.TestCase):
    def test_diagnostic_null_is_not_falsely_supported(self):
        state,caps,ids=fixture_state();answer=answer_for(state,caps,ids)
        answer.update(purpose='diagnose',uncertainty='Direction and magnitude are unknown.',expected_change=None)
        validate_proposal(answer,state,caps,ids,'calibrated-v2')
        self.assertEqual(check_expectation(answer,{'status':'completed'},state['history']),{'status':'not_stated'})
        with self.assertRaises(ValueError):validate_proposal(answer,state,caps,ids)

    def test_predeclared_decrease_is_checked_as_decrease_only(self):
        state,caps,ids=fixture_state();answer=answer_for(state,caps,ids)
        answer.update(purpose='coverage',uncertainty='This route may be weaker.',expected_change=dict(
            metric='search_score',reference_trial=1,direction='decrease',min_delta=.01))
        validate_proposal(answer,state,caps,ids,'calibrated-v2')
        previous=state['history'][0]['search_score']
        reference=state['history'][0]
        outcome=dict(status='completed',search_score=previous-.02)
        self.assertEqual(check_expectation(answer,outcome,state['history'])['status'],'not_evaluable')
        scope={k:reference[k] for k in ('split_id','evaluation_payload_sha256','label_sha256','row_ids_sha256',
                                       'primary_metric','metric_spec') if k in reference}
        check=check_expectation(answer,{**outcome,**scope},state['history'])
        self.assertEqual(check['status'],'supported');self.assertLess(check['observed_delta'],0)

    def test_v2_does_not_relax_identity_or_measurement_boundaries(self):
        state,caps,ids=fixture_state();answer=answer_for(state,caps,ids)
        answer.update(purpose='diagnose',uncertainty='Direction unknown.',expected_change=None)
        mutations=[dict(configuration_claim={'classifier':'invented'}),dict(candidate='invented'),dict(evidence_trials=[999]),
            dict(purpose='override_budget'),dict(uncertainty=''),dict(extra='ignored'),
            dict(expected_change=dict(metric='validation_score',reference_trial=1,direction='increase',min_delta=.1)),
            dict(expected_change=dict(metric='search_score',reference_trial=1,direction='increase',min_delta=float('nan')))]
        for mutation in mutations:
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_proposal({**answer,**mutation},state,caps,ids,'calibrated-v2')
        answer['expected_change']=dict(metric='search_score',reference_trial=1,direction='decrease',min_delta=0)
        state['history'][0]['split_id']='confirmation_1'
        with self.assertRaisesRegex(ValueError,'different split'):validate_proposal(answer,state,caps,ids,'calibrated-v2')

    def test_calibrated_context_is_common_to_raw_and_checked(self):
        state,caps,ids=fixture_state()
        raw=context_for(state,caps,ids,False,'calibrated-v2');checked=context_for(state,caps,ids,True,'calibrated-v2')
        self.assertEqual(raw,{k:v for k,v in checked.items() if k not in ('checks','memory')})
        self.assertIn('Do not invent',raw['instruction'])


if __name__=='__main__':unittest.main()
