"""Evidence scope counterexamples; no model training or provider calls."""
from copy import deepcopy
import unittest
from superagent.comparison_scope import comparison_scope
from superagent.hypothesis_agent import check_expectation
from superagent.targeted_feedback import check_slice_expectation


class ComparisonScopeTests(unittest.TestCase):
    def fixture(self):
        before=dict(trial=1,status='completed',primary_metric='accuracy',search_score=.6,split_id='primary',
            evaluation_payload_sha256='a'*64,label_sha256='b'*64,row_ids_sha256='c'*64)
        return before,dict(before,trial=2,search_score=.8),dict(expected_change=dict(metric='search_score',reference_trial=1,direction='increase',min_delta=.1))

    def test_valid_scope_and_explicit_conflicts(self):
        before,after,proposal=self.fixture()
        self.assertEqual(check_expectation(proposal,after,[before])['status'],'supported')
        for key,value in (('split_id','confirmation_2'),('evaluation_payload_sha256','d'*64),
                          ('label_sha256','e'*64),('row_ids_sha256','f'*64),('primary_metric','roc_auc')):
            altered=dict(after,**{key:value})
            with self.subTest(key=key):self.assertEqual(check_expectation(proposal,altered,[before])['status'],'not_evaluable')

    def test_missing_one_side_and_duplicate_trial_do_not_confirm(self):
        before,after,proposal=self.fixture();del after['label_sha256']
        self.assertEqual(check_expectation(proposal,after,[before])['status'],'not_evaluable')
        self.assertEqual(check_expectation(proposal,after,[before,deepcopy(before)])['status'],'not_evaluable')

    def test_absent_null_or_invalid_legacy_provenance_is_not_complete(self):
        for value in ({},{'split_id':None,'evaluation_payload_sha256':None,'label_sha256':None},
                      {'split_id':'primary','evaluation_payload_sha256':'wrong','label_sha256':'wrong'}):
            self.assertFalse(comparison_scope(value,value)['provenance_complete'])

    def test_a_scalar_gain_does_not_certify_legacy_provenance(self):
        before,after,proposal=self.fixture()
        for row in (before,after):
            for key in ('split_id','evaluation_payload_sha256','label_sha256','row_ids_sha256'):del row[key]
        result=check_expectation(proposal,after,[before])
        self.assertEqual(result['status'],'supported');self.assertFalse(result['comparison_scope']['provenance_complete'])

    def test_slice_row_identity_and_duplicate_slice_are_not_interchangeable(self):
        before,after,_=self.fixture();slice=dict(slice_id='short',feature='length',threshold_from_training=5,relation='low',rows=20,accuracy=.6,row_ids_sha256='1'*64)
        before['error_slices']={'slices':[slice]};after['error_slices']={'slices':[dict(slice,accuracy=.9,row_ids_sha256='2'*64)]}
        expected=dict(reference_trial=1,slice_id='short',min_delta=.1)
        self.assertEqual(check_slice_expectation(expected,after,[before])['status'],'not_evaluable')
        after['error_slices']={'slices':[dict(slice,accuracy=.9),dict(slice,accuracy=.9)]}
        self.assertEqual(check_slice_expectation(expected,after,[before])['status'],'not_evaluable')


if __name__=='__main__':unittest.main()
