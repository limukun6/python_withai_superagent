import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVR
from sklearn.ensemble import ExtraTreesClassifier

from superagent.contracts import ExperimentProposal
from superagent.estimator_facts import observed_classifier, verify_configuration_claim
from superagent.models import DummyClassifier
from superagent.targeted_feedback import candidate_facts


class ObservedClassifierTests(unittest.TestCase):
    def test_ridge_is_not_logistic_and_regressor_is_not_classifier(self):
        self.assertEqual(observed_classifier(RidgeClassifier())['family'],'ridge')
        self.assertIsNone(observed_classifier(LinearSVR())['family'])

    def test_each_kernel_has_its_own_identity(self):
        for kernel in ('poly','sigmoid','precomputed'):
            report=observed_classifier(SVC(kernel=kernel))
            self.assertEqual(report['family'],kernel+'_svm')
            self.assertEqual(report['parameters']['kernel'],kernel)
        self.assertIsNone(observed_classifier(SVC(kernel=lambda x,y:x@y.T))['family'])

    def test_pipeline_reports_the_final_estimator(self):
        model = Pipeline([('scale', StandardScaler()), ('clf', LogisticRegression())])
        self.assertEqual(observed_classifier(model)['family'], 'logistic')

    def test_pair_tuple_reports_the_classifier_not_the_vectorizer(self):
        model = (object(), ExtraTreesClassifier(n_estimators=2))
        self.assertEqual(observed_classifier(model)['family'], 'extra_trees')

    def test_svm_kernel_decides_the_family(self):
        self.assertEqual(observed_classifier(SVC(kernel='linear'))['family'], 'linear_svm')
        self.assertEqual(observed_classifier(SVC(kernel='rbf'))['family'], 'rbf_svm')

    def test_dummy_and_unknown_models(self):
        self.assertEqual(observed_classifier(DummyClassifier(strategy='most_frequent'))['family'], 'dummy')
        report = observed_classifier(object())
        self.assertIsNone(report['family'])
        self.assertEqual(report['class'], 'object')

    def test_blend_wrapper_is_unwrapped_through_its_classifier(self):
        class Wrapper:
            def __init__(self, classifier):
                self.classifier = classifier
        Wrapper.__module__ = 'superagent.models'  # only package wrappers are followed
        self.assertEqual(observed_classifier(Wrapper(LogisticRegression()))['family'], 'logistic')

    def test_ensemble_base_estimator_is_not_mistaken_for_the_fitted_model(self):
        self.assertEqual(observed_classifier(ExtraTreesClassifier(n_estimators=2))['family'], 'extra_trees')


class ClaimVerificationTests(unittest.TestCase):
    def observed(self, family):
        return {'family': family, 'class': 'X', 'source': 'fitted_estimator'}

    def test_confirmed_contradicted_unverified_and_absent(self):
        claim = {'configuration_claim': {'classifier': 'logistic'}}
        self.assertEqual(verify_configuration_claim(claim, self.observed('logistic'))['status'], 'confirmed')
        self.assertEqual(verify_configuration_claim(claim, self.observed('catboost'))['status'], 'contradicted')
        self.assertEqual(verify_configuration_claim(claim, self.observed(None))['status'], 'unverified')
        self.assertEqual(verify_configuration_claim({}, self.observed('logistic'))['status'], 'not_declared')

    def test_unknown_catalogue_entry_is_stated_as_unverified(self):
        facts = candidate_facts('numeric_tabular', {'id': 'not_registered', 'params': {}})
        self.assertEqual(facts['source'], 'unverified')
        self.assertIsNone(facts['classifier'])
        self.assertIn('reason', facts)

    def test_unknown_fact_accepts_the_claim_but_known_fact_still_rejects(self):
        history = [dict(trial=1, candidate='logistic_C0.1', status='completed', search_accuracy=.6)]
        unknown = dict(id='not_registered', params={})
        raw = {'candidate': 'not_registered', 'task_kind': 'numeric_tabular',
               'configuration_claim': {'classifier': 'catboost'}}
        proposal = ExperimentProposal.from_decision(2, unknown, 'test', raw, history)
        self.assertEqual(proposal.candidate, 'not_registered')
        known = dict(id='numeric_tabular_logistic_C0.1', params={'C': 0.1})
        with self.assertRaisesRegex(ValueError, 'registered classifier'):
            ExperimentProposal.from_decision(2, known, 'test', raw | {'candidate': known['id']}, history)

    def test_malformed_claim_is_rejected_even_when_the_fact_is_unknown(self):
        history = [dict(trial=1, candidate='logistic_C0.1', status='completed', search_accuracy=.6)]
        unknown = dict(id='not_registered', params={})
        for bad in ({'classifier': ''}, {}, {'classifier': 'logistic', 'extra': 1}, 'logistic'):
            raw = {'candidate': 'not_registered', 'task_kind': 'numeric_tabular', 'configuration_claim': bad}
            with self.assertRaisesRegex(ValueError, 'exactly one classifier'):
                ExperimentProposal.from_decision(2, unknown, 'test', raw, history)


if __name__ == '__main__':
    unittest.main()
