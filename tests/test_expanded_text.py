import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import time

import joblib
import numpy as np

from superagent.confirmation import confirm_candidates
from superagent.data import TrainingTask,DevelopmentTask
from superagent.benchmark import run_one
from superagent.evidence_agent import route_of,state_for
from superagent.events import RunLogger
from superagent.models import search_candidates,predict_with_model,probability_with_model,text_statistics


class ExpandedTextTests(unittest.TestCase):
    def setUp(self):
        self.pairs=[(f"A longer example with number {i} and a question?",f"Short example {i%3}.") for i in range(40)]
        self.labels=np.asarray([i%2 for i in range(40)])

    def test_new_routes_are_additive_and_tabular_unchanged(self):
        original=search_candidates('pair_text')
        expanded=search_candidates('pair_text','expanded')
        self.assertEqual(len(original),14)
        self.assertEqual(len(expanded),22)
        self.assertEqual([c.family for c in expanded[:14]],[c.family for c in original])
        self.assertEqual(route_of('word_char_stats_C0.3'),'word_char_stats')
        self.assertEqual(route_of('char_C0.3'),'char')
        self.assertEqual([c.family for c in search_candidates('categorical_tabular')],
                         [c.family for c in search_candidates('categorical_tabular','expanded')])

    def test_style_model_reverses_probabilities_and_preserves_fitted_preprocessing(self):
        candidate=next(c for c in search_candidates('pair_text','expanded') if c.family=='word_char_stats_C0.3')
        model,pred=candidate.fit_predict(self.pairs,self.labels,self.pairs[:6])
        classes,p=probability_with_model('pair_text',model,self.pairs[:6])
        _,reverse=probability_with_model('pair_text',model,[(b,a) for a,b in self.pairs[:6]])
        np.testing.assert_allclose(p[:,1]+reverse[:,1],1.,atol=1e-6)
        vocabulary=copy.deepcopy(model.word.vocabulary_)
        variance=model.scaler.var_.copy()
        predict_with_model('pair_text',model,[('UNSEENWORDLONGER ?!?','Different unseen input')])
        self.assertEqual(model.word.vocabulary_,vocabulary)
        np.testing.assert_array_equal(model.scaler.var_,variance)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.joblib'
            joblib.dump(model,path)
            np.testing.assert_array_equal(predict_with_model('pair_text',joblib.load(path),self.pairs[:6]),pred)

    def test_char_only_model_uses_common_prediction_contract(self):
        candidate=next(c for c in search_candidates('pair_text','expanded') if c.family=='char_C0.1')
        model,pred=candidate.fit_predict(self.pairs,self.labels,self.pairs[:5])
        np.testing.assert_array_equal(predict_with_model('pair_text',model,self.pairs[:5]),pred)
        classes,probs=probability_with_model('pair_text',model,self.pairs[:5])
        self.assertEqual(probs.shape,(5,2))
        np.testing.assert_allclose(probs.sum(axis=1),1.)

    def test_style_features_are_finite_for_empty_and_unicode_text(self):
        for text in ('','中文标题 １２３！','A ?!? 123','   '):
            self.assertTrue(np.all(np.isfinite(text_statistics(text))))

    def test_planner_receives_evaluation_definition_without_external_scores(self):
        protocol={'grouping':'shared_text','search_rows_sharing_text':0,'selection_data':'training_partition_only'}
        state=state_for({'evaluation_protocol':protocol,'official_validation_accuracy':.987654321},[],
                        [{'id':'char_C0.1','params':{'C':.1}}],5,'budget_routes')
        self.assertEqual(state['profile']['evaluation_protocol'],protocol)
        self.assertNotIn('official_validation_accuracy',state['profile'])

    def test_confirmation_reuses_primary_only_when_grouping_matches(self):
        pairs=[(f'a {i}',f'b {i}') for i in range(60)]
        task=TrainingTask('pair_text',pairs,np.tile([0,1],30))
        history=[{'trial':1,'candidate':'word_C0.05','status':'completed','search_accuracy':.6}]
        for primary,expected in (('shared_text',2),('exact_rows',3)):
            with self.subTest(primary=primary),tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)
                logger=RunLogger(path)
                def train(*args):return None,{'status':'completed','search_accuracy':.65,'training_seconds':.01}
                result=confirm_candidates(task,history,11,path,logger,train,time.monotonic(),30,10,
                    repeats=3,top_k=3,grouping='shared_text',primary_grouping=primary)
                self.assertEqual(result['new_fits_attempted'],expected)
                self.assertTrue(result['valid_confirmation'])

    def test_grouped_search_protocol_reaches_each_policy_decision(self):
        pairs=[]
        for i in range(30):
            pairs.extend([(f'alpha title {i}',f'beta title {i}'),(f'beta title {i}',f'gamma title {i}')])
        labels=np.tile([0,1],30)
        task=DevelopmentTask('pair_text','fixture',pairs,labels,pairs[:4],labels[:4],{})
        observed=[]
        def choose(profile,history,available):
            observed.append(profile['evaluation_protocol'])
            return available[0]['id'],'Local protocol wiring test'
        with tempfile.TemporaryDirectory() as temporary,patch('superagent.benchmark.Policy.choose',side_effect=choose):
            result=run_one(task,'random',11,Path(temporary)/'run',None,max_trials=2,anchor_first=True,
                           search_grouping='shared_text',portfolio='expanded',confirmation_repeats=3,
                           confirmation_grouping='shared_text',refit_full_train=True)
        self.assertTrue(result['valid_policy_run'])
        self.assertEqual(len(observed),1)
        self.assertEqual(observed[0]['grouping'],'shared_text')
        self.assertEqual(observed[0]['search_rows_sharing_text'],0)
        final=observed[0]['planned_final_selection']
        self.assertEqual(final['final_metric'],'mean_confirmation_accuracy')
        self.assertEqual(final['maximum_additional_confirmation_fits'],4)
        self.assertEqual(final['provided_validation_role'],'score_only_after_selection_and_model_freeze')
        self.assertTrue(final['full_train_refit_after_selection'])
        self.assertEqual(result['confirmation']['new_fits_attempted'],final['maximum_additional_confirmation_fits'])


if __name__=='__main__':unittest.main()
