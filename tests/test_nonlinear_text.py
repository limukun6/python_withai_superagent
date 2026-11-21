import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from superagent.models import search_candidates,predict_with_model,probability_with_model
from superagent.evidence_agent import route_of
from superagent.benchmark import run_one
from superagent.data import DevelopmentTask
from superagent.nonlinear_text import fit_style_tree,fit_style_blend,fit_symmetric_linear,style_matrix


def fixture():
    pairs=[];labels=[]
    for i in range(60):
        a='Long report '+('many useful details '*(i%5+1))+str(i)+'?'*(i%3)
        b='Short note '+('small facts '*(i%4+1))+str(i)+'!'*(i%2)
        pairs.append((a,b) if i%2 else (b,a));labels.append(i%2)
    return pairs,np.asarray(labels)


class NonlinearTextTests(unittest.TestCase):
    def test_new_pool_preserves_old_catalogs_and_exposes_component_compute(self):
        base=search_candidates('pair_text','expanded');research=search_candidates('pair_text','research')
        nonlinear=search_candidates('pair_text','nonlinear');all_candidates=search_candidates('pair_text','all')
        from superagent.ab_methods import TEXT
        self.assertEqual([len(base),len(research),len(nonlinear),sum(c.family != TEXT for c in all_candidates)],[22,28,29,43])
        self.assertEqual([(c.family,c.params) for c in base],[(c.family,c.params) for c in nonlinear[:22]])
        self.assertEqual({route_of(c.family) for c in nonlinear[22:]},{'style_tree','style_blend','symmetric_stats'})
        self.assertEqual(sum(c.params['classifier_fits_per_evaluation'] for c in nonlinear[22:]),11)
        self.assertEqual([c.family for c in search_candidates('numeric_tabular','nonlinear')],
            [c.family for c in search_candidates('numeric_tabular','expanded')])

    def test_tree_and_symmetric_linear_probabilities_complement_after_pair_swap(self):
        pairs,y=fixture();reverse=[(b,a) for a,b in pairs]
        for function in (fit_style_tree,fit_symmetric_linear):
            with self.subTest(method=function.__name__):
                model,_=function(pairs,y,pairs[:6])
                classes,forward=probability_with_model('pair_text',model,pairs)
                _,backward=probability_with_model('pair_text',model,reverse)
                self.assertEqual(classes.tolist(),[0,1]);np.testing.assert_allclose(forward.sum(axis=1),1.,atol=1e-12)
                np.testing.assert_allclose(forward[:,1]+backward[:,1],1.,atol=1e-12)
                identical=model.predict_proba([('same text','same text')]);np.testing.assert_allclose(identical,[[.5,.5]],atol=1e-12)

    def test_blend_weight_is_frozen_and_both_classifiers_are_actually_fitted(self):
        pairs,y=fixture();tree_fit=CatBoostClassifier.fit;linear_fit=LogisticRegression.fit;calls=[]
        def tree(model,*args,**kwargs):calls.append('tree');return tree_fit(model,*args,**kwargs)
        def linear(model,*args,**kwargs):calls.append('linear');return linear_fit(model,*args,**kwargs)
        with patch.object(CatBoostClassifier,'fit',new=tree),patch.object(LogisticRegression,'fit',new=linear):
            model,pred=fit_style_blend(pairs,y,pairs[:6],weight=.2)
        self.assertEqual(calls,['linear','tree']);self.assertEqual(model.classifier_fits,2)
        expected=.8*model.linear.predict_proba(pairs[:6])+.2*model.tree.predict_proba(pairs[:6])
        np.testing.assert_allclose(model.predict_proba(pairs[:6]),expected,rtol=0,atol=0)
        np.testing.assert_array_equal(pred,np.argmax(expected,axis=1))
        self.assertEqual(model.style_weight,.2)

    def test_saved_blend_predicts_without_refitting_and_unseen_words_do_not_enter_vocabulary(self):
        pairs,y=fixture();probe=[('ZZUNSEENTOKEN novel symbols?','New report!')]
        model,pred=fit_style_blend(pairs,y,probe)
        vocab=copy.deepcopy(model.linear.base.word.vocabulary_);scale=model.linear.base.scaler.scale_.copy()
        self.assertNotIn('zzunseentoken',vocab)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(model,path)
            with patch.object(CatBoostClassifier,'fit',side_effect=AssertionError('Prediction must not fit')),\
                 patch.object(LogisticRegression,'fit',side_effect=AssertionError('Prediction must not fit')):
                restored=joblib.load(path);np.testing.assert_array_equal(predict_with_model('pair_text',restored,probe),pred)
            self.assertEqual(restored.linear.base.word.vocabulary_,vocab)
            np.testing.assert_array_equal(restored.linear.base.scaler.scale_,scale)

    def test_invalid_labels_and_weights_fail_before_fitting(self):
        pairs,y=fixture()
        with patch.object(CatBoostClassifier,'fit') as tree,patch.object(LogisticRegression,'fit') as linear:
            for function in (fit_style_tree,fit_symmetric_linear,fit_style_blend):
                with self.assertRaises(ValueError):function(pairs,np.arange(len(y))%3,pairs[:2])
                with self.assertRaises(ValueError):function(pairs,np.ones(len(y),dtype=int),pairs[:2])
            for weight in (0,1,float('nan')):
                with self.assertRaises(ValueError):fit_style_blend(pairs,y,pairs[:2],weight=weight)
            tree.assert_not_called();linear.assert_not_called()

    def test_style_features_are_finite_and_depend_only_on_each_input_pair(self):
        pairs=[('',''),('中文 １２３？','Example title!?'),('  ','Upper CAPS 12')]
        matrix=style_matrix(pairs);self.assertEqual(matrix.shape,(3,32));self.assertTrue(np.isfinite(matrix).all())
        reversed_matrix=style_matrix([(b,a) for a,b in pairs])
        np.testing.assert_array_equal(matrix[:,:16],reversed_matrix[:,16:])
        np.testing.assert_array_equal(matrix[:,16:],reversed_matrix[:,:16])

    def test_full_blend_pipeline_counts_both_components_in_search_confirmation_and_refit(self):
        pairs,y=fixture();task=DevelopmentTask('pair_text','synthetic',pairs,y,pairs[:6],y[:6],{})
        candidate=next(c for c in search_candidates('pair_text','nonlinear') if c.family=='style_blend_depth3_weight0.2')
        with tempfile.TemporaryDirectory() as temp,patch('superagent.benchmark.search_candidates',return_value=[candidate]):
            result=run_one(task,'random',707,Path(temp)/'run',None,max_trials=1,anchor_first=True,
                confirmation_repeats=3,confirmation_top_k=1,refit_full_train=True,portfolio='nonlinear',
                search_grouping='shared_text',confirmation_grouping='shared_text',feature_cache=True)
        self.assertTrue(result['confirmation']['valid_confirmation']);self.assertTrue(result['refit_succeeded'])
        self.assertEqual(result['confirmation']['classifier_fits_completed'],4)
        self.assertEqual(result['refit_classifier_fits_completed'],2)
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],8)
        self.assertTrue(result['classifier_fit_accounting']['complete_count_known'])

    def test_failed_candidate_keeps_partial_component_count_unknown(self):
        pairs,y=fixture();task=DevelopmentTask('pair_text','synthetic',pairs,y,pairs[:6],y[:6],{})
        with tempfile.TemporaryDirectory() as temp,patch('superagent.benchmark.train_candidate',
                return_value=(None,{'status':'failed','error':'Synthetic partial component failure','training_seconds':.01})):
            result=run_one(task,'random',707,Path(temp)/'run',None,max_trials=1,anchor_first=True,
                confirmation_repeats=3,refit_full_train=True)
        counts=result['classifier_fit_accounting']
        self.assertEqual(counts['completed_classifier_fits'],0)
        self.assertEqual(counts['failed_candidate_attempts_with_unknown_component_count'],1)
        self.assertFalse(counts['complete_count_known'])


if __name__=='__main__':unittest.main()
