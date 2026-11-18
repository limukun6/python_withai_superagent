from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from catboost import CatBoostClassifier

from superagent.adaptive_tabular import fit_adaptive_catboost
from superagent.models import _clean_categories,search_candidates,predict_with_model,probability_with_model
from superagent.evidence_agent import route_of,measured_facts,state_for
from superagent.benchmark import run_one
from superagent.data import DevelopmentTask


def fixture(classes=2):
    rows=[];labels=[]
    for group in range(90):
        for missing in (None,' ? '):
            rows.append([f'group_{group}',f'color_{group%classes}',missing]);labels.append(group%classes)
    return np.asarray(rows,dtype=object),np.asarray(labels)


class AdaptiveTabularTests(unittest.TestCase):
    def test_selector_and_refit_receive_only_their_intended_training_rows(self):
        x,y=fixture();probe=np.asarray([['OUTER_NEVER_FIT','unseen','NaN']],dtype=object)
        original=CatBoostClassifier.fit;calls=[]
        def traced(model,*args,**kwargs):
            calls.append({'x':np.asarray(args[0]).copy(),'y':np.asarray(args[1]).copy(),
                'kwargs':kwargs,'params':model.get_params()})
            return original(model,*args,**kwargs)
        with patch.object(CatBoostClassifier,'fit',new=traced):
            model,pred=fit_adaptive_catboost(x,y,probe,depth=3,max_iterations=40,early_stopping_rounds=5)
        metadata=model.training_metadata;fit=metadata['selector_fit_indices'];holdout=metadata['selector_holdout_indices']
        self.assertEqual(len(calls),2);self.assertEqual(model.classifier_fits,2)
        self.assertEqual(sorted(fit+holdout),list(range(len(y))));self.assertFalse(set(fit)&set(holdout))
        for group in range(90):self.assertEqual(2*group in fit,2*group+1 in fit)
        cleaned=_clean_categories(x)
        np.testing.assert_array_equal(calls[0]['x'],cleaned[fit]);np.testing.assert_array_equal(calls[0]['y'],y[fit])
        np.testing.assert_array_equal(calls[0]['kwargs']['eval_set'][0],cleaned[holdout])
        np.testing.assert_array_equal(calls[0]['kwargs']['eval_set'][1],y[holdout])
        np.testing.assert_array_equal(calls[1]['x'],cleaned);np.testing.assert_array_equal(calls[1]['y'],y)
        self.assertNotIn('eval_set',calls[1]['kwargs']);self.assertNotIn('use_best_model',calls[1]['kwargs'])
        for call in calls:self.assertNotIn('OUTER_NEVER_FIT',call['x'])
        self.assertEqual(metadata['selector_overlap_audit']['search_duplicate_rows'],0)
        self.assertEqual(calls[1]['params']['iterations'],metadata['selected_iterations'])
        self.assertEqual(model.classifier.tree_count_,metadata['selected_iterations']);self.assertEqual(pred.shape,(1,))
        self.assertAlmostEqual(min(metadata['selector_holdout_loss_curve']),metadata['selected_holdout_loss'])
        self.assertEqual(metadata['selector_holdout_loss_curve'][metadata['selected_iterations']-1],metadata['selected_holdout_loss'])

    def test_binary_and_multiclass_saved_models_predict_without_any_refitting(self):
        for classes in (2,3):
            with self.subTest(classes=classes):
                x,y=fixture(classes);model,pred=fit_adaptive_catboost(x,y,x[:9],depth=3,max_iterations=20,early_stopping_rounds=3)
                self.assertEqual(model.training_metadata['stopping_metric'],'Logloss' if classes==2 else 'MultiClass')
                with tempfile.TemporaryDirectory() as temp:
                    path=Path(temp)/'model.joblib';joblib.dump(model,path)
                    with patch.object(CatBoostClassifier,'fit',side_effect=AssertionError('Prediction cannot fit')):
                        restored=joblib.load(path)
                        np.testing.assert_array_equal(predict_with_model('categorical_tabular',restored,x[:9]),pred)
                        fitted_classes,probs=probability_with_model('categorical_tabular',restored,x[:9])
                    np.testing.assert_array_equal(fitted_classes,np.arange(classes))
                    self.assertEqual(probs.shape,(9,classes));np.testing.assert_allclose(probs.sum(axis=1),1.)

    def test_invalid_configuration_and_missing_internal_class_fail_before_fitting(self):
        x,y=fixture()
        with patch.object(CatBoostClassifier,'fit') as fitted:
            for params in ({'max_iterations':0},{'early_stopping_rounds':False},{'depth':-1},{'l2_leaf_reg':float('nan')}):
                with self.assertRaises(ValueError):fit_adaptive_catboost(x,y,x[:2],**params)
            with patch('superagent.adaptive_tabular.inner_split',return_value=(np.flatnonzero(y==0),np.flatnonzero(y==1))):
                with self.assertRaisesRegex(ValueError,'Every class'):fit_adaptive_catboost(x,y,x[:2])
            fitted.assert_not_called()

    def test_optional_catalog_preserves_original_controls_and_documents_two_fits(self):
        base=search_candidates('categorical_tabular','expanded');new=search_candidates('categorical_tabular','adaptive')
        self.assertEqual([len(base),len(new)],[14,18]);self.assertEqual([(c.family,c.params) for c in base],[(c.family,c.params) for c in new[:14]])
        self.assertEqual({route_of(c.family) for c in new[14:]},{'adaptive_catboost'})
        self.assertTrue(all(c.params['classifier_fits_per_evaluation']==2 for c in new[14:]))
        self.assertEqual(len(search_candidates('categorical_tabular','all')),19)
        self.assertEqual([c.family for c in search_candidates('pair_text','adaptive')],
            [c.family for c in search_candidates('pair_text','expanded')])

    def test_measured_length_and_compute_reach_feedback_without_large_internal_arrays_or_blind_leak(self):
        history=[{'trial':0,'candidate':'adaptive_catboost_depth6_reg3.0','params':{},'status':'completed',
            'search_accuracy':.7,'selected_iterations':123,'classifier_fits':2,
            'training_metadata':{'selector_fit_indices':['PRIVATE_INTERNAL_ARRAY']}}]
        facts={f['metric']:f['value'] for f in measured_facts(history)}
        self.assertEqual(facts['selected_iterations'],123);self.assertEqual(facts['classifier_fits'],2)
        for policy in ('budget_routes','evidence_blind'):
            state=state_for({},history,[],3,policy)
            self.assertNotIn('PRIVATE_INTERNAL_ARRAY',str(state))
            if policy=='evidence_blind':self.assertEqual(state['facts'],[])
            else:self.assertIn('selected_iterations',str(state['facts']))

    def test_full_pipeline_counts_selection_and_refit_classifiers_in_all_phases(self):
        x,y=fixture();task=DevelopmentTask('categorical_tabular','synthetic',x,y,x[:12],y[:12],{})
        candidate=next(c for c in search_candidates('categorical_tabular','adaptive') if c.family=='adaptive_catboost_depth6_reg3.0')
        with tempfile.TemporaryDirectory() as temp,patch('superagent.benchmark.search_candidates',return_value=[candidate]):
            result=run_one(task,'random',707,Path(temp)/'run',None,max_trials=1,anchor_first=True,
                confirmation_repeats=3,confirmation_top_k=1,refit_full_train=True,portfolio='adaptive')
        self.assertTrue(result['confirmation']['valid_confirmation']);self.assertTrue(result['refit_succeeded'])
        self.assertEqual(result['confirmation']['classifier_fits_completed'],4)
        self.assertEqual(result['refit_classifier_fits_completed'],2)
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],8)
        self.assertTrue(result['classifier_fit_accounting']['complete_count_known'])


if __name__=='__main__':unittest.main()
