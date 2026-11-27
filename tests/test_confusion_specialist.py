"""Boundary and frozen-publication tests; deterministic probes have no trained classifier or model API."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np

from superagent.ab_run import sha
from superagent.confusion_specialist import ClassPairModel, design_experiment, propose_pair
from superagent.data import TrainingTask
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model, probability_with_model


class Original:
    classes_=np.array([2,0,1])
    def predict_proba(self,x):
        values=np.asarray(x)[:,0].astype(int)%3
        return np.asarray([[.05,.55,.4] if i<2 else [.8,.12,.08] for i in values])
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]


class PairProbe:
    classes_=np.array([1,0])
    def __init__(self,helpful=True):self.helpful=helpful
    def predict_proba(self,x):
        value=np.asarray(x)[:,0].astype(int)%3
        choose_one=value==1 if self.helpful else np.zeros(len(value),dtype=bool)
        return np.where(choose_one[:,None],np.array([[.9,.1]]),np.array([[.2,.8]]))
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(axis=1)]


class MeasuredWorker:
    limit=11
    def __init__(self,helpful):
        self.helpful=helpful
        self.calls=[]
        self.caps={'measured':dict(supports_probabilities=True,classifier_fits=1,refit_classifier_fits=1,
                                  estimator='linear',params={},representation='numeric')}
    def accounting(self):return dict(attempted_classifier_fits=8+len(self.calls),unknown_failed_attempts=0)
    def worker(self,payload,name,destination,timeout,*,reserve):
        record=joblib.load(payload)
        assert set(record['y_train'])=={0,1} and self.accounting()['attempted_classifier_fits']+1+reserve<=self.limit
        self.calls.append(record['X_train'][:,0].astype(int))
        return PairProbe(self.helpful),dict(training_rows=len(record['y_train']))


class PairDesignTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path=Path(temporary.name)
        self.x=np.arange(90)[:,None]
        self.y=self.x[:,0]%3
        self.metric=MetricSpec('accuracy',(0,1,2))
        self.base=Original()
        self.log=RunLogger(self.path)
        joblib.dump(dict(model=self.base),self.path/'selected_model.joblib')
        self.incumbent=dict(model_file='selected_model.joblib',model_sha256=sha(self.path/'selected_model.joblib'))
        self.folds={'primary':(np.arange(30,90),np.arange(30),{}),
                    'confirmation_1':(np.r_[0:30,60:90],np.arange(30,60),{})}
        self.opening=SimpleNamespace(log=self.log,task=TrainingTask('numeric_tabular',self.x,self.y,{}),
                metric=self.metric,folds=self.folds,record={},save=lambda:None)
        self.bases={}
        for fold,(_,held,_) in self.folds.items():
            pred=self.base.predict(self.x[held])
            self.bases[fold]=dict(probabilities=self.base.predict_proba(self.x[held])[:,[1,2,0]],labels=self.y[held],
                                  predictions=pred,score=float((pred==self.y[held]).mean()),row_key=fold)
        self.observed={k:{'measured':v} for k,v in self.bases.items()}

    def run_design(self,worker):
        return design_experiment(self.opening,dict(members=['measured']),self.incumbent,self.bases,list(self.folds),
            self.observed,dict(statistics=dict(train_rows=60),minority_class_fraction=1/3),worker,30,set(),None)

    def test_probability_order_and_outside_predictions_survive_reload(self):
        model=ClassPairModel('numeric_tabular',self.base,PairProbe(),np.array([0,1,2]),(0,1))
        archive=self.path/'model.joblib'
        joblib.dump(model,archive)
        restored=joblib.load(archive)
        predicted=predict_with_model('numeric_tabular',restored,self.x)
        classes,probs=probability_with_model('numeric_tabular',restored,self.x)
        np.testing.assert_array_equal(classes,[0,1,2])
        np.testing.assert_array_equal(predicted,self.y)
        before=self.base.predict_proba(self.x)[:,[1,2,0]]
        np.testing.assert_array_equal(probs[self.y==2],before[self.y==2])
        np.testing.assert_array_equal(probs[:,2],before[:,2])
        np.testing.assert_allclose(probs.sum(axis=1),1)

    def test_repeated_confusion_and_scoring_contract_needed(self):
        self.assertEqual(propose_pair(self.bases['primary'],self.metric)['pair'],[0,1])
        self.assertIsNone(propose_pair(self.bases['primary'],MetricSpec('accuracy',(0,1))))
        no_scores=dict(self.bases['primary'],probabilities=None)
        self.assertEqual(propose_pair(no_scores,self.metric)['gate'],'predicted_member')
        self.assertIsNone(propose_pair(no_scores,MetricSpec('roc_auc',(0,1,2))))

    def test_two_confirmed_gains_publish_new_file_using_only_pair_rows(self):
        w=MeasuredWorker(True)
        selected,_=self.run_design(w)
        self.assertEqual(selected['composition'],'class_pair_refinement')
        bundle=joblib.load(self.path/selected['model_file'])
        self.assertEqual(selected['family'],bundle['candidate'])
        legacy=dict(selected,family='class_pair_refinement',task_spec_sha256='test-spec')
        self.assertEqual(bundle['model'].verified_legacy_id(legacy),bundle['candidate'])
        self.assertIsNone(bundle['model'].verified_legacy_id(dict(legacy,selected_experiment='unbound')))
        self.assertEqual(selected['incumbent_model_sha256'],sha(self.path/'selected_model.joblib'))
        self.assertEqual(json.loads((self.path/'final_selection.json').read_text()),selected)
        for ids,(_,held,_) in zip(w.calls,self.folds.values()):
            self.assertFalse(set(ids)&set(held))
        self.assertEqual(len(w.calls),3)
        self.assertEqual(set(w.calls[-1]),set(np.flatnonzero(self.y!=2)))
        np.testing.assert_array_equal(joblib.load(self.path/selected['model_file'])['model'].predict(self.x),self.y)

    def test_negative_primary_stops_before_confirmation_or_refit(self):
        w=MeasuredWorker(False)
        selected,_=self.run_design(w)
        self.assertEqual(selected,self.incumbent)
        self.assertEqual(len(w.calls),1)
        self.assertEqual(self.opening.record['designed_method']['status'],'rejected')
        self.assertFalse((self.path/'designed_model.joblib').exists())

    def test_hard_route_uses_labels_without_synthesizing_probability(self):
        for row in self.bases.values():row['probabilities']=None
        w=MeasuredWorker(True)
        w.caps['measured']['supports_probabilities']=False
        selected,_=self.run_design(w)
        fitted=joblib.load(self.path/selected['model_file'])['model']
        self.assertEqual(fitted.gate,'predicted_member')
        np.testing.assert_array_equal(fitted.predict(self.x),self.y)
        with self.assertRaisesRegex(ValueError,'no probability'):fitted.predict_proba(self.x)

    def test_reserve_only_the_last_feasible_independent_confirmation(self):
        from superagent.task_opening import TaskOpening
        o=SimpleNamespace(mode='opening-feedback-confirm',folds=self.folds,record={'opportunity':{'reserved_fits':3}},save=lambda:None)
        cap={'classifier_fits':1,'refit_classifier_fits':1}
        caps={'base':cap,'new':cap}
        menu=[dict(action='explore',members=['new'],split='primary'),dict(action='stop',members=['base'],split=None),
              dict(action='confirm',members=['base'],split='confirmation_1',evidence={'target_members':['base']})]
        observed={'primary':{'base':{}},'confirmation_1':{}}
        self.assertIsNone(TaskOpening.protect_confirmation(o,menu,caps,observed,3))
        self.assertEqual(TaskOpening.protect_confirmation(o,menu,caps,observed,2),menu[2])
        self.assertIsNone(TaskOpening.protect_confirmation(o,menu,caps,observed,1))
        observed['confirmation_1']['base']={}
        self.assertIsNone(TaskOpening.protect_confirmation(o,menu,caps,observed,2))

    def test_restrictions_and_short_allowance_fall_back_without_fitting(self):
        w=MeasuredWorker(True)
        self.opening.task.spec['method_restrictions']=['linear_only']
        self.assertIsNone(self.run_design(w))
        self.opening.task.spec.clear()
        w.limit=10
        self.assertIsNone(self.run_design(w))
        self.assertFalse(w.calls)
