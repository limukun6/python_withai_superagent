import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import joblib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from superagent.ab_run import sha
from superagent.analogy_review import blend_counts, can_review, nominate, promoted, review, representation_source, incumbent_evidence
from superagent.data import TrainingTask
from superagent.events import RunLogger
from superagent.preference_analogy import AnalogyIndex,WeightedProbabilityVote


class ToyModel(ClassifierMixin,BaseEstimator):
    def __init__(self,probability,classes):
        self.probability=np.asarray(probability)
        self.classes_=np.asarray(classes)

    def fit(self,rows,labels):
        return self

    def predict_proba(self,rows):
        return np.tile(self.probability,(len(rows),1))


class Comparison(unittest.TestCase):
    def test_optional_review_reads_default_and_explicit_weights_of_integrated_vote(self):
        from superagent.analogy_review import incumbent_parts
        from superagent.models import SoftVoteModel
        members=['first','second']
        estimators=[object(),object()]
        model=SoftVoteModel('pair_text',estimators,np.array([0,1]))
        self.assertEqual(incumbent_parts(members,model),(estimators,[.5,.5]))
        model.weights=(.25,.75)
        self.assertEqual(incumbent_parts(members,model),(estimators,[.25,.75]))
        model.weights=np.nan
        with self.assertRaises(ValueError):
            incumbent_parts(members,model)

    def test_derived_preference_vote_is_symmetric_and_does_not_count_a_related_group_twice(self):
        vectors={'a':[1.,0,1],'b':[0,1,1],'c':[1,0,2],'d':[0,1,2]}
        def provider(texts):
            return np.asarray([vectors[t] for t in texts])
        duplicate=[('a','b'),('a','b'),('d','c')]
        short=AnalogyIndex().fit([duplicate[0],duplicate[-1]],[1,0],['g','h'],provider)
        full=AnalogyIndex().fit(duplicate,[1,1,0],['g','g','h'],provider)
        rows=[('a','b'),('b','a'),('a','a')]
        output=full.predict_probability(rows,provider)
        np.testing.assert_allclose(output,short.predict_probability(rows,provider))
        self.assertGreater(output[0],.5)
        self.assertAlmostEqual(output[0]+output[1],1.)
        self.assertEqual(output[2],.5)

    def test_ordered_probabilities_match_after_class_reversal(self):
        first=Pipeline([('classifier',ToyModel([.3,.7],[0,1]))])
        second=Pipeline([('classifier',ToyModel([.1,.9],[1,0]))])
        ensemble=WeightedProbabilityVote('pair_text',[first,second],[0,1],[.75,.25])
        np.testing.assert_allclose(ensemble.predict_proba([('x','y')]),[[.45,.55]])

    def test_adviser_election_follows_the_declared_class_order(self):
        truth=np.array([0,0,1,1,1])
        base=np.array([[.7,.3],[.4,.6],[.7,.3],[.3,.7],[.2,.8]])
        other=np.array([[.8,.2],[.9,.1],[.1,.9],[.3,.7],[.1,.9]])
        def observed(p):
            return dict(row_key='same fold',labels=truth,probabilities=p)
        reference=nominate(observed(base),observed(other),truth,classes=[0,1])
        reversed_order=nominate(observed(base[:,::-1]),observed(other[:,::-1]),truth,classes=[1,0])
        self.assertEqual(reference,reversed_order)
        self.assertEqual(reference['scores'][0],3)
        self.assertEqual(reference['scores'][-1],5)
        tie=observed(np.full((len(truth),2),.5))
        self.assertEqual(blend_counts(tie,tie,truth,[0.,1.],classes=[1,0]),[3,3])

    def test_reject_mismatched_rows_and_hold_a_gain_to_confirmation(self):
        truth=np.array([1,0])
        a=dict(row_key='source',labels=truth,probabilities=np.array([[.6,.4],[.8,.2]]))
        b=dict(a,probabilities=np.array([[.2,.8],[.1,.9]]))
        self.assertEqual(blend_counts(a,b,truth,[0.,.5,1.]),[1,1,1])
        with self.assertRaisesRegex(ValueError,'aligned'):
            blend_counts(a,dict(b,row_key='different'),truth,[.5])
        self.assertFalse(promoted(1,[dict(rows=1000,gain=12),dict(rows=1000,gain=-1)]))
        self.assertTrue(promoted(1,[dict(rows=1000,gain=6),dict(rows=1000,gain=0)]))


    def test_entry_respects_modality_allowance_and_connected_holdouts(self):
        schema={'input_schema':{'pair_semantics':'preference'}}
        task=TrainingTask('pair_text',[('a','b'),('c','d'),('e','f')],np.array([0,1,0]),schema)
        folds={key:(np.asarray([j for j in range(3) if j!=i]),np.asarray([i]),{})
               for i,key in enumerate(('primary','confirmation_1','confirmation_2'))}
        metric=SimpleNamespace(name='accuracy',labels=[0,1])
        caps={'anonymous':dict(resource='local_frozen_encoder_cpu',params={'encoder':'opaque'})}
        self.assertTrue(can_review(task,metric,folds,caps,16))
        self.assertFalse(can_review(task,metric,folds,caps,10))
        self.assertFalse(can_review(task,metric,folds,{},16))
        task.spec={'input_schema':{'pair_semantics':'symmetric'}}
        self.assertFalse(can_review(task,metric,folds,caps,16))
        task.spec=schema
        task.X_train[2]=('b','different')
        self.assertFalse(can_review(task,metric,folds,caps,16))

    def test_failed_adviser_preserves_the_already_delivered_incumbent(self):
        class BrokenExecutor:
            limit=16
            restrictions=()
            caps={'anchor':dict(classifier_fits=1,params={'embedding':'opaque'})}
            def accounting(self):
                return {'attempted_classifier_fits':10,'unknown_failed_attempts':0}
            def worker(self,*args,**kwargs):
                raise RuntimeError('staged candidate failed')
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            path=folder/'selected_model.joblib'
            joblib.dump(dict(model='already delivered'),path)
            incumbent=dict(model_file=path.name,model_sha256=sha(path))
            stop={'members':['anchor']}
            logger=RunLogger(folder)
            logger.write_json('final_selection.json',incumbent)
            frozen=sha(folder/'final_selection.json')
            candidate=SimpleNamespace(family='opaque',params=dict(encoder='public'))
            folds={'primary':{'anchor':dict(probabilities=np.array([[.2,.8]]),labels=np.array([1]),score=1.,row_key='primary')},'confirmation_1':{},'confirmation_2':{}}
            with patch('superagent.analogy_review.derived_candidate',return_value=candidate):
                outcome=review(SimpleNamespace(kind='pair_text'),SimpleNamespace(labels=[0,1]),stop,incumbent,
                   BrokenExecutor(),folder,logger,10,folds,{'primary':{'path':folder/'split.joblib'}})
            self.assertEqual(outcome,(incumbent,stop))
            self.assertEqual(sha(folder/'final_selection.json'),frozen)
            self.assertEqual(sha(path),incumbent['model_sha256'])
            self.assertIn('retained_after_failed_opportunity',(folder/'analogy_review.json').read_text())


    def test_one_close_loss_can_earn_evidence_but_cannot_deploy_itself(self):
        truth=np.ones(7,dtype=int)
        def obs(pos):
            v=np.asarray(pos)
            return dict(row_key='same ordered rows',labels=truth,probabilities=np.column_stack((1-v,v)))
        before=obs([.55,.55,.55,.55,.45,.45,.95])
        near=obs([.05,.05,.05,.05,.95,.95,.95])
        decision=nominate(before,near,truth)
        self.assertEqual(decision['reason'],'paired_near_tie')
        self.assertGreater(decision['weight'],0)
        self.assertEqual(nominate(before,before,truth)['weight'],0)
        self.assertFalse(promoted(-2,[dict(rows=1000,gain=20),dict(rows=1000,gain=0)],primary_rows=1000))
        self.assertTrue(promoted(-2,[dict(rows=1000,gain=9),dict(rows=1000,gain=2)],primary_rows=1000))


    def test_ensemble_source_tracks_measured_quality_after_renaming_and_reordering(self):
        primary={'x':dict(score=.7,probabilities=[[.3,.7]]),'y':dict(score=.75,probabilities=[[.1,.9]])}
        caps={'x':dict(params={'view':1}),'y':dict(params={'view':2})}
        x=SimpleNamespace(family='a',params={'view':1})
        y=SimpleNamespace(family='b',params={'view':2})
        with patch('superagent.analogy_review.derived_candidate',side_effect=lambda m:m):
            selected,_=representation_source(['x','y'],[x,y],primary,caps)
            self.assertEqual(selected[1].params,{'view':2})
            renamed={'opaque2':primary['y'],'opaque1':primary['x']}
            choices={'opaque2':caps['y'],'opaque1':caps['x']}
            chosen,_=representation_source(['opaque2','opaque1'],[y,x],renamed,choices)
            self.assertEqual(chosen[1].params,selected[1].params)
            primary['x']['score']=.8
            changed,_=representation_source(['y','x'],[y,x],primary,caps)
            self.assertEqual(changed[1].params,{'view':1})
        truth=np.array([1,0])
        a=dict(row_key='f',labels=truth,probabilities=np.array([[.2,.8],[.7,.3]]))
        b=dict(a,probabilities=np.array([[.8,.2],[.3,.7]]))
        obs=incumbent_evidence([a,b],[.75,.25],SimpleNamespace(labels=[0,1]))
        np.testing.assert_allclose(obs['probabilities'],[[.35,.65],[.6,.4]])
