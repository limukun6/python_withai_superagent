from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from superagent.autonomous_run import partitions
from superagent.data import TrainingTask
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.proposal_value import ProposalStudy, checked_reply, fragile, split_search


class ProposalValueTests(unittest.TestCase):
    def test_fallback_and_nested_partitions_keep_groups_out_of_every_fit(self):
        x=np.repeat(np.arange(120),2).reshape(-1,1)
        task=TrainingTask('numeric_tabular',x,(x[:,0]%2).astype(int),{})
        inner,folds,sealed,record=split_search(task,273,partitions)
        self.assertEqual(record['status'],'sealed')
        self.assertEqual(set(x[sealed,0]) & set(inner.X_train[:,0]),set())
        for fit,held,_ in folds.values():
            self.assertFalse(set(sealed)&set(fit))
            self.assertFalse(set(sealed)&set(held))
        task=TrainingTask('numeric_tabular',np.arange(8).reshape(-1,1),np.arange(8)%2,{})
        def small(t,s):
            if len(t.y_train)<8:
                raise ValueError('Too few classes')
            return {'primary':(np.array([0,1]),np.array([2,3]),{})}
        unused,_,sealed,record=split_search(task,1,small)
        self.assertIs(unused,task)
        self.assertEqual((len(sealed),record['status']),(0,'unavailable'))

    def test_weak_evidence_selects_fresh_check_and_support_can_pass(self):
        self.assertTrue(fragile([],[]))
        self.assertTrue(fragile([.016,.002,.102],[60,60,60]))
        self.assertTrue(fragile([-.01,.03],[40,40]))
        self.assertFalse(fragile([.04,.05,.03],[60,60,60]))

    def test_proposal_must_cite_actual_trials_and_make_a_falsifiable_reachable_claim(self):
        pkt=dict(reference_trials=[2,5],shared_errors=5,reference_score=.7,options=[{'name':'a'},{'name':'b'}])
        reply=dict(decision='experiment',option=1,cited_trials=[5,2],min_rescues=2,min_gain=.04,
                   hypothesis='Shared errors depend on interactions.',refute_if='Less than two rescues.')
        self.assertEqual(checked_reply(reply,pkt)['candidate'],'b')
        for bad in (dict(cited_trials=[2,8]),dict(min_rescues=6),dict(min_gain=.8),dict(min_gain=float('nan')),
                    dict(option=-1),dict(refute_if=''),dict(option=True)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                checked_reply(dict(reply,**bad),pkt)
        self.assertEqual(checked_reply({'decision':'keep','reason':'Already scheduled.'},pkt)['decision'],'keep')

    def test_independent_confirmation_binds_the_frozen_models_and_can_go_either_way(self):
        for expected in (0,1):
            with self.subTest(expected=expected),TemporaryDirectory() as tmp:
                folder=Path(tmp)
                log=RunLogger(folder)
                pilot=ProposalStudy('evidence-local',folder,log,None,time.monotonic(),600)
                x=np.zeros((12,1))
                task=TrainingTask('numeric_tabular',x,np.array([0,1]*3+[expected]*6),{})
                metric=MetricSpec('accuracy',(0,1))
                files={}
                for name,c in [('a',0),('b',1)]:
                    path=folder/(name+'.joblib')
                    joblib.dump(dict(model=DummyClassifier(strategy='constant',constant=c).fit([[0],[0]],[0,1])),path)
                    files[name]=path
                bank={n:dict(score=.5,predictions=np.full(6,c),probabilities=None,labels=task.y_train[:6],row_key='rows')
                      for n,c in [('a',0),('b',1)]}
                selected=dict(members=['b'])
                final=pilot.guard(task,metric,selected,'a',files,{'primary':bank},np.arange(6,12))
                self.assertEqual(final['members'],['b' if expected else 'a'])
                self.assertTrue((folder/'guard_choice_before_labels.json').exists())
                self.assertTrue(pilot.record['seal_opened'])

    def test_async_reply_can_arrive_late_without_becoming_an_active_plan(self):
        from concurrent.futures import Future
        with TemporaryDirectory() as tmp:
            pilot=ProposalStudy('proposal-flash',Path(tmp),RunLogger(Path(tmp)),Mock(),time.monotonic(),600)
            future=Future()
            pilot.future,pilot.packet=future,{}
            pilot.record.update(reply_status='pending',proposal_used=False)
            self.assertIsNone(pilot.collect(0))
            future.set_result(dict(decision='keep',reason='Insufficient support'))
            self.assertEqual(pilot.collect(0)['decision'],'keep')
            self.assertFalse(pilot.record['proposal_used'])
