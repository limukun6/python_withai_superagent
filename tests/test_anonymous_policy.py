import unittest
from unittest.mock import Mock

import numpy as np

from superagent.autonomous_run import choose_with_flash, partitions
from superagent.data import TrainingTask
from superagent.experiment_policy import actions, measured_options
from superagent.metrics import MetricSpec
from superagent.native_schema import declared_preference


class AnonymousPolicyTests(unittest.TestCase):
    profile = dict(statistics=dict(train_rows=80),majority_class_fraction=.5,minority_class_fraction=.5)
    metric = MetricSpec('accuracy',(0,1))
    caps = {n:dict(estimator='linear',representation=n,params={},resource='cpu',
                   classifier_fits=1,refit_classifier_fits=1) for n in ('red','blue','green')}

    def observation(self, predicted, fold='f'):
        y=np.tile([0,1],20)
        return dict(labels=y,score=float(np.mean(predicted==y)),predictions=predicted,row_key=fold,
                    probabilities=np.eye(2)[predicted])

    def test_counterfactual_scores_change_winner_and_ignore_menu_order(self):
        y=np.tile([0,1],20)
        good,bad=self.observation(y),self.observation(1-y)
        def pick(store,caps):
            return actions(caps,self.profile,[],{'primary':store},self.metric,['primary'],1)[0]
        self.assertEqual(pick({'red':good,'blue':bad},self.caps)['members'],['red'])
        self.assertEqual(pick({'red':bad,'blue':good},self.caps)['members'],['blue'])
        self.assertEqual(pick({'red':good,'blue':bad},dict(reversed(self.caps.items())))['members'],['red'])
        seen={'primary':{'red':bad},'confirmation_1':{}}
        self.assertEqual(actions(self.caps,self.profile,[],seen,self.metric,list(seen),3)[0]['action'],'explore')
        seen['primary']['red']=good
        self.assertEqual(actions(self.caps,self.profile,[],seen,self.metric,list(seen),3)[0]['action'],'confirm')

    def test_disjoint_validation_groups_for_repeated_inputs(self):
        task=TrainingTask('numeric_tabular',np.repeat(np.arange(60)[:,None],2,axis=0),np.repeat(np.arange(60)%3,2),{})
        folds=partitions(task,144)
        held=set()
        for a,b,_ in folds.values():
            self.assertFalse(set(task.X_train[a,0]) & set(task.X_train[b,0]))
            self.assertFalse(held & set(b))
            held.update(b.tolist())

    def test_unnamed_pairs_require_observable_direction(self):
        names=['left_text','right_text']
        self.assertEqual(declared_preference('Label 1 means that the left alternative was preferred and label 0 means that the right was preferred.', 'label', names),names)
        self.assertEqual(declared_preference('Label 1 means right wins and 0 means left wins.', 'label', names),names[::-1])
        self.assertIsNone(declared_preference('Does the pair express the same concept?', 'label', names))
        self.assertIsNone(declared_preference('Label 1 means left wins. Label 1 means right wins.', 'label', names))

    def test_unmatched_prediction_rows_and_fabricated_actions_rejected(self):
        y=np.tile([0,1],20)
        with self.assertRaises(ValueError):
            measured_options({'primary':{'red':self.observation(y), 'blue':self.observation(y,'g')}},self.metric)
        menu=actions(self.caps,self.profile,[],{},self.metric,['primary'],3)
        client=Mock()
        client.ask.return_value=dict(option=0,reason='explore',evidence_trials=[14])
        with self.assertRaises(ValueError):
            choose_with_flash(client,Mock(),menu,self.caps,self.profile,self.metric,[],3)

    def test_no_extra_fit_or_single_fold_vote(self):
        y=np.tile([0,1],20)
        seen={'primary':{'red':self.observation(y),'blue':self.observation(y)}}
        choices=actions(self.caps,self.profile,[],seen,self.metric,['primary'],1)
        self.assertTrue(all(a['action']=='stop' and len(a['members'])==1 for a in choices))
