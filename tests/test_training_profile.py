"""Training facts must not inspect external labels or expose identity/raw examples."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import numpy as np

from superagent.task_profile import training_facts
from superagent.ab_flash import context_for
from tests.test_ab_flash import fixture_state


class PoisonValidation(SimpleNamespace):
    @property
    def y_val(self):raise AssertionError('External labels must never be inspected')
    @property
    def X_val(self):raise AssertionError('External features must never be inspected')


class TrainingProfileTests(unittest.TestCase):
    def test_train_only_identity_invariant_and_explicit_imbalance(self):
        task=PoisonValidation(kind='single_text',X_train=['secret sentence']*4,y_train=np.array([0,0,0,1]),spec={'metric':'macro_f1'})
        facts=training_facts(task)
        renamed=PoisonValidation(**{**vars(task),'spec':{**task.spec,'display_name':'private dataset','source_path':'secret.zip'}})
        self.assertEqual(facts,training_facts(renamed))
        self.assertEqual(facts['majority_class_fraction'],.75)
        self.assertEqual(facts['statistics']['primary_metric'],'macro_f1')
        for forbidden in ('secret sentence','private dataset','secret.zip','validation_label_counts'):
            self.assertNotIn(forbidden,json.dumps(facts))

    def test_feature_lists_are_bounded_with_visible_truncation(self):
        task=PoisonValidation(kind='numeric_tabular',X_train=np.zeros((4,100)),y_train=np.array([0,0,1,1]),spec={})
        facts=training_facts(task)
        self.assertEqual(len(facts['statistics']['missing_values_per_column']),32)
        self.assertEqual(facts['statistics']['feature_count'],100)
        self.assertEqual(facts['truncated_fields']['missing_values_per_column']['total_items'],100)

    def test_same_profile_in_both_prompts_detached_from_state(self):
        state,caps,ids=fixture_state()
        state['training_profile']={'profile_version':'fixture','statistics':{'train_rows':256}}
        raw=context_for(state,caps,ids,False);checked=context_for(state,caps,ids,True)
        self.assertEqual(raw,{k:v for k,v in checked.items() if k not in ('checks','memory')})
        raw['training_profile']['statistics']['train_rows']=1
        self.assertEqual(state['training_profile']['statistics']['train_rows'],256)

    def test_default_does_not_add_profile(self):
        state,caps,ids=fixture_state()
        self.assertNotIn('training_profile',context_for(state,caps,ids,False))


if __name__=='__main__':unittest.main()
