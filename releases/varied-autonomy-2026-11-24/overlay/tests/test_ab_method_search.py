import hashlib
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from superagent import ab_sentence
from superagent.ab_catalog import capabilities_for
from superagent.ab_method_search import rank
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.ab_varied_preferences import BINARY, TOPIC, DirectionalWords, TopicStyleFeatures
from superagent.models import search_candidates


def example():
    cap = lambda method, rep, c: dict(estimator=method, representation=rep, resource='cpu', params={'C':c})
    caps = {'left':cap('logistic','word',1.), 'right':cap('linear_svm','char',1.),
            'u':cap('logistic','word',.9), 'v':cap('linear_svm','char',.9)}
    rows = [dict(trial=1, candidate='left', search_score=.9), dict(trial=2, candidate='right', search_score=.5)]
    state = {'training_profile':{'statistics':{'train_rows':128}, 'majority_class_fraction':.5},
             'prediction_evidence':[{'trial':1,'predictions':[0,0,1,1]}, {'trial':2,'predictions':[1,1,0,0]}]}
    return state, caps, rows


class TestLocalMethodSearch(unittest.TestCase):
    def test_first_choice_ignores_identifiers_and_catalogue_order(self):
        state, caps, _ = example()
        first = rank(state,caps,tuple(caps),[])
        rename = {name:str(i) for i,name in enumerate(reversed(caps))}
        changed = {rename[k]:v for k,v in reversed(caps.items())}
        other = rank(state,changed,tuple(reversed(changed)),[])
        self.assertEqual(rename[first['chosen']],other['chosen'])
        self.assertEqual([v['config_sha256'] for v in first['candidates']],
                         [v['config_sha256'] for v in other['candidates']])

    def test_reversing_observed_performance_changes_next_model(self):
        state,caps,observed = example()
        first = rank(state,caps,['u','v'],observed)
        changed = deepcopy(observed)
        changed[0]['search_score'],changed[1]['search_score'] = .5,.9
        other = rank(state,caps,['v','u'],changed)
        self.assertEqual((first['chosen'],other['chosen']),('u','v'))
        self.assertEqual(first['evidence_trials'],[1,2])
        revised = deepcopy(state)
        revised['prediction_evidence'][1]['predictions'] = [0,0,1,1]
        self.assertLess(rank(revised,caps,['u','v'],observed)['exploration_weight'],first['exploration_weight'])

    def test_only_feasible_entries_can_win(self):
        state,caps,rows = example()
        self.assertEqual(rank(state,caps,['v'],rows)['chosen'],'v')
        with self.assertRaises(ValueError):
            rank(state,caps,[],rows)

    def test_enlarged_menu_is_a_strict_executable_superset(self):
        spec={'contract_version':1,'objective':'classification','kind':'pair_text',
              'metric':{'name':'accuracy','labels':[0,1],'direction':'maximize'},
              'input_schema':{'modality':'pair_text','pair_semantics':'preference','pair_label_one':'left'}}
        with patch.object(ab_sentence,'verified_directory',return_value='fixture'), \
                patch.object(ab_sentence,'verified_files',return_value='fixture'), patch.dict(os.environ):
            os.environ[PREFERENCE_CATALOG_ENV]='deep-e5-familiarity'
            old=capabilities_for(spec,train_rows=100)
            os.environ[PREFERENCE_CATALOG_ENV]='varied'
            new=capabilities_for(spec,train_rows=100)
            worker=search_candidates('pair_text','all')
            self.assertEqual(old,{key:new[key] for key in old})
            from importlib.util import find_spec
            extra={BINARY,TOPIC,'word_char_stats_svm_C0.2','nb_word_C0.2'}
            if find_spec('catboost'):
                extra.add('style_tree_depth3')
            self.assertEqual(set(new)-set(old),extra)
            for name in new:
                matches=[c for c in worker if c.family==name]
                self.assertEqual(len(matches),1,name)
                self.assertEqual(new[name]['params'],matches[0].params)
            spec['metric']['name']='roc_auc'
            self.assertNotIn('word_char_stats_svm_C0.2',capabilities_for(spec,train_rows=100))

    def test_binary_and_topic_features_respect_pair_order(self):
        rows=[('red river rises','red river falls'),('green field stays','green field changes'),
              ('old town rises','old town falls'),('light rain stays','light rain changes')]
        reverse=[(b,a) for a,b in rows]
        words=DirectionalWords().fit(rows)
        a,b=words.transform(rows).toarray(),words.transform(reverse).toarray()
        n=a.shape[1]//2
        np.testing.assert_array_equal(a,b[:,np.r_[n:2*n,0:n]])
        def fake_encode(texts,deep=False):
            vectors=[np.frombuffer(hashlib.sha256(s.encode()).digest()[:12],dtype=np.uint8) for s in texts]
            return np.asarray(vectors,dtype=np.float64),len(texts)
        with patch.object(ab_sentence,'encode',side_effect=fake_encode):
            model=TopicStyleFeatures('pair_text').fit(rows)
            values=model.transform(rows)
            self.assertGreater(model.components_.n_components_,0)
            np.testing.assert_allclose(values.toarray(),-model.transform(reverse).toarray(),atol=1e-10)
            equal=model.transform([('red river rises','red river rises')])
            self.assertEqual(equal.count_nonzero(),0)


if __name__=='__main__':
    unittest.main()
