import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from superagent.evidence_agent import route_of
from superagent.models import predict_with_model, probability_with_model
from superagent.pretrained_choice import consider, budget
from superagent.pricing import resolve_pricing
from superagent.tabular_encoder import LevelClassifier, MODEL, PERMUTED, SEMANTIC, eligible_columns
from superagent.task_budget import BudgetStateError, TaskBudgetLedger


class FixtureEncoder:
    def __init__(self):
        self.seen = []

    def encode(self, names):
        self.seen.extend(names)
        rng = np.random.default_rng(120)
        return rng.normal(size=(len(names), 8))


class CategoryTransferTests(unittest.TestCase):
    def test_gate_uses_training_schema_shared_errors_and_lower_support(self):
        counts = ['Factory']*25+['Office']*10+['Cafe']*5
        train = np.array(list(zip(counts, range(40))))
        labels = np.zeros(20, dtype=int)
        search = np.array([['Office','200']]*10 + [['Factory','200']]*10)
        history = [dict(trial=i, candidate=f'logistic_C{c}', params={'C':c}, status='completed',
                        search_score=score, primary_metric='accuracy') for i,c,score in [(1,.25,.1),(2,1.,.2)]]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in (1,2):
                joblib.dump({'search_predictions':1-labels},root/f'trial_{i}.joblib')
            with patch('superagent.tabular_encoder.budget') as charge:
                def decide(features=train, observations=search, **options):
                    return consider('categorical_tabular',history,root,labels,dict(tabular_weights='fixture',**options),features,observations)
                decision = decide()
                self.assertEqual(decision['candidate'],SEMANTIC)
                self.assertEqual(decision['params']['backbone'],'logistic_C1.0')
                self.assertEqual(decision['params']['columns'],[0])
                self.assertEqual(decision['shared_errors_with_low_support_level'],10)
                self.assertEqual(charge.call_args.kwargs['model_id'],MODEL)
                self.assertEqual(decide(level_assignment='permuted')['candidate'], PERMUTED)
                self.assertEqual(decide(observations=np.tile(['Factory','202'],(20,1)))['decision'],'skip')
                self.assertEqual(decide(features=train[:,1:],observations=search[:,1:])['decision'],'skip')
                charge.side_effect=BudgetStateError('cap')
                self.assertEqual(decide()['reason'],'cap')
                joblib.dump({'search_predictions':labels},root/'trial_1.joblib')
                self.assertEqual(decide()['reason'],'few_shared_errors')
        self.assertEqual(route_of(SEMANTIC),route_of(PERMUTED))

    def test_features_are_fold_local_permutable_and_predict_without_encoder(self):
        raw = np.array([['rest','1'],['work','0'],['study','1'],['play','0']]*8)
        labels = np.tile([0,1,1,0],8)
        extra = np.array([['unseen','7'],['?','8']])
        params = dict(columns=[0],dimensions_per_column=3,backbone='logistic_C1.0',backbone_params={'C':1.})
        for_model = {}
        for assignment in ('semantic','permuted'):
            encoder = FixtureEncoder()
            model = LevelClassifier(dict(params,assignment=assignment)).fit(raw,labels,encoder)
            self.assertEqual(set(encoder.seen),set(raw[:,0]))
            self.assertEqual(len(encoder.seen), len(set(encoder.seen)))
            self.assertTrue((model.features(extra).toarray()[:,-3:]==0.).all())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model'
                joblib.dump(model,path)
                loaded = joblib.load(path)
                with patch('superagent.tabular_encoder.LevelEncoder.load',side_effect=AssertionError('download in inference')):
                    np.testing.assert_array_equal(predict_with_model('categorical_tabular',loaded,extra),model.predict(extra))
                    classes, probs = probability_with_model('categorical_tabular',loaded,extra)
                    self.assertEqual(probs.shape,(2,len(classes)))
            for_model[assignment] = model.lookups[0]
        real, shuffled = for_model.values()
        self.assertEqual({tuple(x) for x in real.values()}, {tuple(x) for x in shuffled.values()})
        self.assertTrue(all(not np.array_equal(real[s], shuffled[s]) for s in real))
        numeric = np.array([['1','001'],['2','002'],['3','003']]*4)
        self.assertEqual(eligible_columns(numeric),[])

    def test_official_embedding_tariff_and_hard_limit(self):
        self.assertEqual(resolve_pricing('ops-qwen3-embedding-0.6b').course_cost(1000,0),.005)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'ledger'
            ledger = TaskBudgetLedger.initialize(path,new_project=True)
            config = dict(ledger=path,scope='fixture',run='one',cap_rmb=.001)
            key = budget(config,200,reserve=True,model_id=MODEL)
            ledger.settle(key,MODEL,200,0)
            with self.assertRaises(BudgetStateError):
                budget(config,1,model_id=MODEL)


if __name__=='__main__':
    unittest.main()
