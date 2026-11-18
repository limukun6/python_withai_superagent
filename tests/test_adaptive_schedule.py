import unittest
import numpy as np
from superagent.evidence_schedule import schedule,duel,catalog_for

class AdaptiveTests(unittest.TestCase):
    def run_case(self,score):
        records=[];seen=set()
        def sample(name,split):
            self.assertNotIn((name,split),seen);seen.add((name,split))
            return score(name,split)
        catalog=catalog_for('categorical_tabular')
        choice=schedule(catalog,sample,records.append)
        self.assertLessEqual(len(seen),10)
        self.assertGreaterEqual(len({name for name,split in seen if split==0}),4)
        return choice,records

    def test_identical_predictions_stop_without_spending_replication_fits(self):
        winner,events=self.run_case(lambda name,split:np.arange(100)<60)
        self.assertEqual(winner,'logistic_C0.05')
        self.assertEqual(events[-1]['search_fits'],4)

    def test_primary_gain_that_reverses_does_not_displace_incumbent(self):
        def scores(name,split):
            n=60 if name=='logistic_C0.05' else (65 if split==0 else 50)
            return np.arange(100)<n
        winner,events=self.run_case(scores)
        self.assertEqual(winner,'logistic_C0.05')
        self.assertTrue(any(e.get('first_gain_failed_recheck') for e in events))
        self.assertTrue(any(e.get('split')==2 for e in events))

    def test_replicated_gain_can_replace_incumbent(self):
        anchor='logistic_C0.05'
        winner,events=self.run_case(lambda name,split:np.arange(100)<(50 if name==anchor else 65))
        self.assertNotEqual(winner,anchor)
        self.assertTrue(any(e.get('promoted') for e in events))

    def test_reject_misaligned_feedback(self):
        with self.assertRaises(ValueError):duel(np.array([True]),np.array([True,False]))
