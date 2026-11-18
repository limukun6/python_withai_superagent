import json
import unittest
from superagent.confirmation_memory import ConfirmationMemory,PersistentSearch


def catalog():return [{'id':f'{route}{i}','route':route,'params':{'C':10.**i}}
                      for route in ('a','b','c') for i in range(4)]


class MemoryTests(unittest.TestCase):
    def test_confirmed_best_survives_new_single_split_high_score(self):
        memory=ConfirmationMemory('fixed',[1,2,3])
        for seed in (1,2,3):memory.add('old',seed,'completed',.75)
        memory.add('new',1,'completed',.9)
        self.assertEqual(memory.incumbent(),('old',.75))
        for seed in (2,3):memory.add('new',seed,'completed',.6)
        self.assertEqual(memory.incumbent(),('old',.75))

    def test_rejects_rewrite_unknown_split_and_cross_protocol_restore(self):
        memory=ConfirmationMemory('fixed',[1,2,3]);memory.add('x',1,'failed')
        with self.assertRaises(ValueError):memory.add('x',1,'completed',.9)
        with self.assertRaises(ValueError):memory.add('x',4,'completed',.9)
        with self.assertRaises(ValueError):ConfirmationMemory.restore(memory.snapshot(),'different')
        with self.assertRaises(ValueError):memory.add('x',2,'completed',float('nan'))

    def test_failure_cannot_become_implicit_zero_or_complete_winner(self):
        memory=ConfirmationMemory('fixed',[1,2,3]);memory.add('x',1,'failed')
        for seed in (2,3):memory.add('x',seed,'completed',.99)
        self.assertEqual(memory.incumbent(),(None,None))

    def test_mid_confirmation_json_resume_matches_uninterrupted_queries(self):
        first=PersistentSearch(catalog(),'protected_routes',11,[1,2,3],1,'fixed')
        for _ in range(6):
            q=first.next_query();first.observe('completed',.5+int(q['candidate'][-1])*.03)
        self.assertTrue(first.pending)
        second=PersistentSearch.restore(json.loads(json.dumps(first.snapshot())),'fixed')
        for _ in range(14):
            a,b=first.next_query(),second.next_query();self.assertEqual(a,b)
            self.assertEqual(a,first.next_query())
            if a is None:break
            score=.5+int(a['candidate'][-1])*.03
            self.assertEqual(first.observe('completed',score),second.observe('completed',score))
        self.assertEqual(first.query_count,second.query_count)
        self.assertEqual(first.memory.snapshot(),second.memory.snapshot())

    def test_incumbent_is_monotonic_and_every_confirmation_counts(self):
        engine=PersistentSearch(catalog(),'random',31,[1,2,3],1,'fixed')
        prior=-1.
        stages=[]
        for i in range(20):
            query=engine.next_query();stages.append(query['stage'])
            _,score=engine.observe('completed',.6+int(query['candidate'][-1])*.02-query['seed']*.01)
            if score is not None:self.assertGreaterEqual(score,prior);prior=score
            self.assertEqual(engine.query_count,i+1)
        self.assertEqual(stages[:5],['primary']*5)
        self.assertEqual(stages[5:11],['confirmation']*6)


if __name__=='__main__':unittest.main()
