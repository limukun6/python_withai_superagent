import copy
import unittest

from superagent.local_search import LocalSearchPolicy


def catalog():
    return [{'id':f'{route}_C{c}','route':route,'params':{'C':c}}
            for route in ('word','word_char','word_char_stats','nb_word','char','word_char_stats_svm') for c in (.05,.2,1.)]


def observe(cid,score,rows,status='completed'):
    route=next(r['route'] for r in rows if r['id']==cid)
    return {'candidate':cid,'route':route,'status':status,'search_accuracy':score}


class InterleavedPolicyTests(unittest.TestCase):
    def test_third_choice_can_react_to_scores_before_all_routes_are_covered(self):
        rows=catalog();policy=LocalSearchPolicy('interleaved_routes',rows,11)
        a,_=policy.choose([]);history=[observe(a,.6,rows)]
        b,_=policy.choose(history);history.append(observe(b,.8,rows))
        first=copy.deepcopy(policy);second=copy.deepcopy(policy)
        chosen,reason=first.choose(history)
        self.assertEqual(next(r['route'] for r in rows if r['id']==chosen),history[1]['route'])
        changed=copy.deepcopy(history);changed[0]['search_accuracy']=.9
        alternative,_=second.choose(changed)
        self.assertEqual(next(r['route'] for r in rows if r['id']==alternative),history[0]['route'])
        self.assertNotEqual(chosen,alternative);self.assertEqual(reason,'interleaved_feedback_priority')
        self.assertEqual(len({r['route'] for r in history}),2)

    def test_alternation_keeps_exploration_and_never_repeats_a_candidate(self):
        rows=catalog();policy=LocalSearchPolicy('interleaved_routes',rows,7);history=[];reasons=[]
        for index in range(8):
            chosen,reason=policy.choose(history);self.assertNotIn(chosen,[r['candidate'] for r in history])
            history.append(observe(chosen,.6+.01*(index%3),rows));reasons.append(reason)
        self.assertEqual(reasons[0],'shared_anchor')
        for index in (1,3,5,7):self.assertEqual(reasons[index],'interleaved_explore_route')
        for index in (2,4,6):self.assertEqual(reasons[index],'interleaved_feedback_priority')
        self.assertEqual(len({r['route'] for r in history}),5)

    def test_failed_or_exhausted_seen_routes_do_not_prevent_discovery(self):
        rows=catalog();policy=LocalSearchPolicy('interleaved_routes',rows,3)
        history=[observe(rows[0]['id'],None,rows,'failed'),observe(rows[3]['id'],None,rows,'timeout')]
        chosen,reason=policy.choose(history)
        self.assertEqual(reason,'interleaved_explore_route');self.assertNotIn(chosen,[r['candidate'] for r in history])
        singletons=[{'id':f'route{i}','route':f'route{i}','params':{}} for i in range(3)]
        policy=LocalSearchPolicy('interleaved_routes',singletons,1)
        history=[observe('route0',.8,singletons),observe('route1',.7,singletons)]
        self.assertEqual(policy.choose(history)[0],'route2')

    def test_new_policy_does_not_mutate_evidence_or_prune_other_routes(self):
        rows=catalog();policy=LocalSearchPolicy('interleaved_routes',rows,5)
        history=[observe(rows[0]['id'],.8,rows),observe(rows[3]['id'],.6,rows)]
        before=copy.deepcopy(history);catalog_before=copy.deepcopy(rows)
        policy.choose(history)
        self.assertEqual(history,before);self.assertEqual(rows,catalog_before);self.assertEqual(policy.catalog,rows)


if __name__=='__main__':unittest.main()
