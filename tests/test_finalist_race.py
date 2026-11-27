import unittest

import numpy as np

from superagent.finalist_race import FinalistRace, additional_finalist, shortlist
from superagent.metrics import MetricSpec


class Recorder(dict):
    def write_json(self, key, data):
        self[key] = data


class TestFinalistRace(unittest.TestCase):
    def setUp(self):
        self.names = ["pipeline_a","pipeline_b","pipeline_c"]
        self.y = np.arange(100)%2
        self.metric = MetricSpec("accuracy",(0,1))
        self.caps = {n:dict(classifier_fits=1,refit_classifier_fits=1) for n in self.names}
        self.folds = {"primary":([],np.arange(100),{}),"fresh":([],np.arange(100,200),{})}
        self.log = Recorder()

    def row(self, fold, wrong):
        confidence = np.full(100,.95)
        confidence[wrong]=.40
        p = np.where(self.y==1,confidence,1-confidence)
        predicted = (p>.5).astype(int)
        return dict(row_key=fold,labels=self.y,predictions=predicted,probabilities=np.column_stack([1-p,p]),
                    score=float(np.mean(self.y==predicted)))

    def run_race(self, reverse):
        self.log = Recorder()
        race = FinalistRace(self.log,self.folds)
        bank = {f:{} for f in self.folds}
        for i,name in enumerate(self.names):
            bank["primary"][name] = self.row("primary",slice(20*i,20*i+20))
        decide = lambda remaining: race.consider([{"action":"stop"}],self.caps,bank,self.metric,remaining,set(),True)
        action = decide(8)
        self.assertEqual(action["action"],"confirm")
        self.assertNotIn("finalist_result.json",self.log)
        anchor=self.log["finalist_lock.json"]["anchor"][0]
        remaining=8
        while action["action"]=="confirm":
            name=action["members"][0]
            i=self.names.index(name)
            wrong = slice(0,0) if reverse and name==anchor else slice(0,100) if reverse else slice(20*i,20*i+20)
            bank["fresh"][name]=self.row("fresh",wrong)
            remaining-=1
            action=decide(remaining)
        self.assertEqual(action["action"],"stop")
        return action["members"],anchor

    def test_new_evidence_can_select_or_veto_the_same_vote(self):
        members,_=self.run_race(False)
        self.assertEqual(len(members),2)
        members,anchor=self.run_race(True)
        self.assertEqual(members,[anchor])

    def test_no_disagreement_does_not_trigger_a_race(self):
        primary={name:self.row("primary",slice(20)) for name in self.names}
        result=FinalistRace(self.log,self.folds).consider([{"action":"stop"}],self.caps,
                   dict(primary=primary,fresh={}),self.metric,6,set(),True)
        self.assertIsNone(result)

    def test_nearby_vote_can_earn_confirmation_and_spare_fit_cannot_spend_its_reserve(self):
        a, b, c = self.names
        bank = {a:self.row("primary", slice(10)),
                b:self.row("primary", list(range(10,22))+list(range(30,36))),
                c:self.row("primary", list(range(10,22))+list(range(36,42)))}
        bank[a]["probabilities"] = None
        self.assertEqual(len(shortlist(bank,self.metric,True,self.names)[1]),1)
        anchor, challenger, _, paired = shortlist(bank,self.metric,True,self.names,frontier=True)
        self.assertEqual(anchor,(a,))
        self.assertEqual(challenger,(b,c))
        self.assertEqual((paired["gained"],paired["lost"]),(10,12))
        caps = dict(self.caps, untried=dict(classifier_fits=1,refit_classifier_fits=1))
        menu = [dict(action="stop"),dict(action="explore",members=["untried"],evidence=dict(expected=.89,novelty=.4))]
        race = FinalistRace(self.log,self.folds,frontier=True)
        observed = dict(primary=bank,fresh={})
        self.assertEqual(race.consider(menu,caps,observed,self.metric,6,set(),True)["action"],"explore")
        self.assertIsNone(race.plan)
        self.assertEqual(race.consider(menu,caps,observed,self.metric,5,set(),True)["action"],"confirm")
        self.assertEqual(race.plan["reserve"],2)

    def test_second_challenger_must_fit_two_complete_comparisons_and_refit(self):
        a,b,c = self.names
        d = "additional_model"
        primary={a:self.row("primary",slice(10)),
                 b:self.row("primary",list(range(10,22))+list(range(30,36))),
                 c:self.row("primary",list(range(10,22))+list(range(36,42))),
                 d:self.row("primary",slice(24,35))}
        primary[a]["probabilities"]=None
        caps=dict(self.caps,additional_model=dict(classifier_fits=1,refit_classifier_fits=1))
        groups=[(a,),(b,),(c,),(b,c)]
        def decide(budget):
            return additional_finalist(primary,self.metric,set(primary),caps,(a,),(b,c),groups,
                                       ["one","two"],budget,True)
        self.assertIsNone(decide(9))
        self.assertIn(d,decide(10))
