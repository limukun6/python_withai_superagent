import unittest
from unittest.mock import patch

from superagent.coverage_review import CoverageReview


class CoverageReviewTests(unittest.TestCase):
    def setUp(self):
        self.folds = ["primary","a","b"]
        self.caps = {n:dict(classifier_fits=1,refit_classifier_fits=1) for n in ("old","other")}
        self.stop = dict(action="stop",members=["old"],utility=.70,split=None,
                         evidence=dict(scores={f:.7 for f in self.folds},refit_fits=1))
        self.probe = dict(action="explore",members=["other"],split="primary",utility=.65,
                          evidence=dict(novelty=.96,reserved_refits=1))
        self.reviewer = CoverageReview()
        self.observed = {f:{"old":{"labels":list(range(20))}} for f in self.folds}
        self.options = {("old",):{f:.7 for f in self.folds}}

    def decide(self, remaining=10):
        with patch("superagent.coverage_review.measured_options",return_value=self.options):
            return self.reviewer.choose(self.stop,[self.stop,self.probe],self.caps,self.observed,None,
                                        self.folds,remaining,set(),300,[])

    def test_uncorroborated_primary_improvement_retains_original(self):
        self.assertEqual(self.decide()["action"],"explore")
        self.options[("other",)] = {"primary":.9}
        self.observed["primary"]["other"] = {"labels":[0]*20}
        self.assertEqual(self.decide()["split"],"a")
        self.options[("other",)]["a"] = .65
        self.observed["a"]["other"] = {"labels":[0]*20}
        self.assertEqual(self.decide()["split"],"b")
        self.options[("other",)]["b"] = .85
        self.observed["b"]["other"] = {"labels":[0]*20}
        self.assertEqual(self.decide()["members"],["old"])

    def test_corroborated_challenger_becomes_the_deliverable(self):
        self.decide()
        self.options[("other",)] = {f:.75 for f in self.folds}
        for f in self.folds:
            self.observed[f]["other"] = {"labels":[0]*20}
        self.assertEqual(self.decide()["members"],["other"])

    def test_no_surplus_no_probe(self):
        self.assertEqual(self.decide(2), self.stop)
        self.assertEqual(self.reviewer.probes, [])
