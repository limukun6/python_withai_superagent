import unittest

import numpy as np

from superagent.experiment_policy import actions, identity
from superagent.metrics import MetricSpec


class DisplayTieTests(unittest.TestCase):
    def test_a_tied_exploration_uses_the_capabilities(self):
        profile=dict(majority_class_fraction=.5,minority_class_fraction=.5,statistics={"train_rows":600})
        metric=MetricSpec("accuracy",(0,1))
        specs=["first classifier","second classifier","third classifier"]
        def first(alias,reverse):
            keys=list(reversed(specs)) if reverse else specs
            caps={alias(k):dict(id=alias(k),route=alias(k),params={},representation="scaled numeric columns",estimator=k,
                               resource="cpu",classifier_fits=1,refit_classifier_fits=1) for k in keys}
            menu=actions(caps,profile,[],{"primary":{}},metric,["primary"],4)
            self.assertEqual(len({a["utility"] for a in menu}),1)
            return caps[menu[0]["members"][0]]["estimator"]
        reference=first(str,False)
        for i in range(12):
            self.assertEqual(first(lambda k:identity([i,k]),True),reference)

    def test_deployment_keeps_an_advantage_in_real_predictions(self):
        labels=np.arange(40)%2
        def obs(wrong):
            pred=labels.copy()
            pred[:wrong]=1-pred[:wrong]
            prob=np.column_stack([1-pred,pred])
            return dict(row_key="one",score=float(np.mean(labels==pred)),labels=labels,predictions=pred,probabilities=prob)
        caps={name:dict(params={},representation="numeric",estimator=algo,classifier_fits=1,refit_classifier_fits=1)
              for name,algo in (("x","linear"),("y","tree"))}
        profile=dict(majority_class_fraction=.5,minority_class_fraction=.5,statistics={"train_rows":600})
        def selected(bank):
            return actions(caps,profile,[],{"primary":bank},MetricSpec("accuracy",(0,1)),["primary"],1)[0]["members"]
        self.assertEqual(selected({"x":obs(3),"y":obs(5)}),["x"])
        self.assertEqual(selected({"x":obs(5),"y":obs(3)}),["y"])
