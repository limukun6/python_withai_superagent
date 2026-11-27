from unittest import TestCase
from types import SimpleNamespace

import numpy as np

from superagent.autonomous_run import partitions
from superagent.near_full_confirmation import partition,shortlist


class NearFinalSizeTests(TestCase):
    def test_reserve_never_enters_a_search_trial(self):
        x=np.repeat(np.arange(1600),2)[:,None]
        y=(x[:,0]%2).astype(int)
        task=SimpleNamespace(kind="numeric_tabular",X_train=x,y_train=y,spec={})
        inner,search,held,info=partition(task,19,partitions)
        self.assertEqual(info["status"],"sealed")
        self.assertEqual(len(inner.y_train)+len(held),len(y))
        for fit,check,_ in search.values():
            self.assertFalse(set(x[held,0]).intersection(x[np.concatenate([fit,check]),0]))

    def test_retraining_choices_respect_reserved_full_refits(self):
        scores={"primary":.75,"confirmation_1":.75}
        def stop(members,u):
            return dict(action="stop",members=members,utility=u,evidence={"scores":scores})
        caps={m:dict(classifier_fits=1,refit_classifier_fits=1) for m in "abcd"}
        lead=stop(["a","b"],.75)
        menu=[lead,stop(["b"],.748),stop(["a","c"],.744),stop(["c","d"],.742)]
        proposed=shortlist(lead,menu,caps,4)
        self.assertEqual([v["members"] for v in proposed],[["a","b"],["b"]])
        self.assertEqual(len(shortlist(lead,menu,caps,5)),3)
