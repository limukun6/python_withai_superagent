import unittest
import numpy as np
from catboost import CatBoostClassifier

from superagent.boosting_curve import measured_curve,suggested_trees
from superagent.evidence_agent import route_of
from superagent.method_choice import HORIZON,SHRINKAGE


class BoostingCurveTests(unittest.TestCase):
    def test_read_prefixes_from_one_real_fit(self):
        x=np.array([['a','z'],['b','y'],['c','z'],['a','x']]*12)
        y=np.array([0,1,1,0]*12)
        model=CatBoostClassifier(iterations=16,depth=2,verbose=False,thread_count=1,allow_writing_files=False)
        model.fit(x,y,cat_features=[0,1])
        measured=measured_curve(model,x,y)
        self.assertEqual([r['trees'] for r in measured],[4,8,12,16])
        self.assertTrue(all(np.isfinite(r['log_loss']) for r in measured))
        self.assertEqual(measured[-1]['accuracy'],float((model.predict(x).reshape(-1)==y).mean()))
        self.assertEqual(route_of(HORIZON),route_of(SHRINKAGE))

    def test_flat_worsening_and_improving_loss(self):
        make=lambda v:[dict(trees=i*150,log_loss=x) for i,x in enumerate(v,1)]
        self.assertEqual(suggested_trees(make([.6,.5,.55,.57])),300)
        self.assertEqual(suggested_trees(make([.6,.57,.55,.54])),900)
        self.assertIsNone(suggested_trees(make([.6,.57,.55,.5499])))
