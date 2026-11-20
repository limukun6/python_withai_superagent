import unittest
import numpy as np

from superagent.confirmed_vote import majority
from superagent.evidence_agent import route_of
from superagent.method_choice import DIRECT, TYPED, TYPED_DIRECT, select_action
from superagent.typed_categories import TypedCategoryClassifier, column_types, parse_value


class TypedCategoryTests(unittest.TestCase):
    def test_strict_numeric_and_clock_scales(self):
        self.assertEqual(parse_value(' $1,000 - $2,000 '),('$',(1000.,2000.,1500.)))
        self.assertEqual(parse_value('−10..-2'),('number',(-10.,-2.,-6.)))
        self.assertEqual(parse_value('10% to 20%'),('%',(10.,20.,15.)))
        np.testing.assert_allclose(parse_value('at most 5')[1],[np.nan,5,5],equal_nan=True)
        np.testing.assert_allclose(parse_value('50+')[1],[50,np.nan,50],equal_nan=True)
        np.testing.assert_allclose(parse_value('12AM')[1],parse_value('00:00')[1])
        np.testing.assert_allclose(parse_value('12 PM')[1],parse_value('12:00')[1])
        for value in ('SKU12','00128','30PM','25:00','3:99','3-2','£5–€8','hi','12.3.7'):
            with self.subTest(value=value):self.assertIsNone(parse_value(value))

    def test_training_column_selection_and_unseen_values(self):
        rows=np.array([['1-3','east','03:30'],['4-6','south','12:30'],['7-9','west','18:30']]*12)
        self.assertEqual(column_types(rows),{0:'number',2:'clock'})
        model=TypedCategoryClassifier({'backbone_params':{'depth':2,'l2_leaf_reg':3}})
        model.width=3;model.types=column_types(rows)
        features=model.features(np.array([['10-12','east','00:30'],['unknown','west','missing']]))
        self.assertEqual(features.shape,(2,8))
        np.testing.assert_array_equal(features[0,3:6].astype(float),[10,12,11])
        self.assertTrue(np.isnan(features[1,3:].astype(float)).all())

    def test_decision_and_tie_resolution(self):
        ev=dict(kind='categorical_tabular',semantic='q',shared_errors=50,required_shared_errors=12,short_latin_share=1.,
                low_support_shared_errors=25,primary_metric='accuracy',corrected=35,lost=20)
        options={DIRECT,TYPED,TYPED_DIRECT}
        self.assertEqual(select_action(ev,options,'typed-values',[])[0],TYPED)
        self.assertEqual(select_action(ev,options,'typed-values',[TYPED])[0],TYPED_DIRECT)
        self.assertEqual(select_action(ev|dict(corrected=20),options,'typed-values',[TYPED])[0],DIRECT)
        self.assertEqual(route_of(TYPED),route_of(TYPED_DIRECT))
        np.testing.assert_array_equal(majority([['x','x'],['a','a'],['a','b']]),['a','x'])
