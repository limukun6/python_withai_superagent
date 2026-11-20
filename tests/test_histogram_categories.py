import unittest
import numpy as np

from superagent.evidence_agent import route_of
from superagent.histogram_categories import HistogramCategoryClassifier
from superagent.method_choice import HIST,HIST_ADJUSTED,COOCCUR,select_action
from superagent.models import probability_with_model


class HistogramCategoryTests(unittest.TestCase):
    def test_capped_categories_multiclass_and_unseen_rows(self):
        rows=np.array([[str(i%160),str(i%3)] for i in range(480)])
        labels=np.array(['left','middle','right']*160)
        model=HistogramCategoryClassifier(dict(leaf_nodes=7,minimum_leaf=4,l2=2.,iterations=12)).fit(rows,labels)
        new=np.array([['new','1'],['new','2'],['3','unseen']])
        classes,prob=probability_with_model('categorical_tabular',model,new)
        np.testing.assert_allclose(prob.sum(axis=1),1.)
        np.testing.assert_array_equal(model.predict(new),classes[prob.argmax(axis=1)])

    def test_large_paired_loss_leaves_histogram_family(self):
        ev=dict(kind='categorical_tabular',semantic='q',shared_errors=100,required_shared_errors=20,
                short_latin_share=.9,low_support_shared_errors=40,primary_metric='accuracy',corrected=10,lost=40)
        methods=[HIST,HIST_ADJUSTED,COOCCUR]
        self.assertEqual(select_action(ev,methods,'histogram',[])[0],HIST)
        self.assertEqual(select_action(ev,methods,'histogram',[HIST])[0],COOCCUR)
        self.assertEqual(select_action(ev|dict(corrected=35),methods,'histogram',[HIST])[0],HIST_ADJUSTED)
        self.assertEqual(route_of(HIST),route_of(HIST_ADJUSTED))
