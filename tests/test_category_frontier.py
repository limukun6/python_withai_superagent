import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from superagent.category_interactions import CategoryInteractionClassifier
from superagent.method_choice import COOCCUR, CROSSES, DIRECT, DIRECT_CONTEXT, MINILM, NEURAL, SEMANTIC, select_action
from superagent.models import predict_with_model


class CategoryFrontierTests(unittest.TestCase):
    def test_feedback_and_fixed_counterfactual(self):
        ev=dict(kind='categorical_tabular',semantic=SEMANTIC,shared_errors=70,required_shared_errors=10,
                short_latin_share=.92,low_support_shared_errors=38,nonlinear_gain=.07,rare_enrichment=1.1,
                previous_delta=0.,margin=.01,primary_metric='accuracy',search_rows=1200,corrected=47,lost=20,discordant=67)
        available={DIRECT,DIRECT_CONTEXT,CROSSES,MINILM,COOCCUR,NEURAL}
        self.assertEqual(select_action(ev,available,'representation',[])[0],DIRECT)
        self.assertEqual(select_action(ev,available,'representation',[DIRECT])[0],DIRECT_CONTEXT)
        self.assertEqual(select_action(ev|dict(corrected=23),available,'representation',[DIRECT])[0],CROSSES)
        self.assertEqual(select_action(ev|dict(shared_errors=3),available,'representation',[])[0],None)
        self.assertEqual(select_action(ev|dict(shared_errors=3),available,'context-control',[MINILM])[0],COOCCUR)
        text=ev|dict(kind='pair_text',short_latin_share=0.)
        self.assertEqual(select_action(text,{NEURAL},'paired',[])[0],select_action(text,{NEURAL},'representation',[])[0])

    def test_pair_dictionary_handles_repeats_unseen_levels_and_nonbinary_labels(self):
        rows=np.array([['a','b'],['b','a'],['x','a'],['a','a']]*30)
        y=np.array(['east','west','north','north']*30)
        model=CategoryInteractionClassifier(.1).fit(rows,y)
        np.testing.assert_array_equal(model.predict(rows),y)
        held=np.array([['unseen','a'],['a','unseen'],['unseen','unseen'],['b','b']])
        features=model.matrix(model.encoder.transform(held))
        self.assertEqual(features.getnnz(axis=1).tolist(),[1,1,0,2])
        np.testing.assert_allclose(model.predict_proba(held).sum(axis=1),1.)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'classifier.joblib'
            joblib.dump(model,file)
            np.testing.assert_array_equal(predict_with_model('categorical_tabular',joblib.load(file),held),model.predict(held))
