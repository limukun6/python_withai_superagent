import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from superagent.data import TrainingTask
from superagent.encoder_mixture import EncoderMixture, bind_text_prediction, learn_from_confirmations
from superagent.method_choice import TEXT_MINILM, TEXT_DEEP
from superagent.minilm_features import MiniLMTextClassifier, DeeperMiniLMTextClassifier
from superagent.models import predict_with_model, probability_with_model


class ConstantText(MiniLMTextClassifier):
    def predict_proba(self, rows):return np.tile([.35,.65],(len(rows),1))


class ConstantDeep(DeeperMiniLMTextClassifier):
    def predict_proba(self, rows):return np.tile([.75,.25],(len(rows),1))


class LiveMixtureTests(unittest.TestCase):
    def test_two_weight_directories_and_pair_prediction_are_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            small,deep=ConstantText('pair_text','previous',False),ConstantDeep('pair_text','previous',False)
            small.classes_=deep.classes_=np.array(['away','home'])
            model=EncoderMixture(small,deep,.75)
            settings={'minilm_weights':root/'a','deep_minilm_weights':root/'b'}
            bind_text_prediction(model,settings,root/'cache')
            self.assertEqual(small.runtime,deep.runtime)
            self.assertEqual(set(json.loads(Path(deep.runtime).read_text())),set(settings))
            inputs=[('left','right'),('slow','fast')]
            np.testing.assert_allclose(probability_with_model('pair_text',model,inputs)[1],[[.45,.55],[.45,.55]])
            np.testing.assert_array_equal(predict_with_model('pair_text',model,inputs),['home','home'])
            with self.assertRaises(ValueError):bind_text_prediction(model,settings|{'minilm_weights':root/'changed'},root/'cache')

    def test_calibration_uses_retained_probabilities_with_arbitrary_binary_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[f'neutral text {i}' for i in range(10)]
            labels=np.asarray(['a','b']*5)
            task=TrainingTask('single_text',rows,labels)
            hist=[dict(candidate=n,trial=i) for i,n in enumerate((TEXT_MINILM,TEXT_DEEP),6)]
            conf=dict(valid_confirmation=True,selected_candidate=TEXT_MINILM,shortlist=[TEXT_MINILM,TEXT_DEEP],primary_metric='accuracy')
            for i in range(3):
                split='split.json' if i==0 else f'confirmation_split_{i}.json'
                (root/split).write_text(json.dumps({'search_indices':list(range(10))}))
                for j,p in ((6,.8),(7,.2)):
                    out=root/(f'trial_{j}.joblib' if i==0 else f'confirmation_{i}_{j}.joblib')
                    classes=np.unique(labels)
                    proba=np.tile([1-p,p],(10,1))
                    joblib.dump(dict(search_scores=proba,score_classes=classes,search_predictions=classes[proba.argmax(axis=1)]),out)
            decision=learn_from_confirmations(task,hist,conf,root)
            self.assertAlmostEqual(decision['raw'],.5)
            self.assertEqual(decision['visits'],30)
            self.assertEqual(decision['unique_rows'],10)
            self.assertFalse(learn_from_confirmations(task,hist,conf|{'shortlist':[TEXT_MINILM]},root)['eligible'])
