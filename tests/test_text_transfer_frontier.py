import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from superagent.method_choice import COOCCUR, MINILM, NEURAL, TEXT_MINILM, TEXT_MINILM_ONLY, resolved_mode, select_action
from superagent.minilm_features import MiniLMTextClassifier, feature_usage
from superagent.models import predict_with_model, probability_with_model


class StaticEncoder:
    def __init__(self, runtime):
        self.runtime = runtime

    def encode(self, texts):
        return np.asarray([[len(t), t.count("o"), t.count("b")] for t in texts], dtype=float)


class FrozenTextTests(unittest.TestCase):
    def test_unified_policy_uses_available_capabilities_and_keeps_previous_categorical_decisions(self):
        original={MINILM,NEURAL,COOCCUR}
        text={TEXT_MINILM,TEXT_MINILM_ONLY}
        ev=dict(kind="categorical_tabular",semantic="other",shared_errors=32,required_shared_errors=8,
                low_support_shared_errors=15,short_latin_share=.95,rare_enrichment=1.2,
                primary_metric="accuracy",corrected=3,lost=19,discordant=22,search_rows=300,
                nonlinear_gain=.04,previous_delta=-.05,margin=.01)
        self.assertEqual(resolved_mode("auto",original),"paired")
        for used in ([],[MINILM]):
            self.assertEqual(select_action(ev,original,"auto",used),select_action(ev,original,"paired",used))
        self.assertEqual(resolved_mode("auto",text),"text-encoder")
        for used in ([],[TEXT_MINILM]):
            self.assertEqual(select_action(ev|{"kind":"pair_text"},text,"auto",used),
                             select_action(ev|{"kind":"pair_text"},text,"text-encoder",used))

    def test_agent_can_decline_or_run_the_second_probe(self):
        evidence=dict(kind="pair_text",semantic="model",shared_errors=31,required_shared_errors=10,
                      short_latin_share=0.,low_support_shared_errors=0,primary_metric="accuracy",corrected=20,lost=5)
        feasible={TEXT_MINILM,TEXT_MINILM_ONLY}
        self.assertEqual(select_action(evidence,feasible,"text-encoder",[])[0],TEXT_MINILM)
        self.assertEqual(select_action(evidence,feasible,"text-encoder",[TEXT_MINILM])[0],TEXT_MINILM_ONLY)
        self.assertIsNone(select_action(evidence | {"corrected":5},feasible,"text-encoder",[TEXT_MINILM])[0])
        self.assertIsNone(select_action(evidence | {"shared_errors":1},feasible,"text-encoder",[])[0])
        self.assertEqual(select_action(evidence | {"shared_errors":1},feasible,"text-encoder-control",[])[0],TEXT_MINILM)

    def test_pair_class_order_serialization_and_single_multiclass(self):
        pairs=[("red boat","blue ocean"),("blue ocean","red boat")]*10
        y=np.array([1,0]*10)
        with tempfile.TemporaryDirectory() as tmp, patch.object(MiniLMTextClassifier,"encoder_class",StaticEncoder):
            root=Path(tmp)
            for dense in (False,True):
                model=MiniLMTextClassifier("pair_text",root/"train.json",dense).fit(pairs,y)
                file=root/"model.joblib"
                joblib.dump(model,file)
                restored=joblib.load(file)
                self.assertIsNone(restored.encoder)
                restored.bind_prediction_weights(root/"weights",root/"inference")
                labels,scores=probability_with_model("pair_text",restored,pairs[:2])
                np.testing.assert_array_equal(labels,[0,1])
                np.testing.assert_allclose(scores[0],scores[1][::-1],atol=1e-6)
                np.testing.assert_array_equal(predict_with_model("pair_text",restored,pairs),y)
            single=MiniLMTextClassifier("single_text",root/"train.json",False).fit(
                ["blue ocean","orange boat","red rock"]*8,np.array(["x","y","z"]*8))
            self.assertEqual(single.predict(["red rock"])[0],"z")
            self.assertEqual(feature_usage(root/"inference")["remote_calls"],0)
