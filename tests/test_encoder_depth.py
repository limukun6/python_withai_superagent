import unittest

from superagent.method_choice import TEXT_DEEP,TEXT_MINILM,select_action
from superagent.minilm_features import DEEP_FILES,FILES,DeeperMiniLMEncoder,MiniLMEncoder


class EncoderDepthTests(unittest.TestCase):
    def test_weights_have_independent_caches_and_verified_networks(self):
        self.assertNotEqual(DeeperMiniLMEncoder.cache_name,MiniLMEncoder.cache_name)
        self.assertNotEqual(DeeperMiniLMEncoder.weight_key,MiniLMEncoder.weight_key)
        self.assertNotEqual(DEEP_FILES['model.safetensors'][1],FILES['model.safetensors'][1])
        self.assertEqual(DEEP_FILES['tokenizer.json'][1],FILES['tokenizer.json'][1])
        self.assertEqual(DeeperMiniLMEncoder.max_length,128)

    def test_gain_residual_and_feasibility_control_depth(self):
        ev=dict(kind='pair_text',semantic='s',shared_errors=45,required_shared_errors=12,
                short_latin_share=0.,low_support_shared_errors=0,primary_metric='accuracy',
                corrected=28,lost=9,residual_shared_errors=40,search_rows=430)
        feasible={TEXT_DEEP,TEXT_MINILM}
        self.assertEqual(select_action(ev,feasible,'encoder-depth',[])[0],TEXT_MINILM)
        self.assertEqual(select_action(ev,feasible,'encoder-depth',[TEXT_MINILM])[0],TEXT_DEEP)
        self.assertIsNone(select_action(ev|{'corrected':8},feasible,'encoder-depth',[TEXT_MINILM])[0])
        self.assertIsNone(select_action(ev,{TEXT_MINILM},'encoder-depth',[TEXT_MINILM])[0])
        self.assertIsNone(select_action(ev|{'residual_shared_errors':3},feasible,'encoder-depth',[TEXT_MINILM])[0])
