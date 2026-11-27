from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import json

import numpy as np

from superagent import local_sentence
from tests.test_local_sentence_modules import package


class LocalEmbeddingDevice(TestCase):
    def test_last_pool_requires_a_declared_pool_and_clocked_length(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp)/'anonymous'
            package(folder)
            model={'model_type':'qwen3','hidden_size':16,'max_position_embeddings':32768}
            (folder/'config.json').write_text(json.dumps(model))
            (folder/'sentence_bert_config.json').unlink()
            (folder/'tokenizer_config.json').write_text(json.dumps({'model_max_length':131072}))
            (folder/'1_Pooling/config.json').write_text(json.dumps({'word_embedding_dimension':16,'pooling_mode_lasttoken':True}))
            s=local_sentence.describe(folder)
            self.assertEqual((s['pooling'],s['publisher_max_length'],s['max_length']),('last',32768,1024))
            (folder/'1_Pooling/config.json').write_text(json.dumps({'word_embedding_dimension':16,'pooling_mode_mean_tokens':True}))
            with self.assertRaises(ValueError):
                local_sentence.describe(folder)

    def test_a_fast_inaccurate_backend_cannot_change_the_embedding(self):
        import torch
        base=np.array([[1.,0.]]*12)
        def trial(spec,samples,device):
            values=base if device=='cpu' else -base
            return values[:len(samples)]
        with patch.dict(local_sentence.BACKEND_CHOICES,{},clear=True),patch.object(local_sentence,'_encode',side_effect=trial), \
             patch.object(torch.cuda,'is_available',return_value=False),patch.object(torch.backends.mps,'is_available',return_value=True), \
             patch.object(local_sentence.time,'perf_counter',side_effect=[0.,10.,15.,15.5]):
            choice=local_sentence.choose_backend({'signature':'test','pooling':'last'},[str(i) for i in range(12)])
        self.assertEqual(choice,'cpu')

    def test_padding_side_and_last_real_positions(self):
        import torch
        with TemporaryDirectory() as temp:
            folder=Path(temp)/'opaque'
            package(folder)
            config=folder/'tokenizer_config.json'
            config.write_text(json.dumps({'padding_side':'left'}))
            self.assertEqual(local_sentence.describe(folder)['padding_side'],'left')
            config.write_text(json.dumps({'padding_side':'right'}))
            self.assertNotIn('padding_side',local_sentence.describe(folder))
            config.write_text(json.dumps({'padding_side':'invalid'}))
            with self.assertRaisesRegex(ValueError,'padding side'):
                local_sentence.describe(folder)
        hidden=torch.arange(12).reshape(3,4,1)
        mask=torch.tensor([[1,1,0,0],[0,0,1,1],[1,0,1,0]])
        self.assertEqual(local_sentence.last_real_token(hidden,mask).flatten().tolist(),[1,7,10])
