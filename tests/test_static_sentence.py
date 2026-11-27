import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

from superagent.local_sentence import describe, discover, encode


class StaticSentenceTest(TestCase):
    def test_embedding_bag_respects_declared_normalization_and_empty_inputs(self):
        with TemporaryDirectory() as temp, patch.dict(os.environ,{},clear=True):
            root=Path(temp)
            table=np.array([[0,0],[3,0],[0,3]],dtype=np.float32)
            save_file({"embeddings":table},str(root/"model.safetensors"))
            tok=Tokenizer(models.WordLevel({"[UNK]":0,"red":1,"blue":2},unk_token="[UNK]"))
            tok.pre_tokenizer=pre_tokenizers.Whitespace()
            tok.save(str(root/"tokenizer.json"))
            defs=[dict(idx=0,type="sentence_transformers.models.StaticEmbedding",path="."),
                  dict(idx=1,type="sentence_transformers.models.Normalize",path="1_Normalize")]
            previous=None
            for normalized in (False,True):
                (root/"modules.json").write_text(json.dumps(defs[:1+int(normalized)]))
                descriptor=describe(root)
                discover(root,os.environ)
                result=encode(descriptor,["red blue blue","","unknown"])
                expected=np.array([1.,2.])
                if normalized:
                    expected/=np.linalg.norm(expected)
                np.testing.assert_allclose(result,[expected,[0,0],[0,0]],atol=1e-6)
                np.testing.assert_equal(encode(descriptor,[]).shape,(0,2))
                if previous:
                    self.assertNotEqual(descriptor["signature"],previous["signature"])
                    with self.assertRaises(ValueError):
                        encode(previous,["red"])
                previous=descriptor
            save_file({"embeddings":table,"mapping":np.array([0,1])},str(root/"model.safetensors"))
            with self.assertRaises(ValueError):
                describe(root)
