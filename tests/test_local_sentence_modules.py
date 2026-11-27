import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from superagent.external_assets import model_resources, verify_resources
from superagent.local_sentence import LocalSentenceFeatures, candidates, describe, discover
from superagent.task_constraints import permitted


def package(root):
    (root/'1_Pooling').mkdir(parents=True)
    modules = [{'idx':0,'type':'sentence_transformers.models.Transformer','path':''},
               {'idx':1,'type':'sentence_transformers.models.Pooling','path':'1_Pooling'}]
    for name, data in {'modules.json':modules,'1_Pooling/config.json':{'word_embedding_dimension':16,'pooling_mode_mean_tokens':True},
                       'config.json':{'model_type':'bert','hidden_size':16,'max_position_embeddings':32},
                       'sentence_bert_config.json':{'max_seq_length':24},'tokenizer.json':{}}.items():
        (root/name).write_text(json.dumps(data))
    (root/'model.safetensors').write_bytes(b'resource manifest fixture')
    return modules


class LocalSentenceModules(TestCase):
    def test_rename_export_restrictions_and_content_pin(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            base=Path(tmp)
            folder=base/'arbitrary'
            package(folder)
            initial=describe(folder)
            other=base/'renamed'
            folder.rename(other)
            self.assertEqual(describe(other),initial)
            discover(base,os.environ)
            choices=candidates('single_text')
            self.assertEqual(len(choices),1)
            cap=dict(params=choices[0].params,resource='local_frozen_encoder_cpu',origin='b_port')
            for restriction in ('no_external_data','no_pretrained_assets','linear_only'):
                self.assertFalse(permitted(cap,[restriction]))
            resources,_=model_resources([LocalSentenceFeatures('single_text',initial)])
            self.assertEqual([r['environment_variable'] for r in resources],[initial['environment_variable']])
            verify_resources(resources)
            (other/'model.safetensors').write_bytes(b'changed bytes')
            with self.assertRaises(ValueError):
                verify_resources(resources)

    def test_incomplete_unknown_or_mixed_operations_do_not_enter_inventory(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp)
            folder=base/'unlabelled'
            modules=package(folder)
            (folder/'modules.json').unlink()
            env={}
            discover(base,env)
            self.assertFalse(env)
            modules.append({'idx':2,'path':'2_Dense','type':'custom.Unrecognized'})
            (folder/'modules.json').write_text(json.dumps(modules))
            discover(base,env)
            self.assertFalse(env)
            (folder/'modules.json').write_text(json.dumps(modules[:2]))
            (folder/'1_Pooling/config.json').write_text(json.dumps(dict(word_embedding_dimension=16,
                pooling_mode_cls_token=True,pooling_mode_mean_tokens=True)))
            discover(base,env)
            self.assertFalse(env)
