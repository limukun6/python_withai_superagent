import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import joblib
import numpy as np
from superagent.method_choice import select_action, weights_available, MINILM, SEMANTIC, NEURAL, COOCCUR
from superagent.models import predict_with_model, probability_with_model
from superagent.schema import MixedRows, TextTableRows


def library_available(name):
    import importlib.util
    return importlib.util.find_spec(name) is not None


class MethodRoutingTests(unittest.TestCase):
    def test_same_catalog_responds_to_observed_results(self):
        evidence=dict(kind='categorical_tabular',semantic=SEMANTIC,shared_errors=40,required_shared_errors=10,
                      short_latin_share=1.,nonlinear_gain=.06,low_support_shared_errors=28,previous_delta=0.,margin=.01)
        feasible={MINILM,SEMANTIC,NEURAL,COOCCUR}
        def action(used=(),mode='adaptive',**changes):
            return select_action(evidence|changes,feasible,mode,used)[0]
        self.assertEqual(action(),NEURAL)
        self.assertEqual(action(nonlinear_gain=0.),MINILM)
        self.assertEqual(action(nonlinear_gain=0.,short_latin_share=.1),SEMANTIC)
        self.assertIsNone(action(shared_errors=3))
        self.assertEqual(action([NEURAL],previous_delta=-.02),MINILM)
        self.assertIsNone(action([NEURAL],previous_delta=.02))
        self.assertEqual(action([MINILM],previous_delta=-.02),NEURAL)
        self.assertEqual(action(mode='fixed',shared_errors=3),MINILM)
        self.assertEqual(action([MINILM],mode='fixed',nonlinear_gain=1.),SEMANTIC)
        self.assertNotEqual(select_action(evidence,{MINILM},'adaptive',[])[0],NEURAL)

    def test_offline_missing_weights_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ',{'SUPERAGENT_OFFLINE':'1'}):
            key='minilm_weights'; files={'config.json':('rev','sha',2)}
            self.assertFalse(weights_available({key:tmp},key,files))
            (Path(tmp)/'config.json').write_text('{}')
            self.assertTrue(weights_available({key:tmp},key,files))

    @unittest.skipUnless(library_available('torch'),'Optional PyTorch')
    def test_scratch_classifiers_reload_across_schemas_with_original_classes(self):
        from superagent.scratch_classifier import ScratchClassifier
        n=30
        rng=np.random.default_rng(7)
        y=np.tile(['East','North','West'],10)
        category=np.array([['alpha','one'],['beta','two'],['gamma','three']]*10)
        values=np.column_stack([np.arange(n),category[:,0]]).astype(object)
        samples=dict(categorical_tabular=category,numeric_tabular=rng.normal(size=(n,5)),
                     mixed_tabular=MixedRows(values,(0,),('number','feature')),
                     text_tabular=TextTableRows(values,(0,),('number','sentence'),(1,)),
                     single_text=['some '+s for s in y],pair_text=[('left '+s,'right '+s) for s in y],
                     image=rng.integers(0,255,(n,8,9,3),dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp, patch('superagent.scratch_classifier.device_probe',return_value=('cpu',{'selected':'cpu'})):
            for kind,rows in samples.items():
                with self.subTest(kind=kind):
                    m=ScratchClassifier(kind,dict(seed=41,hidden=8,batch_size=8,epochs=2)).fit(rows,y)
                    file=Path(tmp)/'fitted';joblib.dump(m,file)
                    loaded=joblib.load(file)
                    test=rows[:3] if kind!='categorical_tabular' else np.array([['unseen','three']]*3)
                    np.testing.assert_array_equal(m.predict(test),predict_with_model(kind,loaded,test))
                    classes,probs=probability_with_model(kind,loaded,test)
                    self.assertEqual(probs.shape,(3,3))
                    np.testing.assert_array_equal(classes,sorted(set(y)))
                    np.testing.assert_allclose(probs.sum(1),1.,atol=1e-6)

    @unittest.skipUnless(library_available('torch') and library_available('tokenizers'),'Optional encoder dependencies')
    def test_minilm_is_masked_cached_and_separate_from_generation(self):
        import torch
        from tokenizers import Tokenizer,models,pre_tokenizers,processors
        from superagent.minilm_features import MiniLMEncoder
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); path=folder/'runtime.json'
            path.write_text(json.dumps({'minilm_weights':tmp}))
            enc=MiniLMEncoder(path)
            words=['[PAD]','[UNK]','[CLS]','[SEP]','a','b','c']
            tok=Tokenizer(models.WordLevel(dict(zip(words,range(7))),unk_token='[UNK]'))
            tok.pre_tokenizer=pre_tokenizers.Whitespace()
            tok.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
            enc.tokenizer=tok
            enc.model=Mock(side_effect=lambda input_ids,attention_mask:Mock(last_hidden_state=torch.nn.functional.one_hot(input_ids,384).float()))
            with patch('superagent.pretrained_choice.budget',side_effect=AssertionError('language-model ledger used for MiniLM')):
                v=enc.encode(['a','a b c','a'])
                np.testing.assert_allclose(v[0][[2,3,4]],np.ones(3)/np.sqrt(3),atol=1e-6)
                self.assertEqual(float(v[0][0]),0.)
                with patch.object(MiniLMEncoder,'load',side_effect=AssertionError('unexpected download')):
                    np.testing.assert_array_equal(MiniLMEncoder(path).encode(['a','a b c','a']),v)
            usage=json.loads((folder/'downstream_usage.jsonl').read_text())
            self.assertEqual((usage['input_tokens'],usage['sequences'],usage['api_rmb']),(8,2,0.))
            self.assertFalse((folder/'llm_usage.jsonl').exists())


if __name__=='__main__':
    unittest.main()
