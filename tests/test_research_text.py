import copy
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from superagent.evidence_agent import route_of,state_for
from superagent.models import search_candidates,predict_with_model,probability_with_model
from superagent.supervised_text import fit_nb_word,fit_style_svm


def fixture():
    pairs=[];labels=[]
    for i in range(30):
        preferred=f'Bright helpful report with clear details item {i}'
        other=f'Dull vague report with missing details item {i}'
        pairs.append((preferred,other) if i%2 else (other,preferred));labels.append(i%2)
    return pairs,np.asarray(labels)


class ResearchTextTests(unittest.TestCase):
    def test_new_catalog_is_optional_and_routes_are_distinct(self):
        base=search_candidates('pair_text','expanded');research=search_candidates('pair_text','research')
        self.assertEqual(len(base),22);self.assertEqual(len(research),28)
        self.assertEqual([(c.family,c.params) for c in base],[(c.family,c.params) for c in research[:22]])
        self.assertEqual({route_of(c.family) for c in research[22:]},{'nb_word','word_char_stats_svm'})
        state=state_for({},[],[{'id':c.family,'params':c.params} for c in research],5,'budget_routes')
        self.assertTrue(any(r['route']=='nb_word' for r in state['available_candidates']))
        self.assertIn('supervised_feature_weighting',state['parameter_definitions'])
        self.assertEqual([c.family for c in search_candidates('categorical_tabular','research')],
                         [c.family for c in search_candidates('categorical_tabular','expanded')])

    def test_nb_weighting_uses_train_only_and_never_refits_during_inference(self):
        pairs,y=fixture();probe=[('neverobservedword novel content','anotherunknownword token')]
        model,pred=fit_nb_word(pairs,y,probe)
        self.assertNotIn('neverobservedword',model.vectorizer.vocabulary_)
        vocab=copy.deepcopy(model.vectorizer.vocabulary_);weights=model.log_count_ratio.copy()
        predict_with_model('pair_text',model,probe*3)
        self.assertEqual(vocab,model.vectorizer.vocabulary_);np.testing.assert_array_equal(weights,model.log_count_ratio)
        self.assertTrue(np.isfinite(weights).all());self.assertEqual(pred.shape,(1,))
        bright=model.vectorizer.vocabulary_['bright'];dull=model.vectorizer.vocabulary_['dull']
        self.assertGreater(weights[bright],0);self.assertLess(weights[dull],0)

    def test_nb_label_swap_and_pair_swap_keep_preference_semantics(self):
        pairs,y=fixture();swapped=[(b,a) for a,b in pairs]
        model,_=fit_nb_word(pairs,y,pairs)
        np.testing.assert_allclose(model.decision_function(swapped),-model.decision_function(pairs),atol=1e-12)
        np.testing.assert_allclose(model.predict_proba(pairs)[:,1]+model.predict_proba(swapped)[:,1],1.)
        alternative,_=fit_nb_word(swapped,1-y,pairs)
        np.testing.assert_allclose(alternative.log_count_ratio,model.log_count_ratio)
        np.testing.assert_allclose(alternative.predict_proba(pairs),model.predict_proba(pairs))

    def test_margin_model_is_symmetric_and_keeps_train_vocab_and_scaler(self):
        pairs,y=fixture();probe=[('neverobservedword 123!','anotherunknownword?')]
        model,_=fit_style_svm(pairs,y,probe)
        self.assertNotIn('neverobservedword',model.word.vocabulary_)
        scales=model.scaler.scale_.copy();vocab=copy.deepcopy(model.word.vocabulary_)
        forward=model.classifier.decision_function(model.transform(pairs))
        reverse=model.classifier.decision_function(model.transform([(b,a) for a,b in pairs]))
        np.testing.assert_allclose(forward,-reverse,atol=1e-12)
        predict_with_model('pair_text',model,probe)
        np.testing.assert_array_equal(scales,model.scaler.scale_);self.assertEqual(vocab,model.word.vocabulary_)
        self.assertFalse(hasattr(model.classifier,'predict_proba'))

    def test_saved_new_models_replay_and_reject_nonbinary_labels(self):
        pairs,y=fixture()
        with tempfile.TemporaryDirectory() as temp:
            for i,fit in enumerate((fit_nb_word,fit_style_svm)):
                model,pred=fit(pairs,y,pairs)
                path=Path(temp)/f'model{i}.joblib';joblib.dump(model,path)
                np.testing.assert_array_equal(pred,predict_with_model('pair_text',joblib.load(path),pairs))
                with self.assertRaises(ValueError):fit(pairs,np.arange(len(y))%3,pairs)
            labels,probs=probability_with_model('pair_text',fit_nb_word(pairs,y,pairs)[0],pairs)
            self.assertEqual(labels.tolist(),[0,1]);np.testing.assert_allclose(probs.sum(axis=1),1.)


if __name__=='__main__':unittest.main()
