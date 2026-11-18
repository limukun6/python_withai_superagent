from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from superagent.feature_cache import using_feature_cache,FeatureCacheError
from superagent.models import _difference_features,search_candidates,predict_with_model


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cache=self.root/'cache'
        self.pairs=[(f'Longer example number {i} with question?',f'Short report {i%4} news.') for i in range(50)]
        self.labels=np.asarray([i%2 for i in range(50)])

    def vector(self,**kwargs):return TfidfVectorizer(min_df=1,**kwargs)
    def train(self,vector=None,pairs=None):
        vector=vector if vector is not None else self.vector()
        return vector,_difference_features(vector,self.pairs if pairs is None else pairs,True)

    def test_fresh_worker_context_hits_disk_and_matrix_isolation_preserves_results(self):
        with using_feature_cache(self.cache) as first:
            vector,expected=self.train();self.assertEqual(first.report()['builds'],1)
            again=_difference_features(vector,self.pairs,False)
            np.testing.assert_array_equal(again.toarray(),expected.toarray())
            again.data[:]=123
            np.testing.assert_array_equal(_difference_features(vector,self.pairs,False).toarray(),expected.toarray())
        with using_feature_cache(self.cache) as second,patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('Cache hit must not fit')):
            restored,matrix=self.train()
            self.assertEqual(second.report()['disk_hits'],1)
            np.testing.assert_array_equal(matrix.toarray(),expected.toarray())
            self.assertEqual(restored.vocabulary_,vector.vocabulary_)

    def test_input_order_values_parameters_and_source_changes_invalidate_key(self):
        with using_feature_cache(self.cache) as cache:
            self.train();self.train(pairs=list(reversed(self.pairs)))
            changed=list(self.pairs);changed[0]=('Changed training input','Unique replacement')
            self.train(pairs=changed);self.train(self.vector(ngram_range=(1,2)))
            cache.source_sha256='f'*64;self.train()
            self.assertEqual(cache.report()['builds'],5)
            self.assertEqual(len({e['key'] for e in cache.events}),5)

    def test_validation_transform_never_fits_or_changes_cached_vocabulary(self):
        with using_feature_cache(self.cache) as cache:
            vector,_=self.train();before=dict(vector.vocabulary_)
            with patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('Evaluation must not fit')):
                _difference_features(vector,[('ZZNEVERTRAINEDTOKEN','news')],False)
            self.assertEqual(vector.vocabulary_,before);self.assertNotIn('zznevertrainedtoken',before)
            self.assertEqual(cache.report()['builds'],1)

    def test_corrupt_artifact_is_rejected_before_deserialization(self):
        with using_feature_cache(self.cache):self.train()
        entry=next(p for p in self.cache.iterdir() if not p.name.startswith('.'))
        artifact=entry/'features.joblib';content=bytearray(artifact.read_bytes());content[-1]^=1;artifact.write_bytes(content)
        with using_feature_cache(self.cache),patch('superagent.feature_cache.joblib.load') as load:
            with self.assertRaises(FeatureCacheError):self.train()
            load.assert_not_called()

    def test_interrupted_unpublished_directory_is_not_a_hit(self):
        self.cache.mkdir();orphan=self.cache/'.partial-publish';orphan.mkdir();(orphan/'features.joblib').write_bytes(b'incomplete')
        with using_feature_cache(self.cache) as cache:
            self.train();self.assertEqual(cache.report()['builds'],1);self.assertEqual(cache.report()['disk_hits'],0)
        self.assertTrue(orphan.exists())

    def test_classifiers_are_refitted_for_changed_parameters_and_labels(self):
        catalog={c.family:c for c in search_candidates('pair_text','expanded')}
        original=LogisticRegression.fit;seen=[]
        def fitted(model,x,y,*args,**kwargs):
            seen.append({'C':model.C,'y':np.asarray(y).copy()});return original(model,x,y,*args,**kwargs)
        with patch.object(LogisticRegression,'fit',new=fitted):
            for name,y in [('word_C0.3',self.labels),('word_C0.7',self.labels),('word_C0.3',1-self.labels)]:
                with using_feature_cache(self.cache) as cache:catalog[name].fit_predict(self.pairs,y,self.pairs[:4])
            self.assertEqual(cache.report()['disk_hits'],1)
        self.assertEqual([r['C'] for r in seen],[.3,.7,.3]);np.testing.assert_array_equal(seen[2]['y'],1-seen[0]['y'])

    def test_model_is_self_contained_and_cached_predictions_match_uncached(self):
        candidate=next(c for c in search_candidates('pair_text','expanded') if c.family=='word_char_stats_C0.3')
        ordinary,expected=candidate.fit_predict(self.pairs,self.labels,self.pairs[:8])
        with using_feature_cache(self.cache):model,pred=candidate.fit_predict(self.pairs,self.labels,self.pairs[:8])
        np.testing.assert_array_equal(pred,expected)
        path=self.root/'model.joblib';joblib.dump(model,path);shutil.rmtree(self.cache)
        restored=joblib.load(path)
        np.testing.assert_array_equal(predict_with_model('pair_text',restored,self.pairs[:8]),expected)
        np.testing.assert_allclose(restored.classifier.coef_,ordinary.classifier.coef_,rtol=0,atol=0)

    def test_concurrent_publishers_leave_one_valid_complete_entry(self):
        def work(_):
            with using_feature_cache(self.cache):return self.train()[1].toarray()
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(work,range(2)))
        np.testing.assert_array_equal(results[0],results[1])
        entries=[p for p in self.cache.iterdir() if not p.name.startswith('.')];self.assertEqual(len(entries),1)
        self.assertTrue(json.loads((entries[0]/'metadata.json').read_text())['artifact_sha256'])
        with using_feature_cache(self.cache) as cache:
            self.train();self.assertEqual(cache.report()['disk_hits'],1)


if __name__=='__main__':unittest.main()
