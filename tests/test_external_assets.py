import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.linear_model import LogisticRegression
from superagent.ab_sentence import SentenceFeatures,KernelSentenceFeatures
from superagent.ab_symmetric import UnorderedSentenceFeatures
from superagent.ab_vision import FrozenVisionFeatures
from superagent.external_assets import model_resources,verify_resources


class ExternalAssetTests(unittest.TestCase):
    def test_nested_actual_objects_determine_resources_not_candidate_name(self):
        for feature in (SentenceFeatures('single_text'),KernelSentenceFeatures('pair_text'),UnorderedSentenceFeatures()):
            model=Pipeline([('different_name',FeatureUnion([('nested',feature)])),('head',LogisticRegression())])
            resources,packages=model_resources(model)
            self.assertEqual([r['name'] for r in resources],['minilm'])
            self.assertIn('transformers',packages)
            self.assertNotIn('torchvision',packages)
        resources,packages=model_resources(Pipeline([('feature',FrozenVisionFeatures())]))
        self.assertEqual([r['name'] for r in resources],['vision'])
        self.assertEqual(packages,['torch','torchvision'])
        self.assertEqual(model_resources(Pipeline([('head',LogisticRegression())])),([],[]))

    def test_missing_equal_size_corruption_and_valid_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights';path.write_bytes(b'good')
            descriptor=[dict(name='controlled',environment_variable='ASSET_TEST',path_kind='file',
                             file=dict(bytes=4,sha256=hashlib.sha256(b'good').hexdigest()))]
            with self.assertRaisesRegex(ValueError,'Missing frozen'):verify_resources(descriptor,{})
            verify_resources(descriptor,{'ASSET_TEST':str(path)})
            path.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_resources(descriptor,{'ASSET_TEST':str(path)})
            path.write_bytes(b'large')
            with self.assertRaisesRegex(ValueError,'size'):verify_resources(descriptor,{'ASSET_TEST':str(path)})

    def test_isolated_entry_rejects_missing_and_corrupted_assets_before_output(self):
        from tools.export_model_bundle import ENTRY
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);src=root/'src/superagent';src.mkdir(parents=True)
            (src/'__init__.py').write_text('',encoding='utf-8')
            (src/'external_assets.py').write_bytes((Path(__file__).resolve().parents[1]/'superagent/external_assets.py').read_bytes())
            (src/'predict.py').write_text("from pathlib import Path\ndef create_predictions(features,run,output):\n Path(output).write_text('controlled-gate-only')\n",encoding='utf-8')
            (root/'predict.py').write_text(ENTRY,encoding='utf-8')
            weights=root/'weights';weights.write_bytes(b'good')
            descriptor=[dict(name='controlled',environment_variable='ASSET_TEST',path_kind='file',
                             file=dict(bytes=4,sha256=hashlib.sha256(b'good').hexdigest()))]
            manifest=dict(dependencies={},external_resources=descriptor,files={})
            (root/'bundle_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            for mode in ('good','missing','corrupt'):
                output=root/(mode+'.csv');env={**os.environ,'ASSET_TEST':str(weights) if mode!='missing' else ''}
                if mode=='corrupt':weights.write_bytes(b'evil')
                result=subprocess.run([sys.executable,'-I',str(root/'predict.py'),'--features','unused.zip','--output',str(output)],
                                      cwd=root,env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode==0,mode=='good')
                self.assertEqual(output.exists(),mode=='good')
