import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.linear_model import LogisticRegression
from superagent.ab_sentence import SentenceFeatures,KernelSentenceFeatures
from superagent.ab_symmetric import UnorderedSentenceFeatures
from superagent.ab_vision import FrozenVisionFeatures
from superagent.external_assets import model_resources,verify_resources
from superagent.ab_sentence import AlternativeSentenceFeatures
from superagent.confusion_specialist import ClassPairModel


class ExternalAssetTests(unittest.TestCase):
    def test_shared_directory_exposes_verified_installs_and_preserves_explicit_override(self):
        from superagent.external_assets import activate_resource_directory
        digest=hashlib.sha256(b"good").hexdigest()
        spec=dict(env="TEST_OPTIONAL_ENCODER",model="publisher/encoder",files={"weights":(4,digest)})
        with tempfile.TemporaryDirectory() as tmp, patch("superagent.ab_sentence.ALTERNATIVES",{"generic":spec}):
            root=Path(tmp)
            broken=root/"publisher"/"encoder"
            valid=root/"encoder"
            broken.mkdir(parents=True)
            valid.mkdir()
            (broken/"weights").write_bytes(b"oops")
            env={}
            activate_resource_directory(root,env)
            self.assertNotIn(spec["env"],env)
            (valid/"weights").write_bytes(b"good")
            activate_resource_directory(root,env)
            self.assertEqual(env[spec["env"]],str(valid.resolve()))
            env[spec["env"]]="individually configured"
            activate_resource_directory(root,env)
            self.assertEqual(env[spec["env"]],"individually configured")

    def test_refinement_carries_resources_of_both_fitted_models(self):
        base = Pipeline([('text', SentenceFeatures('single_text'))])
        expert = Pipeline([('features', FeatureUnion([
            ('first', SentenceFeatures('single_text')),
            ('second', AlternativeSentenceFeatures('single_text', 'e5')),
        ]))])
        resources, packages = model_resources(ClassPairModel('single_text', base, expert, [0, 1, 2], (0, 1)))
        self.assertEqual([r['name'] for r in resources], ['minilm', 'e5'])
        self.assertIn('transformers', packages)

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
