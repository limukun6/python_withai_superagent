import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.events import RunLogger
from superagent.format_sources import package_files
from superagent.recovery import RecoveryController
from superagent.semantic_contract import evidence_workspace,semantic_training,ContractProbeError
from superagent.structure_tools import inspect_structure
from tests.test_semantic_contract import Client,fixture,proposal


class RecoveryTests(unittest.TestCase):
    def test_two_distinct_failed_plans_can_recover_on_third_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=fixture(root);(root/'log').mkdir();log=RunLogger(root/'log')
            a=proposal();a['plan']['target']='absent'
            b=proposal();b['plan']['columns'][0]['name']='absent'
            client=Client([a,b,proposal()]);controller=RecoveryController(log,client=client,fit_limit=4)
            task=semantic_training(path,client,log,ValueError('native failure'),recovery=controller)
            self.assertEqual(len(task.y_train),8);self.assertEqual(client.request_count,3)
            self.assertEqual(len(controller.context()['failed_actions']),3)
            self.assertEqual(controller.context()['stage'],'training_method')
            self.assertEqual(controller.context()['remaining_requests'],0)

    def test_repeated_failure_is_reported_to_remaining_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture(Path(tmp));a=proposal();a['plan']['target']='absent'
            client=Client([a,a,proposal()]);semantic_training(path,client,None,ValueError('native'))
            self.assertIn('Identical action',client.payloads[2]['diagnostic']['recovery_hint'])
            self.assertTrue(client.payloads[2]['recovery_state']['failed_actions'][-1]['repeated_without_new_evidence'])

    def test_state_updates_fit_budget_and_keeps_existing_artifact(self):
        c=RecoveryController(fit_limit=4)
        c.advance('training_method',accounting={'attempted_classifier_fits':2},artifacts=['checkpoint'])
        c.failure('training_method',{'code':'RuntimeError','message':'Unavailable'},action={'candidate':'a'})
        self.assertEqual(c.context()['remaining_fits'],2)
        self.assertEqual(c.context()['available_artifacts'],['checkpoint'])


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'task.zip'
        data=io.BytesIO();np.savez(data,opaque_matrix=np.zeros((8,3)),opaque_target=np.arange(8),answers=np.arange(8))
        from PIL import Image
        image=io.BytesIO();Image.new('RGB',(7,9)).save(image,format='PNG')
        obj=io.BytesIO();np.save(obj,np.array([object()],dtype=object))
        with zipfile.ZipFile(self.path,'w') as z:
            z.writestr('README.md','Inspect train.npz, picture.png or object.npy.');z.writestr('train.npz',data.getvalue())
            z.writestr('object.npy',obj.getvalue());z.writestr('picture.png',image.getvalue())
            z.writestr('validation/secret.npz',b'not opened')

    def request(self,member='train.npz',source='training',key=None):
        return dict(action='inspect_array',member=member,key=key,source=source,evidence_refs=['doc:README.md:0'])

    def inspect(self,request):
        with package_files(self.path) as files:
            return inspect_structure(files,evidence_workspace(files),request)

    def test_npz_inventory_does_not_load_arrays_or_disclose_values(self):
        with patch.object(np,'load',side_effect=AssertionError('array decoded')):
            result=self.inspect(self.request())
        self.assertEqual(result['arrays'][0]['shape'],[8,3])
        self.assertFalse(result['values_disclosed'])
        self.assertNotIn('data',result['arrays'][0])

    def test_prediction_label_key_is_not_opened(self):
        result=self.inspect(self.request(source='prediction'))
        blocked=next(a for a in result['arrays'] if a['key']=='answers')
        self.assertEqual(blocked['status'],'not_opened_role_restricted')
        with self.assertRaises(ContractProbeError):self.inspect(self.request(source='prediction',key='answers'))

    def test_object_array_is_not_unpickled(self):
        with patch.object(np,'load',side_effect=AssertionError('pickle loaded')):
            with self.assertRaises(ContractProbeError):self.inspect(self.request('object.npy'))

    def test_evaluation_member_rejected_before_read(self):
        with package_files(self.path) as files:
            workspace=evidence_workspace(files)
            with patch.object(files,'read',side_effect=AssertionError('evaluation opened')):
                with self.assertRaises(ContractProbeError):inspect_structure(files,workspace,self.request('validation/secret.npz'))

    def test_image_metadata_without_pixel_load(self):
        from PIL import PngImagePlugin
        with patch.object(PngImagePlugin.PngImageFile,'load',side_effect=AssertionError('pixels decoded')):
            result=self.inspect(dict(action='inspect_image',member='picture.png',source='training',evidence_refs=['doc:README.md:0']))
        self.assertEqual((result['width'],result['height'],result['mode']),(7,9,'RGB'))

    def test_fabricated_citation_rejected(self):
        req=self.request();req['evidence_refs']=['invented']
        with self.assertRaises(ContractProbeError):self.inspect(req)


if __name__=='__main__':unittest.main()
