"""Cited rule normalization through actual readers, with no classifier fitting."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from superagent.events import RunLogger
from superagent.semantic_contract import resolve_training_task,UnresolvedTaskReport
from superagent.semantic_artifact import load_artifact,task_digest
from superagent.task_constraints import UnknownRestrictionWording,RestrictionError,documented_flags
from superagent.rule_resolution import ACTIVE,scope,lookup
from tests.test_semantic_contract import Client

METHOD='Allowed classifiers: affine class scores followed by a softmax'
METRIC='Primary metric is the fraction of correctly classified rows'


def fixture(root,doc,metric='balanced_accuracy',restrictions=None):
    manifest=dict(schema_version=1,modality='tabular',target='kind',splits=dict(
        train=dict(format='npy',path='X_train.npy',target_path='y_train.npy'),test=dict(format='npy',path='X_test.npy')))
    if metric:manifest['metric']=metric
    if restrictions is not None:manifest['restrictions']=restrictions
    values={'X_train.npy':np.arange(24).reshape(8,3),'y_train.npy':np.array([0,1]*4),'X_test.npy':np.ones((3,3))}
    path=root/'task.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('README.md',doc);z.writestr('manifest.json',json.dumps(manifest))
        for name,value in values.items():
            s=io.BytesIO();np.save(s,value,allow_pickle=False);z.writestr(name,s.getvalue())
    return path


def wrapped(quote=METHOD,field='method_restrictions',value=None):
    if value is None:value=['linear_only'] if field=='method_restrictions' else 'accuracy'
    inner=dict(version='semantic-output-v2',output=None,evidence_refs=[],uncertainties=[])
    return dict(version='semantic-rules-v1',interpretation=inner,rules=[dict(field=field,quote=quote,
        evidence_ref='doc:README.md:0',value=value,reason='Normalize the actual task wording to the supported executor rule.')])


class RuleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def resolve(self,path,response,log=None):return resolve_training_task(path,Client([response],limit=1),log,
        mode='auto',prediction_package=path)

    def test_unknown_method_wording_reaches_model_and_training_contract(self):
        path=fixture(self.root,METHOD+'. Do not use external data.')
        with self.assertRaises(UnknownRestrictionWording):resolve_training_task(path)
        client=Client([wrapped()],limit=1)
        task=resolve_training_task(path,client,mode='auto',prediction_package=path)
        self.assertEqual(task.spec['method_restrictions'],['linear_only','no_external_data'])
        self.assertEqual(task.spec['metric'],'balanced_accuracy');self.assertEqual(client.request_count,1)
        self.assertEqual(client.payloads[0]['diagnostic']['code'],'UnknownRestrictionWording')
        self.assertFalse(task.spec['rule_resolution']['semantic_entailment_certified'])

    def test_unknown_metric_in_explicit_manifest_context_can_be_resolved(self):
        path=fixture(self.root,METRIC+'.',metric=None)
        task=self.resolve(path,wrapped(METRIC,'metric'))
        self.assertEqual(task.spec['metric'],'accuracy')
        self.assertEqual(len(task.X_train),8)

    def test_no_client_and_exhausted_client_preserve_strict_behavior(self):
        path=fixture(self.root,METHOD+'.')
        for client in (None,Client([],limit=0)):
            with self.assertRaises(UnknownRestrictionWording):resolve_training_task(path,client,mode='fallback')

    def test_explicit_manifest_conflict_is_not_overridden(self):
        quote='Do not use pretrained encoders';path=fixture(self.root,quote+'.',restrictions={'no_pretrained_assets':False})
        with self.assertRaises(ValueError):self.resolve(path,wrapped(quote,value=['no_pretrained_assets']))
        with self.assertRaises(RestrictionError):resolve_training_task(path)

    def test_model_can_report_real_method_conflict_as_unresolved(self):
        path=fixture(self.root,'Do not use pretrained encoders.',restrictions={'no_pretrained_assets':False})
        report=dict(action='report_unresolved',reason='Manifest and documentation disagree.',questions=[dict(
            field='method_restrictions',question='Whether pretrained encoders are permitted is contradictory.',evidence_refs=['doc:README.md:0'])])
        with self.assertRaises(UnresolvedTaskReport):self.resolve(path,report)

    def test_empty_restrictions_do_not_authorize_ignoring_unknown_rule(self):
        path=fixture(self.root,METHOD+'.')
        with self.assertRaises(ValueError):self.resolve(path,wrapped(value=[]))

    def test_rules_cannot_target_input_or_labels(self):
        path=fixture(self.root,METHOD+'.')
        for field in ('target','columns','training','positive_label'):
            with self.subTest(field=field),self.assertRaises(ValueError):self.resolve(path,wrapped(field=field,value='different'))

    def test_fabricated_quote_or_document_is_refused(self):
        path=fixture(self.root,METHOD+'.')
        for key,value in [('quote','invented task wording'),('evidence_ref','doc:missing:0')]:
            answer=wrapped();answer['rules'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.resolve(path,answer)

    def test_recognized_declaration_cannot_be_reinterpreted(self):
        quote='Only linear classifiers are permitted';path=fixture(self.root,quote+'.')
        with self.assertRaises(ValueError):self.resolve(path,wrapped(quote,value=['no_external_data']))

    def test_normalization_does_not_allow_replacing_a_manifest(self):
        path=fixture(self.root,METHOD+'.');answer=wrapped();answer['interpretation']=dict(version='semantic-native-v3')
        with self.assertRaises(ValueError):self.resolve(path,answer)

    def test_context_resets_after_a_failed_compilation(self):
        path=fixture(self.root,METHOD+'.');answer=wrapped();answer['interpretation']['output']={'bad':'recipe'}
        with self.assertRaises(ValueError):self.resolve(path,answer)
        self.assertIsNone(ACTIVE.get())
        with self.assertRaises(UnknownRestrictionWording):documented_flags(METHOD)

    def test_scope_cannot_remove_a_known_restriction(self):
        workspace=dict(documents=[dict(id='doc:README.md:0',text=METHOD)])
        with scope(wrapped(),workspace):
            with self.assertRaisesRegex(ValueError,'already recognized'):lookup('method_restrictions',METHOD,['no_external_data'])

    def test_hard_conditional_is_not_silently_flattened(self):
        quote='If training is small, only linear classifiers are permitted';path=fixture(self.root,quote+'.')
        with self.assertRaises(ValueError):self.resolve(path,wrapped(quote))

    def test_cached_normalization_and_rules_are_source_bound(self):
        path=fixture(self.root,METHOD+'.');run=self.root/'run';run.mkdir()
        task=self.resolve(path,wrapped(),RunLogger(run));restored=load_artifact(run/'semantic_artifact.json',path,path)
        self.assertEqual(task_digest(task),task_digest(restored))
        restored.spec['rule_resolution']['rules'][0]['reason']='A changed explanation'
        self.assertNotEqual(task_digest(task),task_digest(restored))
        with self.assertRaises(UnknownRestrictionWording):resolve_training_task(path)


if __name__=='__main__':unittest.main()
