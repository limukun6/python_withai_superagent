"""Synthetic ledger fixtures for explicit current-reader migration; no real provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from superagent.events import RunLogger
from superagent.semantic_contract import resolve_training_task
from superagent.semantic_artifact import load_artifact
from superagent.task_budget import TaskBudgetLedger
from tests.test_default_handoff import npy_case
from tests.test_semantic_contract import Client
from tools.import_semantic_artifact import migrate


class SemanticMigrationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.task,self.query,self.answer,_=npy_case(self.root)
        self.run=self.root/'original';self.run.mkdir();self.log=RunLogger(self.run)
        resolve_training_task(self.task,Client([self.answer]),self.log,mode='proactive',prediction_package=self.query)
        self.ledger=TaskBudgetLedger.initialize(self.root/'SYNTHETIC.sqlite',new_project=True)
        self.model='DeepSeek-V4.1-Flash';self.key=self.ledger.reserve('fixture',.02,'SYNTHETIC',self.model)
        self.ledger.settle(self.key,self.model,100,30)
        with self.ledger._db() as db:units=db.execute('SELECT course_units FROM requests WHERE id=?',(self.key,)).fetchone()[0]
        self.events=[dict(kind='llm_request',role='task_semantic_interpretation',budget_request_id=self.key),
                     dict(kind='llm_response',role='task_semantic_interpretation',content=json.dumps(self.answer)),
                     dict(kind='semantic_contract_accepted',attempt=1)]
        self.usage=[dict(event='llm_call',budget_request_id=self.key,input_tokens=100,output_tokens=30,
                         served_model=self.model,course_cost_rmb=units/1e9,persistent_settlement_complete=True)]
        self.write();self.destination=self.root/'new_artifact.json'

    def write(self):
        for name,rows in [('events.jsonl',self.events),('llm_usage.jsonl',self.usage)]:
            (self.run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')

    def migrate(self):
        return migrate(self.task,self.query,self.run/'semantic_contract.json',self.ledger.path,self.destination)

    def test_native_plan_is_explicitly_recompiled_into_new_artifact(self):
        original=(self.run/'semantic_artifact.json').read_bytes();metadata=self.migrate()
        task=load_artifact(self.destination,self.task,self.query)
        self.assertEqual(len(task.y_train),8);self.assertEqual((self.run/'semantic_artifact.json').read_bytes(),original)
        self.assertEqual(metadata['provider_provenance']['original_request_id'],self.key)
        self.assertEqual(metadata['provider_provenance']['new_API_requests'],0)
        self.assertEqual(metadata['provider_provenance']['original_source_binding'],'original_artifact_content_and_package_hashes_verified')

    def test_later_planner_call_does_not_replace_semantic_receipt(self):
        key=self.ledger.reserve('fixture',.02,'SYNTHETIC',self.model);self.ledger.settle(key,self.model,200,10)
        self.events.extend([dict(kind='llm_request',role='experiment_planner',budget_request_id=key),
                            dict(kind='llm_response',role='experiment_planner',content='{"option":0}')])
        self.usage.append(dict(event='llm_call',budget_request_id=key,input_tokens=200,output_tokens=10,served_model=self.model,course_cost_rmb=.00024))
        self.write();metadata=self.migrate()
        self.assertEqual(metadata['provider_provenance']['original_request_id'],self.key)

    def test_changed_training_and_prediction_packages_are_not_cross_applied(self):
        for field in ['task','query']:
            path=getattr(self,field);before=path.read_bytes()
            with zipfile.ZipFile(path,'a') as archive:archive.writestr('extra_notice.txt','Different package')
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'different .* bytes'):self.migrate()
            path.write_bytes(before)

    def test_bad_source_artifact_checksum_is_refused(self):
        path=self.run/'semantic_artifact.json';record=json.loads(path.read_text(encoding='utf-8'))
        record['payload']['task_sha256']='changed';path.write_text(json.dumps(record),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'content/proposal'):self.migrate()

    def test_old_reader_identity_needs_explicit_tool_but_migration_can_revalidate(self):
        from superagent.semantic_artifact import digest
        path=self.run/'semantic_artifact.json';record=json.loads(path.read_text(encoding='utf-8'))
        record['payload']['reader_sha256']={'old_reader':'old'};record['payload_sha256']=digest(record['payload'])
        path.write_text(json.dumps(record),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'reader source changed'):load_artifact(path,self.task,self.query)
        self.assertTrue(self.migrate()['artifact_sha256'])

    def test_model_tokens_and_course_cost_must_match_canonical_settlement(self):
        original=deepcopy(self.usage)
        for change in [dict(input_tokens=101),dict(output_tokens=31),dict(served_model='other'),dict(course_cost_rmb=.99)]:
            self.usage=deepcopy(original);self.usage[0].update(change);self.write()
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'differs from metering'):self.migrate()
            self.assertFalse(self.destination.exists())

    def test_modified_response_or_missing_acceptance_is_refused(self):
        original=deepcopy(self.events)
        self.events[1]['content']='{}';self.write()
        with self.assertRaisesRegex(ValueError,'recorded semantic response'):self.migrate()
        self.events=[r for r in original if r['kind']!='semantic_contract_accepted'];self.write()
        with self.assertRaisesRegex(ValueError,'acceptance event'):self.migrate()

    def test_unresolved_canonical_request_cannot_attest_a_response(self):
        with self.ledger._db() as db:
            db.execute("UPDATE requests SET status='unknown',course_units=NULL WHERE id=?",(self.key,));db.commit()
        with self.assertRaisesRegex(ValueError,'known canonical'):self.migrate()

    def test_legacy_missing_binding_is_disclosed_instead_of_fabricated(self):
        (self.run/'semantic_artifact.json').unlink();metadata=self.migrate()
        self.assertEqual(metadata['provider_provenance']['original_source_binding'],'not_recorded_in_legacy_receipt')

    def test_three_scoped_question_versions_migrate_with_bound_synthetic_receipts(self):
        from tests.test_auto_semantics import standard_case
        for i,version in enumerate(['semantic-native-v2','semantic-task-v3','semantic-output-v2']):
            directory=self.root/f'new_version_{i}';directory.mkdir()
            if version=='semantic-native-v2':self.task,self.query,answer,_=npy_case(directory)
            else:
                self.task,answer,_=standard_case(directory/'task.zip',version=='semantic-output-v2');self.query=self.task
            answer.update(version=version,uncertainties=[dict(field='positive_label',question='No positive class is supplied.',evidence_refs=['doc:README.md:0'])])
            self.run=directory/'original';self.run.mkdir()
            resolve_training_task(self.task,Client([answer]),RunLogger(self.run),mode='proactive',prediction_package=self.query)
            self.key=self.ledger.reserve('fixture',.02,'SYNTHETIC',self.model);self.ledger.settle(self.key,self.model,100,30)
            with self.ledger._db() as db:units=db.execute('SELECT course_units FROM requests WHERE id=?',(self.key,)).fetchone()[0]
            self.events=[dict(kind='llm_request',role='task_semantic_interpretation',budget_request_id=self.key),
                         dict(kind='llm_response',content=json.dumps(answer)),dict(kind='semantic_contract_accepted',attempt=1)]
            self.usage=[dict(event='llm_call',budget_request_id=self.key,input_tokens=100,output_tokens=30,
                             served_model=self.model,course_cost_rmb=units/1e9,persistent_settlement_complete=True)]
            self.write();self.destination=directory/'new_artifact.json';metadata=self.migrate()
            restored=load_artifact(self.destination,self.task,self.query)
            self.assertEqual(restored.spec['uncertainty_resolution']['status'],'all_reported_questions_not_required')
            self.assertEqual(metadata['provider_provenance']['original_request_id'],self.key)


if __name__=='__main__':unittest.main()
