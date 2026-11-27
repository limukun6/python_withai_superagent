"""Integrated real export/recovery wiring with fixed, untrained arithmetic models."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from superagent.autonomous_checkpoint import standby
from superagent.contracts import TaskSpec
from superagent.events import RunLogger
from superagent.first_delivery import first_prediction,final_prediction
from superagent.format_sources import package_files
from superagent.metrics import MetricSpec
from superagent.prediction_adapter import write_adapter,sha
from superagent.predict import create_predictions,PredictionAdaptationError
from superagent.semantic_contract import compile_plan,evidence_workspace
from tests import test_delivery_recovery as recovery_fixture
from tests.test_delivery_recovery import adapter
from tests.test_semantic_contract import Client,proposal
from tests.test_output_recipe import recipe


class JointDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=recovery_fixture.DeliveryRecoveryTests('test_adapter_changes_only_query_mapping_and_existing_id_source')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);f=self.fixture
        self.root,self.run,self.model,self.task,self.query=f.root,f.run,f.model,f.task,f.query
        reply=proposal();reply['version']='semantic-task-v2';reply['plan']['output']=recipe();reply['evidence']['output']=['doc:02_dictionary.md:0']
        with package_files(self.task) as files:task=compile_plan(files,evidence_workspace(files),reply)
        selection=(self.model/'final_selection.json').read_bytes()
        spec=(self.model/'task_spec.json').read_bytes()
        standby(task,TaskSpec.from_task(task).record(),MetricSpec.for_task(task),self.model,RunLogger(self.model))
        self.output=self.root/'delivery/predictions.csv'
        first_prediction(self.task,self.run,self.output)
        (self.model/'final_selection.json').write_bytes(selection);(self.model/'task_spec.json').write_bytes(spec)
        self.before={n:sha(self.model/n) for n in ('model.joblib','task_spec.json','final_selection.json','classifier_fits.json')}

    def cached_adapter(self):
        path=self.run/'delivery_recovery/cached/prediction_adapter.json';path.parent.mkdir(parents=True)
        write_adapter(path,self.query,self.model,adapter(),{'prediction:doc:README.md:0'},{'rows':2})
        return path

    def receipt(self):return json.loads((self.output.parent/'prediction_manifest.json').read_text(encoding='utf-8'))

    def test_cached_adaptation_is_published_as_selected_with_actual_id_source(self):
        path=self.cached_adapter();reference=self.root/'reference/predictions.csv';create_predictions(self.task,self.model,reference)
        with patch('superagent.delivery_recovery.recover_prediction',side_effect=AssertionError('cache must suffice')):
            final_prediction(self.query,self.run,self.output)
            final_prediction(self.query,self.run,self.output)
        self.assertEqual(reference.read_bytes(),self.output.read_bytes())
        self.assertEqual(self.receipt()['delivery_phase'],'selected')
        self.assertEqual(self.receipt()['output_contract']['columns'][0]['field'],'parcel_id')
        self.assertEqual(len(list((path.parent/'deliveries').glob('*.json'))),2)
        self.assertEqual(self.before,{n:sha(self.model/n) for n in self.before})

    def test_input_recovery_runs_before_partial_fallback(self):
        client=Client([adapter()]);client.spent=0
        with patch('superagent.delivery_recovery.bound_client',return_value=(client,0)):
            final_prediction(self.query,self.run,self.output)
        self.assertEqual(client.request_count,1)
        self.assertEqual(self.receipt()['delivery_phase'],'selected')
        self.assertEqual(self.receipt()['delivery_status'],'completed')
        self.assertEqual(self.before,{n:sha(self.model/n) for n in self.before})

    def test_unrecoverable_new_input_preserves_the_previous_file(self):
        before=self.output.read_bytes()
        with patch('superagent.delivery_recovery.recover_prediction',return_value=None):
            with self.assertRaises(ValueError):final_prediction(self.query,self.run,self.output)
        self.assertEqual(before,self.output.read_bytes())
        self.assertEqual(self.receipt()['delivery_phase'],'first')

    def test_damaged_selected_model_retains_honest_partial_delivery(self):
        early=self.output.read_bytes();(self.model/'model.joblib').write_bytes(b'broken')
        final_prediction(self.task,self.run,self.output)
        self.assertEqual(self.output.read_bytes(),early)
        self.assertEqual((self.receipt()['delivery_status'],self.receipt()['delivery_phase']),('partial','recovered_first'))
        self.assertIn('hash',self.receipt()['delivery_issue'])

    def test_explicit_invalid_adapter_is_not_silently_ignored(self):
        path=self.root/'invalid.json';path.write_text('{}')
        early=self.output.read_bytes()
        with self.assertRaises(ValueError):final_prediction(self.query,self.run,self.output,prediction_adapter=path)
        self.assertEqual(early,self.output.read_bytes())

    def test_publication_error_cannot_downgrade_the_selected_result(self):
        with patch('superagent.first_delivery.publish',side_effect=RuntimeError('publication failure')) as publish:
            with self.assertRaisesRegex(RuntimeError,'publication failure'):final_prediction(self.task,self.run,self.output)
        self.assertEqual(publish.call_count,1)

    def test_input_feature_exclusivity_is_not_a_method_ban(self):
        from superagent.task_constraints import documented_flags,RestrictionError
        self.assertEqual(documented_flags('utterance is the only input feature and is text.'),{})
        with self.assertRaises(RestrictionError):documented_flags('Only random forests are permitted.')

    def test_classifier_worker_cannot_inherit_the_solve_control_pipe(self):
        import subprocess
        from types import SimpleNamespace
        from superagent.benchmark import train_candidate
        with patch('superagent.benchmark.subprocess.run',return_value=SimpleNamespace(returncode=1,stderr='untrained fixture')) as run:
            model,facts=train_candidate(self.root/'unused.joblib','unused',self.root/'result.joblib',1)
        self.assertIs(run.call_args.kwargs['stdin'],subprocess.DEVNULL)
        self.assertIsNone(model)

    def test_early_prior_counts_the_frozen_prediction_aliases(self):
        from superagent.partial_classification import input_count
        from superagent.prediction_adapter import apply_proposal
        prior=json.loads((self.model/'partial_model.json').read_text(encoding='utf-8'))
        prior['frozen_spec']=apply_proposal(self.fixture.spec,adapter(),{'prediction:doc:README.md:0'})
        self.assertEqual(input_count(self.query,prior),2)


if __name__=='__main__':unittest.main()
