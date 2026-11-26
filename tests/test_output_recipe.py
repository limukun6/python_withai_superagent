"""Output contract oracles use a deterministic prediction fixture, never model fitting."""
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import joblib
import numpy as np

from superagent.autonomous_checkpoint import frozen_spec, standby
from superagent.contracts import TaskSpec
from superagent.events import RunLogger
from superagent.format_sources import package_files
from superagent.metrics import MetricSpec
from superagent.output_recipe import validate_recipe, verify_export, write_export
from superagent.predict import create_predictions, validate_prediction_outputs
from superagent.semantic_contract import compile_plan, evidence_workspace
from tests.test_semantic_contract import proposal


REPO = Path(__file__).resolve().parents[1]


class ProbeClassifier:
    """Not a learned model: a fixed arithmetic probe to test actual export/reload wiring."""
    classes_ = np.array(['decline','accept'])
    def predict(self, inputs):
        return np.where(np.asarray(inputs, dtype=float)[:,0] > 5, 'accept', 'decline')
    def predict_proba(self, inputs):
        accept = np.asarray(inputs, dtype=float)[:,0] / 10
        return np.column_stack([1-accept, accept])


def recipe(*, probability=False, template=False):
    return dict(version='output-csv-v1', delimiter=';', encoding='utf-8-sig', header=True,
        columns=[dict(name='Ticket', kind='input_id', field='event_id'),
                 dict(name='Chance', kind='probability', class_label='accept') if probability
                 else dict(name='Answer', kind='label', mapping={'accept':'Y','decline':'N'})],
        order=dict(mode='template', member='sample.csv', id_column='Ticket') if template else dict(mode='input'))


def package(path, *, duplicate=False, missing_id=False, template_ids=('0010','0001'), probability=False):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('01_overview.md',
            'Learn from ledger.data; cohort learn is training and reserve is validation. '
            'Macro F1 evaluates admission. query.data has the requested outputs.')
        archive.writestr('02_dictionary.md',
            'decision_code is the admission disposition. signal is the numeric predictor. '
            'Output is semicolon CSV with BOM and Ticket,Answer. Ticket preserves event_id with leading zeros. '
            'Answer is Y for accept and N for decline. When supplied, follow sample.csv order.')
        archive.writestr('ledger.data', 'signal,decision_code,cohort\n'+''.join(
            f'{i},{"accept" if i%2 else "decline"},learn\n' for i in range(8)))
        archive.writestr('query.data', ('signal\n9\n2\n' if missing_id else
                                      f'event_id,signal\n0001,9\n{"0001" if duplicate else "0010"},2\n'))
        header = 'Ticket;Chance' if probability else 'Ticket;Answer'
        archive.writestr('sample.csv', header+'\n'+''.join(i+';\n' for i in template_ids))
    return path


class OutputRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = package(self.root/'task.zip')

    def task(self, output=None):
        answer = proposal()
        answer['version'] = 'semantic-task-v2'
        answer['plan']['output'] = output
        answer['evidence']['output'] = ['doc:02_dictionary.md:0']
        with package_files(self.path) as files:
            return compile_plan(files, evidence_workspace(files), answer)

    def frozen(self, output=None):
        task = self.task(output)
        metric = MetricSpec.for_task(task)
        spec = frozen_spec(task, TaskSpec.from_task(task).record(), metric)
        directory = self.root/'run'/'agent'; directory.mkdir(parents=True, exist_ok=True)
        logger = RunLogger(directory)
        logger.write_json('task_spec.json', spec)
        model = directory/'selected_model.joblib'
        joblib.dump(dict(kind=task.kind, candidate='probe', model=ProbeClassifier()), model)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        logger.write_json('final_selection.json', dict(model_format='candidate_bundle_v1',
            family='probe', selected_experiment='fixture_not_trained', model_file=model.name,
            model_sha256=sha(model), task_spec_sha256=sha(directory/'task_spec.json'), test_accessed=False))
        return directory, spec

    def read(self, output, rec):
        with output.open(encoding=rec['encoding'], newline='') as stream:
            return list(csv.reader(stream, delimiter=rec['delimiter']))

    def test_actual_prediction_preserves_leading_zeros_and_maps_labels(self):
        rec = recipe(); directory, _ = self.frozen(rec)
        dest = create_predictions(self.path, directory, self.root/'pred.csv')
        self.assertTrue(dest.read_bytes().startswith(b'\xef\xbb\xbf'))
        self.assertEqual(self.read(dest, rec), [['Ticket','Answer'],['0001','Y'],['0010','N']])
        report = json.loads((self.root/'prediction_manifest.json').read_text(encoding='utf8'))
        self.assertTrue(report['output_contract']['checks']['original_id_checked'])

    def test_template_reorders_predictions_by_id_not_by_position(self):
        rec = recipe(template=True); directory, _ = self.frozen(rec)
        dest = create_predictions(self.path, directory, self.root/'pred.csv')
        self.assertEqual(self.read(dest, rec), [['Ticket','Answer'],['0010','N'],['0001','Y']])

    def test_requested_probability_uses_class_identity_even_when_metric_is_f1(self):
        rec = recipe(probability=True); directory, _ = self.frozen(rec)
        dest = create_predictions(self.path, directory, self.root/'pred.csv')
        rows = self.read(dest, rec)
        self.assertEqual(rows[0], ['Ticket','Chance'])
        self.assertAlmostEqual(float(rows[1][1]), .9)
        self.assertAlmostEqual(float(rows[2][1]), .2)
        report = json.loads((self.root/'prediction_manifest.json').read_text(encoding='utf8'))
        self.assertTrue(report['prediction_contract']['probability_contract_checked'])
        self.assertEqual(report['score_class_order'], ['decline','accept'])

    def test_class_prior_partial_uses_same_contract_and_retains_partial_identity(self):
        task = self.task(recipe(probability=True)); metric = MetricSpec.for_task(task)
        directory = self.root/'prior'; directory.mkdir()
        standby(task, TaskSpec.from_task(task).record(), metric, directory, RunLogger(directory))
        dest = create_predictions(self.path, directory, self.root/'prior.csv')
        rows = self.read(dest, recipe(probability=True))
        self.assertEqual([r[0] for r in rows[1:]], ['0001','0010'])
        self.assertEqual([float(r[1]) for r in rows[1:]], [.5,.5])
        report = json.loads((self.root/'prediction_manifest.json').read_text(encoding='utf8'))
        self.assertEqual(report['delivery_status'], 'partial')
        self.assertFalse(report['refit'])

    def test_no_recipe_keeps_legacy_header_and_index(self):
        directory, _ = self.frozen()
        dest = create_predictions(self.path, directory, self.root/'old.csv')
        with dest.open(encoding='utf8', newline='') as stream: rows = list(csv.reader(stream))
        self.assertEqual(rows, [['row_id','prediction'],['0','accept'],['1','decline']])

    def test_headerless_label_only_and_declared_one_based_index(self):
        rec = recipe(); rec.update(header=False, encoding='utf-8', delimiter='\t')
        rec['columns'] = [dict(name='Index',kind='row_index',start=1),dict(name='Result',kind='label')]
        directory, _ = self.frozen(rec)
        dest = create_predictions(self.path, directory, self.root/'plain.tsv')
        self.assertEqual(self.read(dest, rec), [['1','accept'],['2','decline']])

    def test_duplicate_ids_reject_without_overwriting_previous_file(self):
        self.path = package(self.path, duplicate=True)
        directory, _ = self.frozen(recipe())
        out = self.root/'safe.csv'; out.write_text('previous artifact', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'duplicate input IDs'):
            create_predictions(self.path, directory, out)
        self.assertEqual(out.read_text(encoding='utf8'), 'previous artifact')

    def test_missing_id_field_is_not_replaced_by_generated_index(self):
        self.path = package(self.path, missing_id=True)
        directory, _ = self.frozen(recipe())
        with self.assertRaisesRegex(ValueError, 'ID field missing'):
            create_predictions(self.path, directory, self.root/'no.csv')

    def test_documented_repeated_entity_ids_can_preserve_input_order(self):
        self.path = package(self.path, duplicate=True)
        rec = recipe(); rec['columns'][0]['unique'] = False
        directory, _ = self.frozen(rec)
        dest = create_predictions(self.path, directory, self.root/'repeated.csv')
        self.assertEqual(self.read(dest,rec), [['Ticket','Answer'],['0001','Y'],['0001','N']])
        rec['order'] = dict(mode='template',member='sample.csv',id_column='Ticket')
        with self.assertRaisesRegex(ValueError,'unambiguous unique'):
            validate_recipe(rec,['accept','decline'])

    def test_template_cannot_add_drop_or_duplicate_prediction_ids(self):
        for i, identifiers in enumerate([('0010','9999'),('0010',),('0010','0010')]):
            self.path = package(self.path, template_ids=identifiers)
            directory, _ = self.frozen(recipe(template=True))
            with self.assertRaisesRegex(ValueError, 'ID sets differ'):
                create_predictions(self.path, directory, self.root/f'bad{i}.csv')

    def test_wrong_template_header_is_rejected(self):
        directory, _ = self.frozen(recipe(template=True, probability=True))
        with self.assertRaisesRegex(ValueError, 'template headers'):
            create_predictions(self.path, directory, self.root/'bad.csv')

    def test_bad_probability_output_and_unknown_class_are_rejected(self):
        rec = recipe(probability=True)
        bad = deepcopy(rec); bad['columns'][1]['class_label'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'outside training'):
            validate_recipe(bad, ['accept','decline'])
        _, spec = self.frozen(rec)
        with self.assertRaises(ValueError):
            validate_prediction_outputs(['accept','decline'], 2, spec, ['accept','decline'], [[1.2,-.2],[.1,.9]])
        with self.assertRaises(ValueError):
            write_export(self.path, spec, self.root/'bad.csv', ['accept','decline'], None, None)

    def test_label_mapping_must_cover_training_domain_and_be_injective(self):
        for mapping in ({'accept':'Y'}, {'accept':'Y','decline':'Y'}):
            rec = recipe(); rec['columns'][1]['mapping'] = mapping
            with self.assertRaises(ValueError): validate_recipe(rec, ['accept','decline'])

    def test_fabricated_output_evidence_is_rejected_during_compilation(self):
        answer = proposal(); answer['version'] = 'semantic-task-v2'
        answer['plan']['output'] = recipe(); answer['evidence']['output'] = ['fake']
        with package_files(self.path) as files:
            with self.assertRaisesRegex(ValueError, 'evidence for output'):
                compile_plan(files, evidence_workspace(files), answer)

    def test_independent_readback_detects_tampered_label_or_id(self):
        rec = recipe(template=True)
        path = self.root/'tampered.csv'
        for body in ('Ticket;Answer\n0010;Y\n0001;Y\n','Ticket;Answer\n0001;N\n0010;Y\n'):
            path.write_text(body, encoding='utf-8-sig')
            with self.assertRaisesRegex(ValueError, 'label or original ID'):
                verify_export(path, rec, ['accept','decline'], ['accept','decline'],
                              ids=['0001','0010'], order_indices=[1,0])

    def test_validation_failure_keeps_previous_artifact(self):
        rec = recipe(); directory, _ = self.frozen(rec)
        dest = self.root/'old.csv'; dest.write_text('existing', encoding='utf8')
        with patch('superagent.output_recipe.verify_export', side_effect=ValueError('oracle rejected')):
            with self.assertRaisesRegex(ValueError, 'oracle rejected'):
                create_predictions(self.path, directory, dest)
        self.assertEqual(dest.read_text(encoding='utf8'), 'existing')

    def test_frozen_model_predicts_in_a_fresh_cli_process_without_llm(self):
        rec = recipe(template=True); directory, _ = self.frozen(rec)
        first = create_predictions(self.path, directory, self.root/'first.csv')
        second = self.root/'replay'/'second.csv'
        env = dict(os.environ, SUPERAGENT_OFFLINE='1', HF_HUB_OFFLINE='1', PYTHONIOENCODING='utf-8')
        completed = subprocess.run([sys.executable, '-m', 'unified_agent', 'predict', str(self.path),
            '--run-dir', str(directory.parent), '--output', str(second)], cwd=REPO, env=env,
            capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_probability_request_filters_execution_capabilities_without_changing_metric(self):
        from superagent.autonomous_run import _train
        class CapabilitiesChecked(RuntimeError): pass
        task = self.task(recipe(probability=True))
        caps = {'label_only':dict(supports_probabilities=False), 'with_scores':dict(supports_probabilities=True)}
        def choices(eligible, profile, history, observed, metric, *args):
            self.assertEqual(set(eligible), {'with_scores'})
            self.assertEqual(metric.name, 'macro_f1')
            raise CapabilitiesChecked('checked before any fit')
        with patch('superagent.semantic_contract.resolve_training_task',return_value=task):
            with patch('superagent.autonomous_run.registered_capabilities',return_value=caps):
                with patch('superagent.autonomous_run.MethodSpace') as space:
                    space.return_value.refresh.return_value = caps, {}
                    with patch('superagent.autonomous_run.actions', side_effect=choices):
                        with self.assertRaises(CapabilitiesChecked):
                            _train(self.path,self.root/'capabilities',fit_limit=2)


if __name__ == '__main__': unittest.main()
