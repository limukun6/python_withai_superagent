"""SA1 uses a simulated LLM, real package readers, and no model fitting/network."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from superagent.events import RunLogger
from superagent.format_intake import adapt_training, predict_records
from superagent.format_sources import UnsupportedFormat, package_files
from superagent.semantic_contract import (ContractProbeError, compile_plan, evidence_workspace,
                                         resolve_training_task, semantic_training)


class Client:
    def __init__(self, replies, limit=3):
        self.replies, self.payloads = list(replies), []
        self.request_count, self.max_requests, self.blocked = 0, limit, False
    def set_deadline(self, deadline):
        self.deadline = deadline
    def ask(self, payload, log, role):
        self.payloads.append(deepcopy(payload))
        self.request_count += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            self.blocked = True
            raise reply
        return deepcopy(reply)


def fixture(root, manifest=False):
    path = root / 'anonymous.zip'
    records = 'signal,decision_code,cohort\n' + ''.join(
        f'{i},{"accept" if i % 2 else "decline"},learn\n' for i in range(8))
    records += '99,FORBIDDEN_LABEL,reserve\n'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('01_overview.md',
            'Learn from ledger.data. The cohort value learn denotes supervised development; reserve is validation. '
            'Classify the eventual admission disposition. Macro F1 is the primary criterion. '
            'Read query.data for the requested outputs.')
        archive.writestr('02_dictionary.md',
            'decision_code expresses the terminal admission decision, accept or decline. '
            'signal is the numeric predictor. cohort is exclusively a row role marker.')
        archive.writestr('ledger.data', records)
        archive.writestr('query.data', 'signal\n9\n2\n')
        archive.writestr('validation/answers.csv', 'FORBIDDEN_DATA_NEVER_IN_PROMPT')
        if manifest:
            archive.writestr('manifest.json', json.dumps({'schema_version': 1}))
    return path


def proposal():
    first, second = 'doc:01_overview.md:0', 'doc:02_dictionary.md:0'
    return dict(version='semantic-task-v1', plan=dict(
        training=dict(member='ledger.data', collection=None, delimiter=','),
        prediction=dict(member='query.data', collection=None, delimiter=','),
        target='decision_code', columns=[dict(name='signal', type='numeric')],
        metric='macro_f1', positive_label=None,
        split=dict(field='cohort', values=dict(train='learn', validation='reserve', test='query'))),
        evidence=dict(training=[first], prediction=[first], target=[first,second],
                      columns=[second], metric=[first], split=[first]), uncertainties=[])


class SemanticContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = fixture(self.root)

    def compile(self, reply):
        with package_files(self.path) as files:
            return compile_plan(files, evidence_workspace(files), reply)

    def test_real_multidocument_input_compiles_and_replays_prediction_order(self):
        original = self.path.read_bytes()
        client = Client([proposal()])
        with self.assertRaises(ValueError):
            adapt_training(self.path)
        task = resolve_training_task(self.path, client)
        self.assertEqual(client.request_count, 1)
        self.assertEqual(len(task.y_train), 8)
        self.assertEqual(set(task.y_train), {'accept', 'decline'})
        self.assertEqual(task.spec['metric'], 'macro_f1')
        self.assertEqual(task.X_train.shape, (8, 1))
        query = predict_records(self.path, task.kind, task.spec)
        self.assertEqual(query.shape, (2, 1))
        self.assertEqual(list(map(float, query[:, 0])), [9, 2])
        payload = json.dumps(client.payloads)
        self.assertNotIn('FORBIDDEN_LABEL', payload)
        self.assertNotIn('FORBIDDEN_DATA_NEVER_IN_PROMPT', payload)
        self.assertEqual(self.path.read_bytes(), original)

    def test_probe_error_reaches_one_semantic_repair(self):
        wrong = proposal(); wrong['plan']['target'] = 'nonexistent'
        client = Client([wrong, proposal()])
        log = RunLogger(self.root)
        task = semantic_training(self.path, client, log, ValueError('No local interpretation'))
        self.assertEqual(client.request_count, 2)
        self.assertEqual(client.payloads[1]['diagnostic']['message'], 'Target field is absent')
        receipt = json.loads((self.root/'semantic_contract.json').read_text(encoding='utf8'))
        self.assertEqual(receipt['attempt'], 2)
        self.assertFalse(receipt['semantic_truth_certified'])
        self.assertEqual(len(task.y_train), 8)

    def test_two_failed_probes_do_not_create_third_call(self):
        wrong = proposal(); wrong['plan']['target'] = 'absent'
        client = Client([wrong, wrong, proposal()])
        with self.assertRaises(UnsupportedFormat):
            semantic_training(self.path, client, None, ValueError('decode'))
        self.assertEqual(client.request_count, 2)

    def test_shared_remaining_allowance_applies_to_repair(self):
        wrong = proposal(); wrong['plan']['target'] = 'absent'
        client = Client([wrong, proposal()]); client.request_count = 2
        with self.assertRaises(UnsupportedFormat):
            semantic_training(self.path, client, None, ValueError('decode'))
        self.assertEqual(client.request_count, 3)
        self.assertEqual(len(client.payloads), 1)

    def test_transport_failure_is_not_a_semantic_retry(self):
        client = Client([RuntimeError('unresolved API usage'), proposal()])
        with self.assertRaisesRegex(RuntimeError, 'unresolved'):
            semantic_training(self.path, client, None, ValueError('decode'))
        self.assertEqual(client.request_count, 1)

    def test_fabricated_citations_are_rejected(self):
        reply = proposal(); reply['evidence']['target'] = ['invented']
        with self.assertRaisesRegex(ContractProbeError, 'fabricated'):
            self.compile(reply)

    def test_label_cannot_be_a_predictor(self):
        reply = proposal(); reply['plan']['columns'].append(dict(name='decision_code', type='categorical'))
        with self.assertRaisesRegex(ContractProbeError, 'Target, split'):
            self.compile(reply)

    def test_known_evaluation_source_is_rejected_before_read(self):
        reply = proposal(); reply['plan']['training']['member'] = 'validation/answers.csv'
        with self.assertRaisesRegex(ContractProbeError, 'evaluation split'):
            self.compile(reply)

    def test_unknown_row_marker_does_not_become_training(self):
        reply = proposal(); reply['plan']['split']['values'].pop('validation')
        with self.assertRaisesRegex(ValueError, 'Unknown or missing row split'):
            self.compile(reply)

    def test_explicit_manifest_is_not_overridden(self):
        self.path = fixture(self.root, manifest=True)
        client = Client([proposal()])
        with self.assertRaisesRegex(UnsupportedFormat, 'Explicit manifests'):
            semantic_training(self.path, client, None, ValueError('decode'))
        self.assertEqual(client.request_count, 0)

    def test_uncertainty_cannot_be_promoted_to_valid_task(self):
        reply = proposal(); reply['uncertainties'] = ['target unclear']
        with self.assertRaisesRegex(ContractProbeError, 'Unresolved'):
            self.compile(reply)

    def test_existing_native_fast_path_does_not_call_model(self):
        task = object(); client = Client([proposal()])
        with patch('superagent.semantic_contract.load_training_task', return_value=task):
            self.assertIs(resolve_training_task(self.path, client), task)
        self.assertEqual(client.request_count, 0)

    def test_official_training_entry_reaches_validated_checkpoint_without_fitting(self):
        from superagent.autonomous_run import _train
        class ReachedCheckpoint(RuntimeError): pass
        client = Client([proposal()])
        run = self.root/'official'
        with patch('superagent.autonomous_run.standby', side_effect=ReachedCheckpoint('probe complete')) as checkpoint:
            with patch('superagent.autonomous_run.Executor.worker', side_effect=AssertionError('No fits allowed')):
                with self.assertRaises(ReachedCheckpoint):
                    _train(self.path, run, fit_limit=2, flash=client)
        self.assertEqual(checkpoint.call_count, 1)
        task, spec, metric = checkpoint.call_args.args[:3]
        self.assertEqual(len(task.y_train), 8)
        self.assertEqual(metric.name, 'macro_f1')
        self.assertEqual(client.request_count, 1)
        self.assertTrue((run/'semantic_contract.json').is_file())


if __name__ == '__main__': unittest.main()
