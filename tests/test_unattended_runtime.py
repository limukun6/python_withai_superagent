"""Fault boundaries for unattended research: zero model fits and zero HTTP."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.run_unattended import run
from scripts.status_a29_choice_matrix import completion_failure
from tools.unattended_runtime import prevent_idle_sleep, ES_CONTINUOUS, ES_SYSTEM_REQUIRED


class UnattendedTests(unittest.TestCase):
    def test_lease_only_requests_system_and_releases_on_exception(self):
        setter=Mock(return_value=1)
        with self.assertRaisesRegex(ValueError,'child failure'):
            with prevent_idle_sleep(setter):
                raise ValueError('child failure')
        self.assertEqual([call.args[0] for call in setter.call_args_list],
                         [ES_CONTINUOUS|ES_SYSTEM_REQUIRED,ES_CONTINUOUS])

    def test_refused_lease_never_yields_or_retries(self):
        setter=Mock(return_value=0)
        with self.assertRaisesRegex(OSError,'not started'):
            with prevent_idle_sleep(setter):
                self.fail('Cannot execute job without a lease')
        setter.assert_called_once()

    def test_release_error_is_reported(self):
        with self.assertRaisesRegex(OSError,'release'):
            with prevent_idle_sleep(Mock(side_effect=[1,0])):
                pass

    def test_finalization_failure_has_specific_stage(self):
        outcome=dict(requested_planner_completed=False,finalization=dict(refit_failure=dict(
            status='timeout',error='No time remains for final refit')))
        self.assertEqual(completion_failure(outcome),dict(stage='final_refit',status='timeout',error='No time remains for final refit'))
        self.assertIsNone(completion_failure(dict(requested_planner_completed=True)))

    def test_guard_failure_never_launches_child(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with patch('scripts.run_unattended.prevent_idle_sleep',side_effect=OSError('refused')), \
                    patch('scripts.run_unattended.subprocess.Popen') as process:
                with self.assertRaises(OSError):
                    run(SimpleNamespace(output=output,max_seconds=1,command=['fake']))
            process.assert_not_called()
            record=json.loads((output/'completion.json').read_text())
            self.assertFalse(record['child_started']); self.assertEqual(record['status'],'stopped')

    def test_nonzero_child_code_propagates_once(self):
        @contextmanager
        def lease(): yield
        child=Mock();child.wait.return_value=7;child.poll.return_value=7
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run'
            with patch('scripts.run_unattended.prevent_idle_sleep',lease), \
                    patch('scripts.run_unattended.subprocess.Popen',return_value=child) as process:
                self.assertEqual(run(SimpleNamespace(output=output,max_seconds=1,command=['fake'])),7)
            process.assert_called_once(); child.wait.assert_called_once_with(timeout=1)
            self.assertEqual(json.loads((output/'completion.json').read_text())['child_exit_code'],7)


if __name__=='__main__':unittest.main()
