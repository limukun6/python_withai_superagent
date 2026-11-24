"""No real training or API: production admission and incompatible-option checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression

from superagent.ab_run import validate_options
from superagent.cli import build_parser, _validate
from superagent.fit_guard import admitted_fits


class OfficialABTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args(['run','unused.zip','--scheduler','ab-v1','--portfolio','ab-v1',*extra])

    def test_supported_options_and_old_default(self):
        self.assertEqual(validate_options(self.args()), 11)
        legacy = build_parser().parse_args(['run','unused.zip'])
        _validate(legacy)
        self.assertEqual(legacy.trials, 5)

    def test_no_silent_option_ignoring(self):
        for flags in (['--trials','4'], ['--planner','llm'], ['--policy','random'],
                      ['--fit-limit','12'], ['--fit-limit','1'], ['--time-limit','nan'],
                      ['--feature-cache','run'], ['--api-key-file','unused'], ['--error-feedback']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_options(self.args(*flags))

    def test_fit_limit_precedes_second_classifier_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'fits.json'
            with patch.object(LogisticRegression, 'fit', return_value=None) as original:
                with self.assertRaisesRegex(RuntimeError, 'exceeds worker admission'):
                    with admitted_fits(dict(maximum_classifier_fits=1,ledger_path=str(ledger)), 'example'):
                        LogisticRegression().fit([], [])
                        LogisticRegression().fit([], [])
                self.assertEqual(original.call_count, 1)
            recorded = json.loads(ledger.read_text())
            self.assertEqual(recorded['attempted_classifier_fits'], 1)
            self.assertEqual(recorded['completed_classifier_fits'], 1)
            self.assertEqual(recorded['status'], 'failed')

    def test_failed_classifier_attempt_is_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'fits.json'
            with patch.object(LogisticRegression, 'fit', side_effect=ValueError('deliberate failure')):
                with self.assertRaisesRegex(ValueError, 'deliberate'):
                    with admitted_fits(dict(maximum_classifier_fits=1,ledger_path=str(ledger)), 'example'):
                        LogisticRegression().fit([], [])
            recorded = json.loads(ledger.read_text())
            self.assertEqual(recorded['attempted_classifier_fits'], 1)
            self.assertEqual(recorded['completed_classifier_fits'], 0)
            self.assertEqual(recorded['calls'][0]['status'], 'failed')

    def test_unknown_or_missing_fit_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Actual supervised fits'):
                with admitted_fits(dict(maximum_classifier_fits=1,ledger_path=str(Path(tmp)/'fits.json')), 'example'):
                    pass

    def test_successful_guard_restores_original_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'fits.json'
            with patch.object(LogisticRegression, 'fit', return_value=None) as original:
                with admitted_fits(dict(maximum_classifier_fits=1,ledger_path=str(ledger)), 'example'):
                    LogisticRegression().fit([], [])
                self.assertIs(LogisticRegression.fit, original)
            self.assertEqual(json.loads(ledger.read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
