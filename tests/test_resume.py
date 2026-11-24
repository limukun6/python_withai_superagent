import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from superagent.resume import BLOCKED, COMPLETE, RESUMABLE, adopt, inspect

SOURCES = {'ab_run.py': 'a' * 64, 'cli.py': 'b' * 64}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_run(root, *, trials=2, attempts=2, completed=2, requests=1, outcome=None,
             archive='c' * 64, sources=None, fit_limit=5, request_limit=3):
    run = Path(root) / 'run'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'run_manifest.json').write_text(json.dumps({
        'fit_limit': fit_limit, 'llm_request_limit': request_limit,
        'task_archive_sha256': archive,
        'source_files': SOURCES if sources is None else sources,
        'task_budget_before_run': {'task_budget_id': 'scope-a'}}), encoding='utf-8')
    history = [{'trial': index, 'candidate': f'c{index}', 'status': 'completed'}
               for index in range(1, trials + 1)]
    (run / 'history.json').write_text(json.dumps(history), encoding='utf-8')
    (run / 'classifier_fits.json').write_text(json.dumps({
        'attempted_classifier_fits': attempts, 'completed_classifier_fits': completed,
        'unknown_failed_attempts': 0}), encoding='utf-8')
    events = ['ledger_initialized'] + ['llm_call'] * requests
    (run / 'llm_usage.jsonl').write_text(
        '\n'.join(json.dumps({'event': event}) for event in events) + '\n', encoding='utf-8')
    if outcome:
        (run / 'run_outcome.json').write_text(json.dumps({'status': outcome}), encoding='utf-8')
    return run


def make_ledger(path, rows):
    con = sqlite3.connect(path)
    con.execute('create table requests(id text primary key, scope text not null, status text not null, '
                'reserved_units integer not null, course_units integer, payload text not null)')
    for index, (scope, status, units) in enumerate(rows):
        con.execute('insert into requests values (?,?,?,?,?,?)',
                    (f'r{index}', scope, status, units or 0, units, '{}'))
    con.commit()
    con.close()
    return path


class CompletedRunTests(unittest.TestCase):
    def test_a_finished_run_is_complete(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, outcome='completed')
            self.assertEqual(inspect(run)['status'], COMPLETE)


class ResumableTests(unittest.TestCase):
    def test_an_interrupted_run_with_budget_left_is_resumable(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=1, completed=1, requests=1)
            ledger = make_ledger(Path(root) / 'ledger.sqlite', [('scope-a', 'known', 10)])
            report = inspect(run, ledger, sources_now=SOURCES)
            self.assertEqual(report['status'], RESUMABLE, report['blockers'])
            self.assertEqual(report['next_trial'], 2)
            self.assertEqual(report['remaining_fits'], 4)
            self.assertEqual(report['remaining_requests'], 2)

    def test_inspection_does_not_write_the_ledger(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=1, completed=1)
            ledger = make_ledger(Path(root) / 'ledger.sqlite', [('scope-a', 'known', 10)])
            before = digest(ledger)
            inspect(run, ledger, sources_now=SOURCES)
            self.assertEqual(digest(ledger), before)
            self.assertEqual(sorted(p.name for p in run.iterdir()),
                             ['classifier_fits.json', 'history.json', 'llm_usage.jsonl', 'run_manifest.json'])


class BlockedTests(unittest.TestCase):
    def test_unknown_ledger_rows_block_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=1, completed=1)
            ledger = make_ledger(Path(root) / 'ledger.sqlite', [('scope-a', 'known', 10),
                                                                ('scope-a', 'unknown', 0)])
            self.assertIn('unknown', ' '.join(inspect(run, ledger)['blockers']))

    def test_changed_sources_block_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=1, completed=1)
            report = inspect(run, sources_now={'ab_run.py': 'z' * 64, 'cli.py': 'b' * 64})
            self.assertEqual(report['status'], BLOCKED)
            self.assertIn('ab_run.py', ' '.join(report['blockers']))

    def test_a_different_task_archive_blocks_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=1, completed=1)
            report = inspect(run, task_archive_sha256='d' * 64)
            self.assertIn('task archive', ' '.join(report['blockers']))

    def test_an_incomplete_fit_attempt_blocks_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=2, completed=1)
            self.assertIn('did not complete', ' '.join(inspect(run)['blockers']))

    def test_no_remaining_request_blocks_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=1, attempts=1, completed=1, requests=3, request_limit=3)
            self.assertIn('no planning request', ' '.join(inspect(run)['blockers']))

    def test_only_the_reserved_refit_left_blocks_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            run = make_run(root, trials=4, attempts=4, completed=4, fit_limit=5)
            self.assertIn('reserved for the final refit', ' '.join(inspect(run)['blockers']))

    def test_a_missing_manifest_blocks_a_resume(self):
        with tempfile.TemporaryDirectory() as root:
            report = inspect(Path(root))
            self.assertEqual(report['status'], BLOCKED)


class AdoptTests(unittest.TestCase):
    def test_adoption_copies_evidence_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            parent = make_run(root, trials=1, attempts=1, completed=1)
            (parent / 'trial_1.joblib').write_bytes(b'fake-model')
            target = Path(root) / 'resumed'
            adopted = adopt(parent, target, sources_now=SOURCES)
            self.assertTrue((target / 'history.json').is_file())
            self.assertTrue((target / 'trial_1.joblib').is_file())
            provenance = json.loads((target / 'resume_from.json').read_text(encoding='utf-8'))
            self.assertEqual(provenance['adopted_trials'], [1])
            self.assertEqual(adopted['fits_used'], 1)
            self.assertEqual(len(adopted['history']), 1)
            self.assertEqual(sorted(p.name for p in parent.iterdir()),
                             ['classifier_fits.json', 'history.json', 'llm_usage.jsonl',
                              'run_manifest.json', 'trial_1.joblib'])

    def test_adoption_refuses_a_completed_run(self):
        with tempfile.TemporaryDirectory() as root:
            parent = make_run(root, outcome='completed')
            with self.assertRaises(ValueError):
                adopt(parent, Path(root) / 'resumed')

    def test_adoption_refuses_to_overwrite_the_parent(self):
        with tempfile.TemporaryDirectory() as root:
            parent = make_run(root, trials=1, attempts=1, completed=1)
            with self.assertRaises(ValueError):
                adopt(parent, parent, sources_now=SOURCES)


class CliWiringTests(unittest.TestCase):
    def test_the_command_parses_and_reports_the_decision(self):
        from superagent.cli import _resume_status, build_parser
        args = build_parser().parse_args(['resume-status', 'nowhere'])
        self.assertEqual(args.handler, _resume_status)
        self.assertEqual(_resume_status(args), 2)


if __name__ == '__main__':
    unittest.main()
