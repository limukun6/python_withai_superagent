"""Read-only inspection of an interrupted AB run, for a safe resume (A150).

The expensive parts of a run are classifier fits and planning requests. A resume
must therefore start from what the run directory and the project ledger already
prove, never from an assumption. This module only inspects: it writes nothing,
charges nothing, and refuses whenever the recorded evidence and the ledger
disagree, because continuing on a mismatch is how a fit or a request gets paid
for twice.

The continuation itself is deliberately not implemented here. What is
implemented is the decision a continuation must obey.
"""
import json
from pathlib import Path

COMPLETE = 'complete'
RESUMABLE = 'resumable'
BLOCKED = 'blocked'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _usage_events(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _ledger_state(ledger_path, scope):
    if ledger_path is None:
        return None
    import sqlite3
    con = sqlite3.connect(f'file:{Path(ledger_path).resolve()}?mode=ro', uri=True)
    try:
        rows = con.execute('select status, count(*), coalesce(sum(course_units),0) '
                           'from requests where scope=? group by status', (scope,)).fetchall()
    finally:
        con.close()
    return {status: {'requests': count, 'course_units': units} for status, count, units in rows}


def inspect(run_dir, ledger_path=None, sources_now=None, task_archive_sha256=None):
    """Return a resume decision for one run directory.

    `sources_now` maps the executor source files to their current digests; when
    supplied it must equal the recorded map, otherwise the code moved under the
    run and a resume would mix two versions.
    """
    run_dir = Path(run_dir)
    blockers = []
    manifest_path = run_dir / 'run_manifest.json'
    if not manifest_path.is_file():
        return {'status': BLOCKED, 'blockers': ['run_manifest.json is missing'],
                'run_dir': str(run_dir)}
    manifest = _read(manifest_path)
    budget_before = manifest.get('task_budget_before_run') or {}
    scope = budget_before.get('task_budget_id')

    outcome_path = run_dir / 'run_outcome.json'
    outcome = _read(outcome_path) if outcome_path.is_file() else None
    history = _read(run_dir / 'history.json') if (run_dir / 'history.json').is_file() else []
    fits = _read(run_dir / 'classifier_fits.json') if (run_dir / 'classifier_fits.json').is_file() else {}
    usage = _usage_events(run_dir / 'llm_usage.jsonl')

    recorded_fits = fits.get('attempted_classifier_fits', 0)
    recorded_requests = sum(1 for event in usage if event.get('event') == 'llm_call')
    completed_trials = [row for row in history if row.get('status') == 'completed']

    if outcome and outcome.get('status') == 'completed':
        return {'status': COMPLETE, 'run_dir': str(run_dir), 'scope': scope,
                'blockers': [], 'reason': 'the run already finished; a resume would repeat work',
                'recorded_fits': recorded_fits, 'recorded_requests': recorded_requests}

    if not scope and (manifest.get('llm_request_limit') or 0) > 0:
        blockers.append('the manifest does not record a task budget id')
    ledger = _ledger_state(ledger_path, scope) if scope else None
    if ledger is not None:
        unknown = ledger.get('unknown', {}).get('requests', 0)
        pending = ledger.get('pending', {}).get('requests', 0)
        if unknown or pending:
            blockers.append(f'the ledger holds {unknown} unknown and {pending} pending rows for this scope')
        known_requests = ledger.get('known', {}).get('requests', 0)
        if known_requests < recorded_requests:
            blockers.append(f'the ledger records {known_requests} settled requests but the run logged '
                            f'{recorded_requests}')

    recorded_sources = manifest.get('source_files') or {}
    if sources_now is not None and recorded_sources != sources_now:
        changed = sorted(key for key in set(recorded_sources) | set(sources_now)
                         if recorded_sources.get(key) != sources_now.get(key))
        blockers.append(f'executor sources changed since the run: {changed}')
    if task_archive_sha256 is not None and manifest.get('task_archive_sha256') != task_archive_sha256:
        blockers.append('the task archive differs from the one the run used')

    trials = sorted(row['trial'] for row in completed_trials)
    if trials and trials != list(range(1, len(trials) + 1)):
        blockers.append(f'completed trials are not contiguous from 1: {trials}')
    if fits.get('attempted_classifier_fits', 0) != fits.get('completed_classifier_fits', 0):
        blockers.append('the last fit attempt did not complete; its cost is already recorded')

    fit_limit = manifest.get('fit_limit')
    request_limit = manifest.get('llm_request_limit')
    remaining_fits = None if fit_limit is None else fit_limit - recorded_fits
    remaining_requests = None if request_limit is None else request_limit - recorded_requests
    if remaining_fits is not None and remaining_fits <= 1:
        blockers.append(f'only {remaining_fits} fit(s) remain; one is reserved for the final refit')
    if remaining_requests is not None and (request_limit or 0) > 0 and remaining_requests <= 0:
        blockers.append('no planning request remains')

    return {'status': BLOCKED if blockers else RESUMABLE,
            'run_dir': str(run_dir), 'scope': scope, 'blockers': blockers,
            'recorded_fits': recorded_fits, 'recorded_requests': recorded_requests,
            'completed_trials': trials, 'next_trial': (trials[-1] + 1) if trials else 1,
            'fit_limit': fit_limit, 'request_limit': request_limit,
            'remaining_fits': remaining_fits, 'remaining_requests': remaining_requests,
            'ledger': ledger}


def adopt(parent_run_dir, target_run_dir, report=None, ledger_path=None,
          sources_now=None, task_archive_sha256=None):
    """Copy an interrupted run's evidence into a new directory, or refuse.

    A resume must never write into the interrupted directory: the original stays
    the record of what actually happened. Only the artifacts a continuation needs
    are copied, and a provenance file records where they came from.
    """
    import hashlib
    import shutil

    parent = Path(parent_run_dir).resolve()
    target = Path(target_run_dir).resolve()
    if parent == target:
        raise ValueError('A resume must write to a new directory, not the interrupted one')
    report = report or inspect(parent, ledger_path, sources_now=sources_now,
                               task_archive_sha256=task_archive_sha256)
    if report['status'] != RESUMABLE:
        raise ValueError('Refusing to adopt a run that is not resumable: '
                         + (report.get('reason') or '; '.join(report.get('blockers') or [])))
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for artifact in sorted(parent.glob('trial_*.joblib')) + sorted(parent.glob('*.joblib')):
        if artifact.name.startswith('trial_') or artifact.name in (
                'primary.joblib', 'confirmation_1.joblib', 'confirmation_2.joblib'):
            if artifact.name in copied:
                continue
            shutil.copy2(artifact, target / artifact.name)
            copied.append(artifact.name)
    for name in ('history.json', 'classifier_fits.json', 'prediction_evidence.json',
                 'primary_split.json', 'confirmation_1_split.json', 'confirmation_2_split.json'):
        source = parent / name
        if source.is_file():
            shutil.copy2(source, target / name)
            copied.append(name)
    history = _read(target / 'history.json') if (target / 'history.json').is_file() else []
    evidence = _read(target / 'prediction_evidence.json') if (target / 'prediction_evidence.json').is_file() else []
    provenance = {'resumed_from': str(parent),
                  'parent_manifest_sha256': hashlib.sha256((parent / 'run_manifest.json').read_bytes()).hexdigest(),
                  'adopted_files': copied, 'adopted_trials': sorted(row['trial'] for row in history),
                  'parent_fits': report.get('recorded_fits'),
                  'parent_requests': report.get('recorded_requests'),
                  'next_trial': report.get('next_trial')}
    (target / 'resume_from.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n',
                                            encoding='utf-8')
    adopted_fits = _read(target / 'classifier_fits.json') if (target / 'classifier_fits.json').is_file() else {}
    return {'history': history, 'evidence': evidence, 'provenance': provenance,
            'workers': adopted_fits.get('workers') or [],
            'fits_used': report.get('recorded_fits', 0),
            'requests_used': report.get('recorded_requests', 0)}