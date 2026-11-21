"""A-owned official CLI execution of B's AB-v1 catalogue and local scheduler."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import joblib

from .ab_catalog import ADAPTER_VERSION, capabilities_for, untried_candidates
from .ab_scheduler import PRIMARY, CONFIRMATIONS, ROW_SCOPE, next_action, selection_evidence
from .benchmark import train_candidate
from .confirmation import decide
from .contracts import TaskSpec
from .data import load_development_task
from .events import RunLogger
from .finalization import finalize
from .metrics import MetricSpec
from .prediction_evidence import fingerprint
from .scheduler_bridge import dispatch_action
from .splits import grouped_partition, take, sample_groups, overlap_audit
from .task_semantics import declared_semantics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*.py'))}


def validate_options(args):
    if args.planner != 'local':
        raise ValueError('AB-v1 CLI currently supports local planning only; Flash integration is pending')
    if any(getattr(args, k) is not None for k in ('api_url', 'api_key_file', 'model', 'budget_ledger',
            'task_budget_id', 'llm_budget_rmb', 'policy', 'trials')):
        raise ValueError('AB-v1 uses fit-limit and its shared local scheduler, without API or fixed-trials options')
    if (args.portfolio != 'ab-v1' or args.architecture != 'single' or args.anchor != 'catalog'
            or args.refinement != 'off' or args.allocation != 'off' or args.feature_cache != 'off'
            or args.time_policy != 'shared' or args.error_feedback):
        raise ValueError('AB-v1 requires its own portfolio, catalog anchor, single architecture and default execution options')
    if args.run_dir and args.output_dir:
        raise ValueError('Use either run-dir or output-dir')
    limit = 11 if args.fit_limit is None else args.fit_limit
    if type(limit) is not int or not 2 <= limit <= 11:
        raise ValueError('AB fit-limit must be an integer in [2,11], including final refit')
    if any(not math.isfinite(x) or x <= 0 for x in (args.time_limit, args.trial_timeout)):
        raise ValueError('Time limits must be positive and finite')
    return limit


def freeze_partitions(task, directory, seed, metric):
    trusted, allowed = {}, {}
    groups = sample_groups(task, 'shared_text')
    seen = set()
    for index, split in enumerate((PRIMARY, *CONFIRMATIONS)):
        split_seed = seed + 101 * index
        train, held, resource = grouped_partition(task, split_seed, 'shared_text')
        identity = frozenset(held.tolist())
        if identity in seen:
            raise ValueError('Registered confirmation repeats the same rows; no independent partition available')
        seen.add(identity)
        payload = directory / (split + '.joblib')
        joblib.dump(dict(kind=task.kind, X_train=take(task.X_train, train), y_train=task.y_train[train],
                        X_search=take(task.X_train, held), y_search=task.y_train[held],
                        metric_spec=metric.record(), pair_semantics=declared_semantics(task)), payload)
        trusted[split] = dict(row_ids=held.tolist(), labels=task.y_train[held].tolist(),
                             groups=groups[held].tolist(), evaluation_payload_sha256=sha(payload))
        allowed[split] = dict(id=split, seed=split_seed, row_ids_sha256=fingerprint(held.tolist()),
                             label_sha256=fingerprint(trusted[split]['labels']),
                             evaluation_payload_sha256=sha(payload))
        (directory / (split + '_split.json')).write_text(json.dumps(dict(
            **allowed[split], train_indices=train.tolist(), search_indices=held.tolist(),
            split_resource=resource, **overlap_audit(task, train, held)), indent=2), encoding='utf-8')
    return trusted, allowed


class Executor:
    def __init__(self, directory, caps, metric, start, wall, timeout, limit, logger):
        self.directory, self.caps, self.metric = directory, caps, metric
        self.start, self.wall, self.timeout, self.limit = start, wall, timeout, limit
        self.logger = logger
        self.records = []

    def accounting(self):
        return dict(attempted_classifier_fits=sum(r.get('attempted_classifier_fits', 0) for r in self.records),
                    completed_classifier_fits=sum(r.get('completed_classifier_fits', 0) for r in self.records),
                    unknown_failed_attempts=sum(r.get('usage_unknown', False) for r in self.records))

    def worker(self, payload, name, destination, timeout, *, reserve=0):
        cap = self.caps[name]
        expected = cap['refit_classifier_fits'] if Path(payload).name == 'refit_data.joblib' else cap['classifier_fits']
        counts = self.accounting()
        if counts['unknown_failed_attempts'] or counts['attempted_classifier_fits'] + expected + reserve > self.limit:
            raise ValueError('Insufficient known actual-fit allowance for worker plus reserved refit')
        remaining = self.wall - (time.monotonic() - self.start)
        if remaining <= 0:
            raise TimeoutError('Shared wall-clock limit reached before worker')
        ledger = Path(destination).with_suffix('.fits.json')
        admission = Path(destination).with_suffix('.admission.json')
        admission.write_text(json.dumps(dict(maximum_classifier_fits=expected, ledger_path=str(ledger))), encoding='utf-8')
        model, facts = train_candidate(payload, name, destination, min(timeout, remaining), fit_admission_path=admission)
        record = json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else dict(
            candidate=name, usage_unknown=True, status='ledger_unavailable')
        record['artifact'] = Path(destination).name
        self.records.append(record)
        self.logger.write_json('classifier_fits.json', dict(**self.accounting(), workers=self.records))
        self.logger.event('worker_accounted', worker=record, result_status=facts['status'])
        if model is None:
            raise RuntimeError('Worker failed without retry: ' + name + ': ' + str(facts.get('error')))
        if (record.get('status') != 'completed' or record['candidate'] != name
                or record.get('attempted_classifier_fits') != expected
                or record.get('completed_classifier_fits') != expected or facts.get('classifier_fits') != expected):
            raise ValueError('Worker result and physical fit admission disagree')
        config = facts['resolved_configuration']
        if config['id'] != name or config['params'] != cap['params']:
            raise ValueError('Worker executed a different candidate configuration')
        if facts['estimator']['family'] != cap['estimator']:
            raise ValueError('Fitted classifier differs from its registered capability')
        if self.metric.check_identity(facts.get('metric_spec'))['status'] != 'confirmed':
            raise ValueError('Worker metric identity is not confirmed')
        return model, facts


def finish(task, history, state, trusted, allowed, caps, directory, executor, logger):
    review = selection_evidence(state, trusted_partitions=trusted)
    chosen = review['incumbent']
    if chosen is None:
        raise ValueError('No verified completed candidate to freeze')
    primary = [r for r in history if r['split_id'] == PRIMARY]
    lookup = {(r['candidate'], r['split_id']): r for r in history}
    last = review['promotions'][-1] if review['promotions'] else None
    names = [last['previous'], chosen] if last else [chosen]
    basis = review['basis']
    measurements = {name: [dict(seed=allowed[s]['seed'], **lookup[name, s]) for s in basis] for name in names}
    decision = decide([lookup[n, PRIMARY] for n in names], measurements,
                      [allowed[s]['seed'] for s in basis], 'shared_text_for_pairs_exact_rows_otherwise')
    if not decision['valid_confirmation'] or decision['selected_candidate'] != chosen:
        raise ValueError('A final selection disagrees with B matched evidence')
    decision.update(b_scheduler_review=review,
        selection_rule='Best of initial three; replace on two paired wins and mean gain >= 0.002',
        independent_confirmation_observed=any(s != PRIMARY for s in basis))
    logger.write_json('confirmation.json', decision)
    row = lookup[chosen, PRIMARY]
    model = joblib.load(directory / f"trial_{row['trial']}.joblib")['model']
    result = finalize(task, primary, model, chosen, row['search_score'], joblib.load(directory / 'primary.joblib'),
                      directory, logger, executor.worker, executor.start, executor.wall, executor.timeout,
                      True, False, confirmation=decision)
    return chosen, decision, result


def command(args):
    limit = validate_options(args)
    task_path = Path(args.task_package).expanduser().resolve()
    task = load_development_task(task_path)
    spec = TaskSpec.from_task(task).record()
    metric = MetricSpec.for_task(task)
    if metric.requires_scores:
        raise ValueError('AB CLI currently requires accuracy or macro_f1; score-level AUC evidence is pending')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = Path(args.run_dir or Path(args.output_dir or 'runs') / ('ab_' + stamp)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(directory)
    logger.initialize_usage_ledger()  # Per-run zero-call log, never a replacement for the project API ledger.
    began = time.monotonic()
    frozen = sources()
    history, evidence, actions = [], [], []
    executor = None
    outcome = dict(run_dir=str(directory), scheduler='ab-v1', planner_mode='local',
                   requested_planner_completed=False, model_available=False, new_api_requests=0,
                   known_course_cost_rmb=0, formal_test_accessed=False, automatic_retries=False)
    try:
        caps = capabilities_for(spec, train_rows=len(task.y_train))
        logger.write_json('run_manifest.json', dict(entrypoint='superagent run', adapter_version=ADAPTER_VERSION,
            scheduler='ab-v1', task_contract=spec, task_archive_sha256=sha(task_path), source_files=frozen,
            seed=args.seed, split_seed_offsets=[0, 101, 202], capabilities=caps, fit_limit=limit,
            llm_request_limit=0, time_limit_seconds=args.time_limit, trial_timeout_seconds=args.trial_timeout,
            python=platform.python_version(), dependencies={k:importlib.metadata.version(k)
            for k in ('numpy','scipy','scikit-learn','joblib')},
            provided_validation_role='score only after final model freeze'))
        if not caps:
            raise ValueError('No registered executable method supports this task/semantics/dependencies')
        trusted, allowed = freeze_partitions(task, directory, args.seed, metric)
        executor = Executor(directory, caps, metric, began, args.time_limit, args.trial_timeout, limit, logger)
        budget = dict(fit_limit=limit, attempted_classifier_fits=0, completed_classifier_fits=0,
                      remaining_fit_allowance=limit, reserved_final_refit=1, unknown_failed_attempts=0,
                      llm_request_limit=0, used_llm_requests=0, time_limit_seconds=args.time_limit)
        while True:
            budget['elapsed_seconds'] = time.monotonic() - began
            state = dict(ab_schema_version=1, task=spec, history=history,
                         available_candidates=untried_candidates(caps, history), prediction_evidence=evidence,
                         budget=budget, registered_confirmation_splits=list(CONFIRMATIONS))
            number = len(history) + 1
            logger.write_json(f'state_{number}.json', state)
            action = next_action(state, capabilities=caps, trusted_partitions=trusted)
            actions.append(action)
            logger.write_json('actions.json', actions)

            def execute(plan):
                if plan['action']['action'] == 'stop':
                    return dict(status='completed', classifier_fits=0)
                split = plan['split']['id']
                payload = directory / (split + '.joblib')
                if sha(payload) != plan['split']['evaluation_payload_sha256']:
                    raise ValueError('Registered evaluation payload changed')
                _, facts = executor.worker(payload, plan['candidate']['id'], directory / f'trial_{number}.joblib',
                                           args.trial_timeout, reserve=budget['reserved_final_refit'])
                return facts

            observed = dispatch_action(state, action, caps, allowed, execute, logger)
            budget = observed['budget']
            actual = executor.accounting()
            if any(actual[k] != budget[k] for k in actual):
                raise ValueError('Dispatcher ledger differs from actual worker fits')
            if action['action'] == 'stop':
                break
            name, split = action['candidate_id'], action['split_id']
            facts = observed['observation']
            history.append(dict(trial=number, candidate=name, params=caps[name]['params'], split_id=split, **facts))
            raw = joblib.load(directory / f'trial_{number}.joblib')['prediction_record']
            evidence.append(dict(trial=number, candidate_id=name, split_id=split, row_id_scope=ROW_SCOPE,
                row_ids=trusted[split]['row_ids'], predictions=raw['predictions'].tolist(),
                **{k:raw[k] for k in ('prediction_sha256','label_sha256','evaluation_payload_sha256')}))
            logger.write_json('history.json', history)
            logger.write_json('prediction_evidence.json', evidence)
        # Recheck every completed observation against locally held labels before selection.
        chosen, decision, result = finish(task, history, state, trusted, allowed, caps, directory, executor, logger)
        outcome.update(selected_candidate=chosen, primary_metric=metric.name, model_available=True,
            full_refit_completed=result['refit_succeeded'], confirmation=decision,
            validation_score=result.get('official_validation_score'),
            requested_planner_completed=result['refit_succeeded'] and not result.get('validation_failure'),
            finalization=result)
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error), status='failed')
        logger.event('run_stopped', error_type=type(error).__name__, message=str(error), automatic_retry=False)
    outcome.update(elapsed_seconds=time.monotonic()-began, source_unchanged=sources()==frozen,
                   action_counts=dict(Counter(a['action'] for a in actions)),
                   classifier_fit_accounting=executor.accounting() if executor else dict(
                       attempted_classifier_fits=0, completed_classifier_fits=0, unknown_failed_attempts=0))
    outcome['model_available'] = (directory / 'final_selection.json').exists()
    outcome['requested_planner_completed'] &= outcome['source_unchanged']
    outcome['status'] = 'completed' if outcome['requested_planner_completed'] else 'failed'
    logger.write_json('run_outcome.json', outcome)
    logger.write_json('summary.json', outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    return 0 if outcome['requested_planner_completed'] else 2
