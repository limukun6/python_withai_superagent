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


def registered_capabilities(task_spec, train_rows, scope='full'):
    """A-owned study projection of B's registry; never constructs new methods."""
    if scope not in ('baseline','full'):
        raise ValueError('Unknown candidate catalogue scope')
    caps=capabilities_for(task_spec,train_rows=train_rows)
    if scope=='baseline':
        if any(c.get('origin') not in ('shared_baseline','b_port') for c in caps.values()):
            raise ValueError('Cannot define baseline from unknown candidate origins')
        caps={name:cap for name,cap in caps.items() if cap['origin']=='shared_baseline'}
    return caps


def validate_options(args):
    if getattr(args,'parameter_semantics','off')!='off' and args.planner!='checked-history':
        raise ValueError('Parameter semantics diagnostics require checked-history planning')
    if args.planner not in ('local','raw-history','checked-history'):
        raise ValueError('AB supports local, raw-history or checked-history planning')
    if args.planner=='local' and args.history_policy!='strict':
        raise ValueError('Historical usage policy applies only to explicit Flash research')
    local_text_mode=getattr(args,'local_text_proposals','off')
    if local_text_mode not in ('off','evidence'):
        raise ValueError('Unknown local text proposal mode')
    if local_text_mode!='off' and (args.planner!='local' or args.scheduler!='ab-v1' or args.catalog_scope!='full'):
        raise ValueError('Evidence text proposals require local ab-v1 and the full shared catalogue')
    if getattr(args,'candidate_coverage','off') not in ('off','routes'):
        raise ValueError('Unknown candidate coverage mode')
    if args.planner=='local' and getattr(args,'candidate_coverage','off')!='off':
        raise ValueError('Candidate coverage projection applies only to the explicit Flash menu')
    if args.planner=='local' and getattr(args,'proposal_contract','strict-v1')!='strict-v1':
        raise ValueError('Proposal contract applies only to an explicit Flash planner')
    forbidden = ('policy','trials')
    if args.planner == 'local':
        forbidden += ('api_url','api_key_file','model','budget_ledger','task_budget_id','llm_budget_rmb')
    if any(getattr(args, k) is not None for k in forbidden):
        raise ValueError('AB uses its shared scheduler and fit-limit; incompatible API/local/fixed-trials options')
    if (args.portfolio != 'ab-v1' or args.architecture != 'single' or args.anchor != 'catalog'
            or args.refinement != 'off' or args.allocation != 'off' or args.feature_cache != 'off'
            or args.time_policy != 'shared' or args.error_feedback):
        raise ValueError('AB-v1 requires its own portfolio, catalog anchor, single architecture and default execution options')
    if args.run_dir and args.output_dir:
        raise ValueError('Use either run-dir or output-dir')
    limit = 11 if args.fit_limit is None else args.fit_limit
    if type(limit) is not int or not 2 <= limit <= 11:
        raise ValueError('AB fit-limit must be an integer in [2,11], including final refit')
    if args.scheduler == 'ab-fixed' and limit != 11:
        raise ValueError('AB fixed control requires fit-limit 11')
    if args.scheduler == 'ab-finalists' and limit < 5:
        raise ValueError('AB matched finalists require at least 5 fits including refit')
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


def finish(task, history, state, trusted, allowed, caps, directory, executor, logger, *, matched_finalists=False):
    if matched_finalists:
        from .ab_finalists import finalist_review
        review = finalist_review(state, capabilities=caps, trusted_partitions=trusted)
    else:
        review = selection_evidence(state, trusted_partitions=trusted)
    chosen = review['incumbent']
    if chosen is None:
        raise ValueError('No verified completed candidate to freeze')
    primary = [r for r in history if r['split_id'] == PRIMARY]
    lookup = {(r['candidate'], r['split_id']): r for r in history}
    last = review['promotions'][-1] if review['promotions'] else None
    names = review['finalists'] if matched_finalists else [last['previous'], chosen] if last else [chosen]
    basis = review['basis']
    measurements = {name: [dict(seed=allowed[s]['seed'], **lookup[name, s]) for s in basis] for name in names}
    decision = decide([lookup[n, PRIMARY] for n in names], measurements,
                      [allowed[s]['seed'] for s in basis], 'shared_text_for_pairs_exact_rows_otherwise')
    if not decision['valid_confirmation'] or decision['selected_candidate'] != chosen:
        raise ValueError('A final selection disagrees with B matched evidence')
    decision.update(b_scheduler_review=review,
        selection_rule=review['selection_rule'] if matched_finalists else
            'Best of initial three; replace on two paired wins and mean gain >= 0.002',
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
    client = account = picker = None
    history_mode = args.planner in ('raw-history','checked-history')
    accounting = None
    if history_mode:
        from .ab_flash import preflight
        ledger, allowance, accounting = preflight(args)  # Before data, credentials, or a request.
    task_path = Path(args.task_package).expanduser().resolve()
    task = load_development_task(task_path)
    spec = TaskSpec.from_task(task).record()
    metric = MetricSpec.for_task(task)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory = Path(args.run_dir or Path(args.output_dir or 'runs') / ('ab_' + stamp)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(directory)
    logger.initialize_usage_ledger()  # Per-run zero-call log, never a replacement for the project API ledger.
    began = time.monotonic()
    frozen = sources()
    training_profile_value=None
    if getattr(args,'task_profile','off')=='train':
        from .task_profile import training_facts
        training_profile_value=training_facts(task)
        logger.write_json('training_profile.json',training_profile_value)
    if history_mode:
        from .ab_flash import FlashPicker
        from .experiment_llm import ExperimentLLM
        from .task_budget import TaskBudgetAccount
        if args.history_policy=='incremental-research':
            from .research_budget import IncrementalResearchAccount
            account=IncrementalResearchAccount(ledger,args.task_budget_id,str(directory))
        else:
            account = TaskBudgetAccount(ledger,args.task_budget_id,str(directory))
        client = ExperimentLLM(args.api_url,args.api_key_file,args.model,allowance,max_requests=3,
            allowed_served_models={args.model},account=account,streaming=True)
        client.set_deadline(began+args.time_limit)
        picker = FlashPicker(client,logger,args.planner=='checked-history',getattr(args,'proposal_contract','strict-v1'))
    history, evidence, actions = [], [], []
    executor = None
    outcome = dict(run_dir=str(directory), scheduler=args.scheduler, planner_mode=args.planner,
                   local_text_proposals=getattr(args,'local_text_proposals','off'),
                   requested_planner_completed=False, model_available=False, new_api_requests=0,
                   known_course_cost_rmb=0, formal_test_accessed=False, automatic_retries=False)
    try:
        caps = registered_capabilities(spec,len(task.y_train),args.catalog_scope)
        logger.write_json('run_manifest.json', dict(entrypoint='superagent run', adapter_version=ADAPTER_VERSION,
            scheduler=args.scheduler, catalog_scope=args.catalog_scope,
            preference_catalog=getattr(args,'preference_catalog','standard'),
            parameter_semantics=getattr(args,'parameter_semantics','off'),
            task_contract=spec, task_archive_sha256=sha(task_path), source_files=frozen,
            task_description_source=task.spec.get('task_description_source'),
            seed=args.seed, split_seed_offsets=[0, 101, 202], capabilities=caps, fit_limit=limit,
            llm_request_limit=3 if history_mode else 0, planner_mode=args.planner,
            history_policy=args.history_policy,
            candidate_coverage=getattr(args,'candidate_coverage','off'),
            task_profile=getattr(args,'task_profile','off'),proposal_contract=getattr(args,'proposal_contract','strict-v1'),
            partial_delivery=getattr(args,'partial_delivery','off'),
            local_text_proposals=getattr(args,'local_text_proposals','off'),
            decision_context_version='ab-history-v1' if history_mode else None,
            planner_failure_policy='stop_without_retry_or_fallback', task_budget_before_run=accounting,
            project_usage_before_run=ledger.project_summary() if history_mode else None,
            incremental_run_allowance_rmb=allowance if history_mode else 0,
            time_limit_seconds=args.time_limit, trial_timeout_seconds=args.trial_timeout,
            python=platform.python_version(), dependencies={k:importlib.metadata.version(k)
            for k in ('numpy','scipy','scikit-learn','joblib')},
            provided_validation_role='score only after final model freeze'))
        if not caps:
            raise ValueError('No registered executable method supports this task/semantics/dependencies')
        trusted, allowed = freeze_partitions(task, directory, args.seed, metric)
        executor = Executor(directory, caps, metric, began, args.time_limit, args.trial_timeout, limit, logger)
        budget = dict(fit_limit=limit, attempted_classifier_fits=0, completed_classifier_fits=0,
                      remaining_fit_allowance=limit, reserved_final_refit=1, unknown_failed_attempts=0,
                      llm_request_limit=3 if history_mode else 0, used_llm_requests=0, time_limit_seconds=args.time_limit)
        while True:
            budget['elapsed_seconds'] = time.monotonic() - began
            state = dict(ab_schema_version=1, task=spec, history=history,
                         available_candidates=untried_candidates(caps, history), prediction_evidence=evidence,
                         budget=budget, registered_confirmation_splits=list(CONFIRMATIONS))
            if training_profile_value is not None:state['training_profile']=training_profile_value
            number = len(history) + 1
            logger.write_json(f'state_{number}.json', state)
            if picker:
                picker.last_proposal = None

            def choose(ids):
                if getattr(args,'candidate_coverage','off')=='routes':
                    from .candidate_coverage import project_routes
                    ids,coverage=project_routes(state,caps,ids)
                    logger.event('candidate_coverage_projection',trial=number,**coverage)
                try:
                    return picker.choose(state,caps,ids)
                finally:
                    budget['used_llm_requests'] = client.request_count

            local_recipe_proposals=[]
            def compose(eligible, verified, capabilities):
                from .ab_text_recipes import propose
                proof=propose(eligible,verified,capabilities)
                if proof is not None:
                    local_recipe_proposals.append(proof)
                    logger.event('local_text_recipe_proposal',trial=number,proposal=proof)
                return proof

            if args.scheduler == 'ab-finalists':
                from .ab_finalists import finalist_action
                action = finalist_action(state, capabilities=caps, trusted_partitions=trusted,
                                         on_explore=choose if picker else None)
            elif args.scheduler == 'ab-fixed':
                from .ab_fixed_schedule import fixed_action
                action = fixed_action(state, capabilities=caps, trusted_partitions=trusted,
                                      on_explore=choose if picker else None)
            else:
                optional_local=({'on_local_recipe':compose} if getattr(args,'local_text_proposals','off')=='evidence' else {})
                action = next_action(state, capabilities=caps, trusted_partitions=trusted,
                                     on_explore=choose if picker else None,**optional_local)
            proposal = picker.last_proposal.raw_proposal if picker and picker.last_proposal else None
            if proposal is not None:
                action['reason'] = proposal['reason']
                action['evidence_trials'] = list(dict.fromkeys([*action['evidence_trials'],*proposal['evidence_trials']]))
            logger.event('planner_routing', trial=number, action=action['action'],
                source='local_text_evidence' if local_recipe_proposals else 'flash' if proposal else 'local_after_request_cap' if picker and client.request_count>=3
                    and action['action']=='explore' else 'shared_local_rule',
                used_llm_requests=client.request_count if client else 0)
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
            raw = joblib.load(directory / f'trial_{number}.joblib')['prediction_record']
            row = dict(trial=number, candidate=name, params=caps[name]['params'], split_id=split,
                       proposal=proposal, **facts,
                       evaluation_payload_sha256=raw['evaluation_payload_sha256'],label_sha256=raw['label_sha256'],
                       row_ids_sha256=fingerprint(trusted[split]['row_ids']))
            from .hypothesis_agent import check_expectation
            from .estimator_facts import verify_configuration_claim
            from .explanation_audit import audit_explanation
            row['metric_identity'] = metric.check_identity(facts.get('metric_spec'))
            row['configuration_fact_check'] = verify_configuration_claim(proposal,facts['estimator'])
            row['hypothesis_check'] = check_expectation(proposal,row,history)
            row['explanation_audit'] = audit_explanation(proposal,history,{'id':name,'params':caps[name]['params']},
                parameter_semantics=getattr(args,'parameter_semantics','off')=='diagnostic',
                selected_facts={'classifier':caps[name]['estimator'],'source':'shared_ab_catalogue'}) if proposal else None
            history.append(row)
            evidence.append(dict(trial=number, candidate_id=name, split_id=split, row_id_scope=ROW_SCOPE,
                row_ids=trusted[split]['row_ids'], predictions=raw['predictions'].tolist(),
                **{k:raw[k] for k in ('prediction_sha256','label_sha256','evaluation_payload_sha256')},
                **({'score_record':raw['score_record']} if metric.requires_scores else {})))
            logger.write_json('history.json', history)
            logger.write_json('prediction_evidence.json', evidence)
        # Recheck every completed observation against locally held labels before selection.
        chosen, decision, result = finish(task, history, state, trusted, allowed, caps, directory, executor, logger,
                                          matched_finalists=args.scheduler == 'ab-finalists')
        outcome.update(selected_candidate=chosen, primary_metric=metric.name, model_available=True,
            full_refit_completed=result['refit_succeeded'], confirmation=decision,
            validation_score=result.get('official_validation_score'),
            requested_planner_completed=result['refit_succeeded'] and not result.get('validation_failure'),
            finalization=result)
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error=str(error), status='failed')
        logger.event('run_stopped', error_type=type(error).__name__, message=str(error), automatic_retry=False)
        if getattr(args,'partial_delivery','off')=='checkpoint' and history and not (directory/'final_selection.json').exists():
            try:
                from .partial_delivery import freeze_checkpoint
                recovery_state=dict(state,history=history,prediction_evidence=evidence)
                recovered=freeze_checkpoint(task,recovery_state,trusted,caps,directory,logger,str(error))
                outcome.update(selected_candidate=recovered['candidate'],primary_metric=metric.name,full_refit_completed=False,
                    partial_recovery=recovered)
            except Exception as recovery_error:
                outcome['partial_recovery_failure']=dict(error_type=type(recovery_error).__name__,error=str(recovery_error))
                logger.event('partial_delivery_failed',**outcome['partial_recovery_failure'])
    outcome.update(elapsed_seconds=time.monotonic()-began, source_unchanged=sources()==frozen,
                   action_counts=dict(Counter(a['action'] for a in actions)),
                   classifier_fit_accounting=executor.accounting() if executor else dict(
                       attempted_classifier_fits=0, completed_classifier_fits=0, unknown_failed_attempts=0))
    usage = [json.loads(line) for line in logger.usage_path.read_text(encoding='utf-8').splitlines()]
    unresolved = sum(r.get('usage_known') is False for r in usage)
    outcome.update(new_api_requests=client.request_count if client else 0,
        known_course_cost_rmb=client.spent if client else 0,
        complete_course_cost_rmb=None if unresolved else client.spent if client else 0,
        unresolved_calls=unresolved, task_budget_after_run=account.snapshot() if account else None,
        partial_trial_models=[p.name for p in sorted(directory.glob('trial_*.joblib'))])
    outcome.update(history_policy=args.history_policy,
        candidate_coverage=getattr(args,'candidate_coverage','off'),
        course_budget_compliance='unverified_history' if args.history_policy=='incremental-research' else
            'verified_local_ledger' if history_mode else 'no_new_llm_calls',
        valid_for_budgeted_final_report=False if args.history_policy=='incremental-research' else None,
        project_usage_after_run=ledger.project_summary() if history_mode else None)
    outcome['complete_project_course_cost_rmb']=ledger.project_summary()['complete_course_cost_rmb'] if history_mode else None
    outcome['model_available'] = (directory / 'final_selection.json').exists()
    outcome['requested_planner_completed'] &= outcome['source_unchanged']
    outcome['status'] = 'completed' if outcome['requested_planner_completed'] else 'failed'
    outcome['delivery_status']='complete' if outcome['requested_planner_completed'] else 'partial' if outcome['model_available'] else 'unavailable'
    if getattr(args,'partial_delivery','off')=='checkpoint' and outcome['model_available'] and not outcome['requested_planner_completed']:
        selection=json.loads((directory/'final_selection.json').read_text(encoding='utf-8'))
        selection.update(delivery_status='partial',requested_planner_completed=False,full_training_rows=len(task.y_train))
        logger.write_json('final_selection.json',selection)
    logger.write_json('run_outcome.json', outcome)
    logger.write_json('summary.json', outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    return 0 if outcome['requested_planner_completed'] else 2
