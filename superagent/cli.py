"""Unified train-internal execution with explicit local and bounded LLM modes."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from urllib.parse import urlsplit,urlunsplit

from . import __version__
from .benchmark import run_one
from .data import load_development_task
from .experiment_llm import ExperimentLLM
from .models import search_candidates
from .predict import create_predictions
from .pricing import resolve_pricing
from .task_budget import TaskBudgetLedger,TaskBudgetAccount
from .contracts import TaskSpec
from .decision_architecture import ARCHITECTURES,calls_per_decision

CAMPUS_CHAT_ENDPOINT = 'https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions'
CAMPUS_FLASH_MODEL = 'DeepSeek-V4.1-Flash'
HISTORY_PLANNERS = ('raw-history','checked-history')


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(args):
    if args.trials is None:
        args.trials = 5
    calls_per_decision(args.architecture)
    if args.planner in HISTORY_PLANNERS:
        if (args.architecture!='single' or args.refinement!='off' or args.allocation!='off'
                or args.anchor!='catalog' or args.portfolio!='expanded'):
            raise ValueError('History planners require single architecture, expanded catalog anchor, refinement/allocation off')
        args.api_url=args.api_url or CAMPUS_CHAT_ENDPOINT
        args.model=args.model or CAMPUS_FLASH_MODEL
    if args.planner == 'scalar-feedback':
        if (args.architecture != 'single' or args.refinement != 'off' or args.error_feedback
                or args.portfolio != 'expanded' or args.anchor != 'catalog'):
            raise ValueError('Scalar feedback uses the catalog anchor, expanded portfolio and score-only feedback')
        args.api_url = args.api_url or CAMPUS_CHAT_ENDPOINT
        args.model = args.model or CAMPUS_FLASH_MODEL
        if args.api_url != CAMPUS_CHAT_ENDPOINT or args.model != CAMPUS_FLASH_MODEL:
            raise ValueError(f'Scalar feedback requires the campus endpoint and {CAMPUS_FLASH_MODEL}')
    if args.anchor=='schema' and (args.allocation!='off' or args.portfolio=='original'):
        raise ValueError('Schema anchor needs an expanded catalog and allocation off')
    if args.refinement=='research' and (args.planner!='llm' or args.architecture!='single' or args.portfolio not in ('original','expanded') or args.time_policy!='shared' or args.trials<4):
        raise ValueError('Research requires single LLM, original/expanded, shared time and at least4trials')
    if type(args.trials) is not int or args.trials<1:
        raise ValueError('trials must be a positive integer')
    if any(not math.isfinite(v) or v<=0 for v in (args.time_limit,args.trial_timeout)):
        raise ValueError('Time limits must be positive and finite')
    if args.run_dir and args.output_dir:
        raise ValueError('Use either run-dir or output-dir')
    if args.planner=='local':
        if args.architecture!='single':
            raise ValueError('Reviewed architectures require explicit --planner llm')
        if any(value is not None for value in (args.api_url,args.api_key_file,args.model,args.llm_budget_rmb,args.budget_ledger,args.task_budget_id)):
            raise ValueError('API options require explicit --planner llm; local mode never loads credentials')
        if args.refinement in ('midpoints','compose') and (args.policy not in ('greedy_routes','protected_routes','interleaved_routes') or args.portfolio not in ('original','expanded')):
            raise ValueError('Refinement needs a feedback local policy and original/expanded portfolio')
        return None
    if args.refinement in ('midpoints','compose') and args.portfolio not in ('original','expanded'):
        raise ValueError('Refinement supports original/expanded portfolios only')
    if os.environ.get('SUPERAGENT_OFFLINE')=='1':
        raise ValueError('LLM mode is disabled in the current offline environment')
    if args.policy is not None:raise ValueError('--policy selects local rules; LLM mode uses budget_routes')
    if args.trials<2:raise ValueError('LLM mode needs at least one decision after the shared anchor')
    if not args.api_url or not args.api_key_file or not args.model:
        raise ValueError('LLM mode requires api-url, api-key-file and model')
    if not args.budget_ledger or not args.task_budget_id:
        raise ValueError('LLM mode requires budget-ledger and a stable task-budget-id shared by all runs of that task')
    endpoint=urlsplit(args.api_url)
    if endpoint.scheme!='https' or not endpoint.netloc or endpoint.username or endpoint.password:
        raise ValueError('API endpoint must be HTTPS with a host and no embedded credentials')
    budget=.10 if args.llm_budget_rmb is None else args.llm_budget_rmb
    if not math.isfinite(budget) or not 0<budget<=20:
        raise ValueError('The incremental run allowance must be finite and in (0,20]')
    pricing=resolve_pricing(args.model)
    TaskBudgetLedger(args.budget_ledger).preflight(args.task_budget_id,budget)
    return pricing


def _run(args):
    catalog = getattr(args, 'preference_catalog', 'standard')
    if getattr(args,'proposal_contract','strict-v1')=='grounded-v1' and args.planner!='checked-history':
        raise ValueError('Grounded proposals require checked-history planning')
    if getattr(args,'issue_layering','off')!='off' and getattr(args,'proposal_contract','strict-v1')!='grounded-v1':
        raise ValueError('Issue layering only applies to the grounded-v1 proposal contract')
    if getattr(args,'feedback_view','full')!='full' and (args.planner!='checked-history' or
            args.scheduler not in ('ab-v1','ab-fixed','ab-finalists','ab-complement','ab-adaptive-complement')):
        raise ValueError('Compact feedback requires the shared checked-history executor')
    if getattr(args,'numeric_kernel',False) and args.scheduler not in ('ab-v1','ab-fixed','ab-finalists'):
        raise ValueError('The numeric kernel candidate requires the shared A/B executor')
    if getattr(args, 'confidence_cascade', False) and args.scheduler not in ('ab-complement', 'ab-adaptive-complement'):
        raise ValueError('Confidence routing requires the matched preference scheduler')
    if getattr(args, 'resolve_single_conflict', False) and args.scheduler != 'ab-adaptive-complement':
        raise ValueError('Single-model conflict resolution requires the adaptive complement scheduler')
    if getattr(args,'parameter_semantics','off')!='off' and args.scheduler not in ('ab-v1','ab-fixed','ab-finalists','ab-complement','ab-adaptive-complement'):
        raise ValueError('Parameter semantics diagnostics require the shared A/B executor')
    if getattr(args, 'cover_untried_encoder', False) and args.scheduler != 'ab-adaptive-complement':
        raise ValueError('Encoder coverage applies to the adaptive complement scheduler')
    if args.scheduler in ('ab-v1','ab-fixed','ab-finalists','ab-complement','ab-adaptive-complement'):
        from .ab_methods import PREFERENCE_CATALOG_ENV
        from .ab_run import command
        previous = os.environ.get(PREFERENCE_CATALOG_ENV)
        os.environ[PREFERENCE_CATALOG_ENV] = catalog
        try:
            return command(args)
        finally:
            if previous is None:
                os.environ.pop(PREFERENCE_CATALOG_ENV, None)
            else:
                os.environ[PREFERENCE_CATALOG_ENV] = previous
    if catalog != 'standard':
        raise ValueError('Expanded preference catalog requires the shared A/B executor')
    if (args.fit_limit is not None or args.portfolio == 'ab-v1' or args.catalog_scope!='full' or args.history_policy!='strict'
            or args.candidate_coverage!='off' or args.task_profile!='off' or args.proposal_contract!='strict-v1' or args.partial_delivery!='off'
            or args.local_text_proposals!='off'):
        raise ValueError('AB fit-limit/portfolio require --scheduler ab-v1')
    if args.trials is None:
        args.trials = 5
    pricing=_validate(args)
    if args.allocation != "off" and (args.planner != "local" or args.refinement != "off"):
        raise ValueError("Resource allocation requires local mode and refinement off")
    task_path=Path(args.task_package).expanduser().resolve()
    task=load_development_task(task_path)
    task_contract=TaskSpec.from_task(task)
    if args.trials>len(search_candidates(task.kind,args.portfolio)):
        raise ValueError('Requested trials exceed the declared candidate catalog')
    if args.run_dir:
        directory=Path(args.run_dir).expanduser().resolve()
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        directory=(Path(args.output_dir or 'runs')/f'{task.kind}_{stamp}').expanduser().resolve()
    if directory.exists():raise FileExistsError('Run directory already exists; do not overwrite previous experiments')
    source=Path(__file__).resolve().parent
    source_files={p.relative_to(source).as_posix():_sha(p) for p in sorted(source.rglob('*.py'))}
    source_hash=hashlib.sha256(json.dumps(source_files,sort_keys=True).encode()).hexdigest()
    local_policy=args.policy or 'balanced_blind'
    scalar_feedback = args.planner == 'scalar-feedback'
    history_feedback = args.planner in HISTORY_PLANNERS
    strategy = {'local': 'local_' + local_policy, 'llm': 'budget_routes',
                'scalar-feedback': 'scalar_feedback','raw-history':'local_interleaved_routes',
                'checked-history':'local_interleaved_routes'}[args.planner]
    budget=0. if pricing is None else (.10 if args.llm_budget_rmb is None else args.llm_budget_rmb)
    maximum_requests=((args.trials-1)*calls_per_decision(args.architecture)+(2 if args.refinement=='research' else 0)) if pricing else 0
    endpoint=urlsplit(args.api_url) if args.api_url else None
    manifest={'agent_version':__version__,'entrypoint':'superagent run (experiment is an alias)',
        'invoked_as':args.command,'planner_mode':args.planner,'strategy':strategy,'decision_architecture':args.architecture,
        'seed':args.seed,'model_training_seed':42,'primary_trials_including_anchor':args.trials,
        'portfolio':args.portfolio,'anchor_policy':args.anchor,'search_grouping':'shared_text_for_pairs_exact_features_otherwise',
        'confirmation_repeats':3,'confirmation_top_k':3,'full_train_refit':True,
        'time_limit_seconds':args.time_limit,'trial_timeout_seconds':args.trial_timeout,
        'time_policy':args.time_policy,
        'feature_cache':args.feature_cache,
        'refinement':args.refinement,'error_feedback':args.error_feedback,'allocation':args.allocation,
        'incremental_run_allowance_rmb':budget,'maximum_api_requests':maximum_requests,
        'calls_per_adaptive_decision_maximum':calls_per_decision(args.architecture) if pricing else 0,
        'maximum_research_model_requests':2 if args.refinement=='research' else 0,
        'review_failure_policy':'retain_valid_original_for_current_trial_disable_future_architecture_calls_mark_impure',
        'planner_failure_policy':'stop_without_retry_or_fallback' if history_feedback else 'existing_planner_policy',
        'history_feedback':{'enabled':history_feedback,'checks_visible':args.planner=='checked-history',
                            'context_version':'history-evidence-v3' if history_feedback else None},
        'model':args.model if pricing else None,'pricing':pricing.metadata() if pricing else None,
        'api_endpoint_origin_and_path':urlunsplit((endpoint.scheme,endpoint.netloc,endpoint.path,'','')) if endpoint else None,
        'credential_contents_logged':False,'automatic_api_retries':False,
        'source_sha256':source_hash,'source_files':source_files,'task_archive_sha256':_sha(task_path),
        'task_contract':task_contract.record(),'primary_metric':task_contract.metric.name,'input_schema':task.spec.get('input_schema'),'schema_origin':task.spec.get('schema_origin','legacy_adapter'),
        'selection_data':'training-derived partitions only','provided_validation_role':'score after selection/model freeze',
        'formal_test_accessed':False,'python':platform.python_version(),
        'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','joblib','catboost','requests','Pillow')}}
    client=None
    account=None
    if pricing:
        ledger=TaskBudgetLedger(args.budget_ledger)
        manifest['task_budget_before_run']=ledger.preflight(args.task_budget_id,budget)
        manifest['budget_ledger_path']=str(ledger.path)
        account=TaskBudgetAccount(ledger,args.task_budget_id,str(directory))
        allowed={args.model,pricing.model_id,pricing.official_model_version}
        if scalar_feedback or history_feedback:
            client=ExperimentLLM(args.api_url,args.api_key_file,args.model,budget,
                max_requests=maximum_requests,allowed_served_models=allowed,account=account,streaming=True)
        else:
            client=ExperimentLLM(args.api_url,args.api_key_file,args.model,budget,reasoning_effort='none',
                disable_template_thinking=True,max_requests=maximum_requests,allowed_served_models=allowed,account=account)
    if args.allocation=='off':execute=run_one
    else:
        from functools import partial
        from .resource_allocation import run_allocated
        execute=partial(run_allocated,mode=args.allocation)
    execution_options={}
    if history_feedback:
        from .checked_history_policy import CheckedHistoryPolicy
        from .events import RunLogger
        execution_options['decision_policy']=CheckedHistoryPolicy(client,RunLogger(directory),args.planner=='checked-history')
    try:
        summary=execute(task,strategy,args.seed,directory,client,max_trials=args.trials,
        time_limit=args.time_limit,trial_timeout=args.trial_timeout,anchor_first=True,anchor_policy=args.anchor,refit_full_train=True,
        confirmation_repeats=3,confirmation_top_k=3,confirmation_grouping='shared_text',
        search_grouping='shared_text',portfolio=args.portfolio,run_metadata=manifest,time_policy=args.time_policy,
        feature_cache=args.feature_cache=='run',refinement=args.refinement,decision_architecture=args.architecture,error_feedback=args.error_feedback,
        **execution_options)
    except Exception as error:
        if history_feedback and directory.is_dir():
            _record_stopped_history_run(directory,args,client,account,error)
        raise
    current={p.relative_to(source).as_posix():_sha(p) for p in sorted(source.rglob('*.py'))}
    unchanged=current==source_files
    ledger=[json.loads(line) for line in (directory/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
    unknown=sum(row.get('usage_known') is False for row in ledger)
    model_available=bool(summary['selected_candidate'])
    task_budget_after=account.snapshot() if account is not None else None
    planned_primary=summary.get('resource_allocation',{}).get('final_primary_trials_planned',args.trials)
    primary_completed=summary['trials_attempted']==planned_primary
    confirmation_completed=summary['confirmation'].get('valid_confirmation',False)
    refit_completed=summary['refit_succeeded']
    validation_available=task.X_val is not None and task.y_val is not None
    evaluation_completed=not validation_available or summary.get('official_validation_score') is not None
    planner_completed=(model_available and summary['valid_policy_run'] and summary['usage_complete'] and unchanged
        and primary_completed and confirmation_completed and refit_completed and evaluation_completed
        and summary.get('prediction_evidence_audit',{}).get('integrity_valid',True)
        and (task_budget_after is None or (task_budget_after['accounting_complete'] and not task_budget_after['limit_exceeded'])))
    outcome={'run_dir':str(directory),'task_kind':task.kind,'planner_mode':args.planner,
        'decision_architecture':summary.get('decision_architecture'),
        'review_fallback_count':summary.get('decision_architecture',{}).get('review_fallback_count',0),
        'model_available':model_available,'requested_planner_completed':planner_completed,
        'selected_candidate':summary['selected_candidate'],'primary_metric':task_contract.metric.name,
        'validation_score':summary.get('official_validation_score',summary['official_validation_accuracy'] if task_contract.metric.name=='accuracy' else None),
        'validation_accuracy':summary['official_validation_accuracy'],
        'provided_validation_available':validation_available,'evaluation_completed':evaluation_completed,
        'validation_failure':summary.get('validation_failure'),
        'primary_trials_attempted':summary['trials_attempted'],'fallback_count':summary['fallback_count'],
        'primary_horizon_completed':primary_completed,'confirmation_completed':confirmation_completed,
        'full_refit_completed':refit_completed,
        'runtime_budget':summary.get('runtime_budget'),
        'feature_cache':summary.get('feature_cache'),
        'classifier_fit_accounting':summary.get('classifier_fit_accounting'),
        'prediction_evidence_audit':summary.get('prediction_evidence_audit'),
        'refinement':summary.get('refinement'),'resource_allocation':summary.get('resource_allocation'),'error_feedback':args.error_feedback,
        'new_api_requests':client.request_count if client else 0,
        'metered_responses':sum(row['event']=='llm_call' for row in ledger),
        'known_course_cost_rmb':sum(row.get('course_cost_rmb') or 0. for row in ledger),
        'complete_course_cost_rmb':summary['course_cost_rmb'],'unresolved_calls':unknown,
        'actual_account_debit_rmb':None if pricing else 0.,'source_unchanged':unchanged,'formal_test_accessed':False,
        'task_budget_after_run':task_budget_after,
        'scope':'Run costs shown separately from the persistent task snapshot; shared cost is counted once in project totals'}
    (directory/'run_outcome.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(outcome,ensure_ascii=False))
    if not model_available:return 1
    return 0 if planner_completed else 2


def _record_stopped_history_run(directory,args,client,account,error):
    """Persist the partial result without further training, API calls, or random recovery."""
    usage_path=directory/'llm_usage.jsonl'
    usage=[json.loads(line) for line in usage_path.read_text(encoding='utf-8').splitlines()] if usage_path.is_file() else []
    trials_path=directory/'trials.json'
    trials=json.loads(trials_path.read_text(encoding='utf-8')) if trials_path.is_file() else []
    known=sum(r.get('course_cost_rmb') or 0 for r in usage)
    unknown=sum(r.get('usage_known') is False for r in usage)
    budget=account.snapshot() if account is not None else None
    complete=unknown==0 and (budget is None or budget['accounting_complete'])
    outcome={'status':'stopped','planner_mode':args.planner,'requested_planner_completed':False,
        'model_available':False,'final_model_available':False,
        'retained_trial_artifacts':[str(p) for p in sorted(directory.glob('trial_*.joblib'))],
        'primary_trials_attempted':len(trials),'completed_search_fits':sum(r.get('classifier_fits',1) for r in trials if r['status']=='completed'),
        'fallback_count':0,'new_api_requests':client.request_count if client else 0,
        'known_course_cost_rmb':known,'complete_course_cost_rmb':known if complete else None,
        'actual_account_debit_rmb':None,'unresolved_calls':unknown,'task_budget_after_run':budget,
        'error':f'{type(error).__name__}: {error}'.replace(client.key,'[redacted]') if client else str(error),
        'scope':'Stopped run; retained trial models are partial artifacts, not a finalized prediction solution'}
    (directory/'run_outcome.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def _predict(args):
    destination=create_predictions(args.task_package,args.run_dir,args.output)
    print(f'Predictions written to: {destination}')
    print('Prediction uses declared features; no target is used for evaluation.')
    return 0


def _budget_init(args):
    ledger=TaskBudgetLedger.initialize(args.ledger,history=args.import_audit,expected_sha256=args.audit_sha256,new_project=args.new_project)
    print(json.dumps({'ledger':str(ledger.path),**ledger.project_summary()},ensure_ascii=False));return 0


def _budget_status(args):
    ledger=TaskBudgetLedger(args.ledger)
    print(json.dumps(ledger.snapshot(args.task_budget_id) if args.task_budget_id else ledger.project_summary(),ensure_ascii=False));return 0


def _resume_status(args):
    from .resume import inspect as resume_inspection
    report=resume_inspection(args.run_dir,args.ledger)
    print(json.dumps(report,ensure_ascii=False,indent=1))
    return {'complete':0,'resumable':0,'blocked':2}.get(report['status'],2)


def build_parser():
    parser=argparse.ArgumentParser(description='Schema-driven classification SuperAgent')
    commands=parser.add_subparsers(dest='command',required=True)
    from .adaptive_local import command
    adaptive=commands.add_parser('adaptive-local',help='Offline accuracy prototype: explore, replicate, or stop')
    adaptive.add_argument('task_package')
    adaptive.add_argument('--run-dir',required=True)
    adaptive.add_argument('--seed',default=707,type=int)
    adaptive.set_defaults(handler=command)
    run=commands.add_parser('run',aliases=['experiment'],help='Train-internal selection, fixed confirmation, full-train refit')
    run.add_argument('task_package')
    run.add_argument('--run-dir',help='New output directory, never overwritten')
    run.add_argument('--output-dir',help='Parent directory for an automatically named new run')
    run.add_argument('--planner',choices=['local','llm','scalar-feedback',*HISTORY_PLANNERS],default='local',
        help='Use local search, structured LLM planning, or campus Flash with scalar search feedback')
    run.add_argument('--architecture',choices=ARCHITECTURES,default='single',
        help='LLM mode: single proposal, same-planner self-review, or Planner plus Reviewer; shares the run allowance')
    run.add_argument('--policy',choices=['random','balanced_blind','greedy_routes','protected_routes','interleaved_routes'],help='Local mode only; default balanced_blind')
    run.add_argument('--seed',type=int,default=707)
    run.add_argument('--trials',type=int,help='Fixed scheduler primary trials (default 5); incompatible with ab-v1')
    run.add_argument('--scheduler',choices=['fixed','ab-v1','ab-fixed','ab-finalists','ab-complement','ab-adaptive-complement'],default='fixed')
    run.add_argument('--fit-limit',type=int,help='ab-v1 total actual classifier fits including final refit (default 11)')
    run.add_argument('--llm-request-limit',type=int,default=3,
                     help='AB planning-request cap per run (default 3, accepted range 1..12)')
    run.add_argument('--resume-from',help='AB: continue an interrupted run into a new directory')
    run.add_argument('--preference-catalog', choices=['standard','deep','deep-edit','deep-e5','deep-bge','deep-familiarity','deep-e5-familiarity'], default='standard',
                     help='Optional verified native twelve-layer preference candidates in the shared A/B executor')
    run.add_argument('--cover-untried-encoder', action='store_true',
                     help='Permit one affordable additional exploration of an unobserved frozen encoder in adaptive complement mode')
    run.add_argument('--feedback-view',choices=['full','compact'],default='full',
                     help='Optional lossless deduplication of checked evidence; shared checked-history only')
    run.add_argument('--numeric-kernel', action='store_true',
                     help='Offer a bounded random Fourier classifier for numeric tables')
    run.add_argument('--confidence-cascade', action='store_true',
                     help='Use confirmed training feedback to skip a second encoder for confident preference pairs')
    run.add_argument('--resolve-single-conflict', action='store_true',
        help='Compare disagreeing single-model leaders on a third split when the remaining budget permits')
    run.add_argument('--catalog-scope',choices=['full','baseline'],default='full',help='AB catalogue study: baseline excludes B-added methods; full is shared by all C1 arms')
    run.add_argument('--history-policy',choices=['strict','incremental-research'],default='strict',help='Research mode retains imported unknown historical charges; full course budget remains unverified')
    run.add_argument('--candidate-coverage',choices=['off','routes'],default='off',help='AB Flash: prefer unseen feasible method routes in early exploration, without extra calls')
    run.add_argument('--local-text-proposals',choices=['off','evidence'],default='off',help='Local AB-v1 only: propose bounded text recipes from verified observations')
    run.add_argument('--task-profile',choices=['off','train'],default='off',help='AB: share bounded training-only facts with both planners')
    run.add_argument('--proposal-contract',choices=['strict-v1','calibrated-v2','grounded-v1'],default='strict-v1',help='AB Flash: optional calibrated or evidence-bound proposal contract')
    run.add_argument('--menu-view',choices=['full','compact'],default='full',
        help='AB Flash only: send the candidate menu as lossless shared tables; default full')
    run.add_argument('--issue-layering',choices=['off','by-class'],default='off',
        help='grounded-v1 only: close knowledge corrections by a checked restatement instead of waiting for a new fit')
    run.add_argument('--parameter-semantics',choices=['off','diagnostic'],default='off',help='AB checked-history only: optional source-backed C semantics warnings, without action veto')
    run.add_argument('--partial-delivery',choices=['off','checkpoint'],default='off',help='AB: on failure freeze an already fitted checkpoint; the requested run remains failed')
    run.add_argument('--anchor',choices=['catalog','schema'],default='catalog')
    run.add_argument('--portfolio',choices=['original','expanded','research','nonlinear','adaptive','semantic','learned','challenger','ab-v1'],default='expanded')
    run.add_argument('--time-limit',type=float,default=300.)
    run.add_argument('--trial-timeout',type=float,default=60.)
    run.add_argument('--time-policy',choices=['shared','reserve'],default='shared',help='Optional train-history-based time reservation for confirmation and full refit')
    run.add_argument('--feature-cache',choices=['off','run'],default='off',help='Optional same-run reuse of train-fitted paired TF-IDF; classifiers still refitted')
    run.add_argument('--refinement',choices=['off','midpoints','compose','research'],default='off',help='Optional feedback candidates; research adds bounded literature retrieval and checked sklearn code generation')
    run.add_argument('--allocation',choices=['off','uniform','successive'],default='off',help='Experimental local resource study; counts all screening fits')
    run.add_argument('--error-feedback',action='store_true',help='Expose train-thresholded descriptive error slices to feedback planners')
    run.add_argument('--api-url')
    run.add_argument('--api-key-file')
    run.add_argument('--model')
    run.add_argument('--thinking',dest='thinking_mode',choices=['enabled','disabled'],
        help='AB Flash: send an explicit thinking switch when the endpoint honours it')
    run.add_argument('--reasoning-effort',dest='reasoning_effort_level',
        choices=['none','low','medium','high'],help='AB Flash: explicit reasoning budget hint')
    run.add_argument('--budget-ledger',help='Previously initialized SQLite course-cost ledger')
    run.add_argument('--task-budget-id',help='Stable lowercase ID for this evaluation task; reuse it across seeds and iterations')
    run.add_argument('--llm-budget-rmb',type=float,help='Incremental allowance for this run, default 0.10 in LLM mode; aggregate prior task costs separately')
    run.set_defaults(handler=_run)
    predict=commands.add_parser('predict',help='Use a frozen model on declared prediction features')
    predict.add_argument('task_package')
    predict.add_argument('--run-dir',required=True)
    predict.add_argument('--output',required=True)
    predict.set_defaults(handler=_predict)
    initialize=commands.add_parser('budget-init',help='Import verified prior usage or explicitly initialize a new project; never overwrite')
    initialize.add_argument('ledger')
    initialize.add_argument('--import-audit')
    initialize.add_argument('--audit-sha256')
    initialize.add_argument('--new-project',action='store_true')
    initialize.set_defaults(handler=_budget_init)
    status=commands.add_parser('budget-status',help='Read course-cost accounting, not the platform account balance')
    status.add_argument('ledger');status.add_argument('--task-budget-id');status.set_defaults(handler=_budget_status)
    resume=commands.add_parser('resume-status',help='Read-only: decide whether an interrupted run may continue')
    resume.add_argument('run_dir');resume.add_argument('--ledger')
    resume.set_defaults(handler=_resume_status)
    return parser


def main():
    parser=build_parser()
    args=parser.parse_args()
    try:result=args.handler(args)
    except (ValueError,FileExistsError,FileNotFoundError) as error:
        parser.error(str(error))
    else:raise SystemExit(result)
