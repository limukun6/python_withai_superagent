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


def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate(args):
    calls_per_decision(args.architecture)
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
    pretrained = None
    method_choice=getattr(args,'method_choice',None)
    if method_choice:
        if (args.planner!='local' or args.policy not in (None,'interleaved_routes') or args.trials!=7
                or args.refinement!='off' or args.allocation!='off' or args.anchor!='catalog'):
            raise ValueError('Method choice requires seven local interleaved trials and catalog anchor')
        args.policy='interleaved_routes'
    if args.pretrained_config:
        if not method_choice and (args.planner != 'local' or args.policy not in (None,'interleaved_routes') or args.allocation != 'off' or args.refinement != 'off' or args.trials != 5 or args.anchor != 'catalog'):
            raise ValueError('Pretrained choice requires local interleaved routes, allocation/refinement off, catalog anchor and exactly 5 trials')
        args.policy = 'interleaved_routes'
        pretrained = json.loads(Path(args.pretrained_config).read_text())
        if not method_choice and not (pretrained.get('weights') or pretrained.get('tabular_weights')):
            raise ValueError('A MiniLM-only configuration uses --method-choice adaptive (or paired) --trials 7')
    pricing=_validate(args)
    if args.allocation != "off" and (args.planner != "local" or args.refinement != "off"):
        raise ValueError("Resource allocation requires local mode and refinement off")
    task_path=Path(args.task_package).expanduser().resolve()
    task=load_development_task(task_path)
    task_contract=TaskSpec.from_task(task)
    if not method_choice and args.trials>len(search_candidates(task.kind,args.portfolio)):
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
    strategy = {'local': 'local_' + local_policy, 'llm': 'budget_routes',
                'scalar-feedback': 'scalar_feedback'}[args.planner]
    budget=0. if pricing is None else (.10 if args.llm_budget_rmb is None else args.llm_budget_rmb)
    maximum_requests=((args.trials-1)*calls_per_decision(args.architecture)+(2 if args.refinement=='research' else 0)) if pricing else 0
    endpoint=urlsplit(args.api_url) if args.api_url else None
    finalists = 2 if pretrained is not None and not method_choice else 3
    manifest={'agent_version':__version__,'entrypoint':'superagent run (experiment is an alias)',
        'invoked_as':args.command,'planner_mode':args.planner,'strategy':strategy,'decision_architecture':args.architecture,
        'seed':args.seed,'model_training_seed':42,'primary_trials_including_anchor':args.trials,
        'portfolio':args.portfolio,'method_choice':method_choice,'anchor_policy':args.anchor,'search_grouping':'shared_text_for_pairs_exact_features_otherwise',
        'confirmation_repeats':3,'confirmation_top_k':finalists,'full_train_refit':True,
        'time_limit_seconds':args.time_limit,'trial_timeout_seconds':args.trial_timeout,
        'time_policy':args.time_policy,
        'feature_cache':args.feature_cache,
        'refinement':args.refinement,'error_feedback':args.error_feedback,'allocation':args.allocation,
        'incremental_run_allowance_rmb':budget,'maximum_api_requests':maximum_requests,
        'calls_per_adaptive_decision_maximum':calls_per_decision(args.architecture) if pricing else 0,
        'maximum_research_model_requests':2 if args.refinement=='research' else 0,
        'review_failure_policy':'retain_valid_original_for_current_trial_disable_future_architecture_calls_mark_impure',
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
        if scalar_feedback:
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
    extra = {'pretrained_config': pretrained} if pretrained is not None else {}
    if method_choice:extra['method_choice']=method_choice
    summary=execute(task,strategy,args.seed,directory,client,max_trials=args.trials,**extra,
        time_limit=args.time_limit,trial_timeout=args.trial_timeout,anchor_first=True,anchor_policy=args.anchor,refit_full_train=True,
        confirmation_repeats=3,confirmation_top_k=finalists,confirmation_grouping='shared_text',
        search_grouping='shared_text',portfolio=args.portfolio,run_metadata=manifest,time_policy=args.time_policy,
        feature_cache=args.feature_cache=='run',refinement=args.refinement,decision_architecture=args.architecture,error_feedback=args.error_feedback)
    current={p.relative_to(source).as_posix():_sha(p) for p in sorted(source.rglob('*.py'))}
    unchanged=current==source_files
    ledger=[json.loads(line) for line in (directory/'llm_usage.jsonl').read_text().splitlines()]
    unknown=sum(row.get('usage_known') is False for row in ledger)
    model_available=bool(summary['selected_candidate'])
    task_budget_after=account.snapshot() if account is not None else None
    planned_primary=summary.get('resource_allocation',{}).get('final_primary_trials_planned',args.trials)
    ordinary_stop=summary.get('runtime_budget',{}).get('search_stop_reason')
    exhausted=bool(method_choice and ordinary_stop in ('method_candidates_exhausted','method_budget_exhausted'))
    primary_completed=summary['trials_attempted']==planned_primary or exhausted
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
        'method_options_exhausted':exhausted,'search_stop_reason':ordinary_stop,'primary_horizon_completed':primary_completed,'confirmation_completed':confirmation_completed,
        'full_refit_completed':refit_completed,
        'runtime_budget':summary.get('runtime_budget'),
        'feature_cache':summary.get('feature_cache'),
        'classifier_fit_accounting':summary.get('classifier_fit_accounting'),
        'prediction_evidence_audit':summary.get('prediction_evidence_audit'),
        'refinement':summary.get('refinement'),'resource_allocation':summary.get('resource_allocation'),'error_feedback':args.error_feedback,
        'new_api_requests':client.request_count if client else 0,'local_model_forwards':summary.get('local_model_forwards',0),
        'downstream_feature_forwards':summary.get('downstream_feature_forwards',0),'downstream_feature_input_tokens':summary.get('downstream_feature_input_tokens',0),
        'metered_responses':sum(row['event']=='llm_call' for row in ledger),
        'known_course_cost_rmb':sum(row.get('course_cost_rmb') or 0. for row in ledger),
        'complete_course_cost_rmb':summary['course_cost_rmb'],'unresolved_calls':unknown,
        'actual_account_debit_rmb':None if pricing else 0.,'source_unchanged':unchanged,'formal_test_accessed':False,
        'task_budget_after_run':task_budget_after,
        'scope':'Run costs shown separately from the persistent task snapshot; shared cost is counted once in project totals'}
    (directory/'run_outcome.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(outcome,ensure_ascii=False))
    if not model_available:return 1
    return 0 if planner_completed else 2


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
    run.add_argument('--planner',choices=['local','llm','scalar-feedback'],default='local',
        help='Use local search, structured LLM planning, or campus Flash with scalar search feedback')
    run.add_argument('--architecture',choices=ARCHITECTURES,default='single',
        help='LLM mode: single proposal, same-planner self-review, or Planner plus Reviewer; shares the run allowance')
    run.add_argument('--policy',choices=['random','balanced_blind','greedy_routes','protected_routes','interleaved_routes'],help='Local mode only; default balanced_blind')
    run.add_argument('--seed',type=int,default=707)
    run.add_argument('--trials',type=int,default=5)
    run.add_argument('--anchor',choices=['catalog','schema'],default='catalog')
    run.add_argument('--portfolio',choices=['original','expanded','research','nonlinear','adaptive','semantic','learned','challenger'],default='expanded')
    run.add_argument('--time-limit',type=float,default=300.)
    run.add_argument('--trial-timeout',type=float,default=60.)
    run.add_argument('--time-policy',choices=['shared','reserve'],default='shared',help='Optional train-history-based time reservation for confirmation and full refit')
    run.add_argument('--feature-cache',choices=['off','run'],default='off',help='Optional same-run reuse of train-fitted paired TF-IDF; classifiers still refitted')
    run.add_argument('--refinement',choices=['off','midpoints','compose','research'],default='off',help='Optional feedback candidates; research adds bounded literature retrieval and checked sklearn code generation')
    run.add_argument('--allocation',choices=['off','uniform','successive'],default='off',help='Experimental local resource study; counts all screening fits')
    run.add_argument('--error-feedback',action='store_true',help='Expose train-thresholded descriptive error slices to feedback planners')
    from .method_choice import MODES
    run.add_argument('--method-choice',choices=MODES,help='Allocate two search slots after up to five ordinary trials. Use --trials 7.')
    run.add_argument('--pretrained-config',help='Optional paths and allowances: minilm_weights for local frozen features (null disables); weights/tabular_weights, ledger, scope and cap_rmb for metered Qwen transfer')
    run.add_argument('--api-url')
    run.add_argument('--api-key-file')
    run.add_argument('--model')
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
    return parser


def main():
    parser=build_parser()
    args=parser.parse_args()
    try:result=args.handler(args)
    except (ValueError,FileExistsError,FileNotFoundError) as error:
        parser.error(str(error))
    else:raise SystemExit(result)
