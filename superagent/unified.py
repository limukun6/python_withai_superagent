"""One local planning configuration for anonymous classification tasks."""
from pathlib import Path

from .ab_method_search import rank
from .data import DevelopmentTask, load_training_task


def training_input(path):
    training = load_training_task(path)
    spec = dict(training.spec, train_rows=len(training.y_train), validation_rows=0,
                provided_validation_available=False, test_accessed=False)
    return DevelopmentTask(training.kind, str(Path(path)), training.X_train, training.y_train, None, None, spec)


def categorical_opening(state, caps, eligible, verified):
    """Compare shallow and deep raw-category trees at equal regularization before broader search."""
    choice = rank(state, caps, eligible, verified)
    if state['task']['kind'] != 'categorical_tabular' or len(verified) > 1:
        return choice
    native = {n: c['params'] for n,c in caps.items() if c['representation'] == 'native categories'
              and 'depth' in c['params'] and 'l2_leaf_reg' in c['params']}
    if len({p['depth'] for p in native.values()}) < 2:
        return choice
    if verified:
        first = verified[0]
        if first['candidate'] not in native or first['train_score'] >= 1:
            return choice
        penalty = native[first['candidate']]['l2_leaf_reg']
    else:
        penalty = min(p['l2_leaf_reg'] for p in native.values())
    cohort = [n for n in eligible if n in native and native[n]['l2_leaf_reg'] == penalty]
    if not cohort:
        return choice
    depth = (max if verified else min)(native[n]['depth'] for n in cohort)
    candidates = {n for n in cohort if native[n]['depth'] == depth}
    target = next(row['candidate'] for row in choice['candidates'] if row['candidate'] in candidates)
    return dict(choice, chosen=target, coverage_rule='matched native-category depth endpoints',
                matched_penalty=penalty, selected_depth=depth, unrestricted_rank_chosen=choice['chosen'])


def legacy_command(args):
    from .cli import build_parser
    native = build_parser().parse_args([
        'run', args.task_package, '--run-dir', args.run_dir, '--planner', 'local',
        '--scheduler', 'ab-adaptive-complement', '--portfolio', 'ab-v1',
        '--preference-catalog', 'deep-e5-familiarity', '--cover-untried-encoder',
        '--task-profile', 'train', '--fit-limit', str(args.fit_limit), '--seed', str(args.seed),
        '--trial-timeout', str(args.trial_timeout), '--time-limit', str(args.time_limit)])
    native.unified = True
    return native.handler(native)


def configured_flash(directory, *, thinking_mode=None):
    """Optional deployment configuration. Billing identifiers never enter decision states."""
    import os
    from types import SimpleNamespace
    from .ab_flash import preflight
    from .experiment_llm import ExperimentLLM
    from .task_budget import TaskBudgetAccount
    settings = [os.getenv(k) for k in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID')]
    if not any(settings):
        return None
    if not all(settings):
        raise ValueError('Configure all three Flash billing settings or leave them all unset')
    accepted_path=os.getenv('SUPERAGENT_ACCEPTED_HISTORY_FILE')
    accepted_sha=os.getenv('SUPERAGENT_ACCEPTED_HISTORY_SHA256')
    if bool(accepted_path)!=bool(accepted_sha):
        raise ValueError('Accepted-history file and SHA256 must both be configured')
    campus_ip=os.getenv('SUPERAGENT_CAMPUS_DIRECT_IP') or None
    if accepted_path:
        if os.environ.get('SUPERAGENT_OFFLINE')=='1':
            raise ValueError('LLM mode is disabled in the current offline environment')
        from .accepted_history import AcceptedHistoryAccount
        from .task_budget import TaskBudgetLedger
        from .ab_flash import ENDPOINT,MODEL
        account=AcceptedHistoryAccount(TaskBudgetLedger(settings[1]),settings[2],str(directory),accepted_path,accepted_sha)
        account.preflight()
        return ExperimentLLM(ENDPOINT,settings[0],MODEL,account.allowance,max_requests=min(3,account.max_requests),
                             allowed_served_models={MODEL},account=account,thinking_mode=thinking_mode,
                             streaming=True,campus_ip=campus_ip)
    args=SimpleNamespace(api_url=None,model=None,api_key_file=settings[0],budget_ledger=settings[1],
                         task_budget_id=settings[2],llm_budget_rmb=None,history_policy='strict')
    ledger,allowance,_=preflight(args)
    return ExperimentLLM(args.api_url,args.api_key_file,args.model,allowance,max_requests=3,
                         allowed_served_models={args.model},account=TaskBudgetAccount(ledger,args.task_budget_id,str(directory)),
                         thinking_mode=thinking_mode,streaming=True,campus_ip=campus_ip)


def reflection_client(directory,mode='flash'):
    import os
    from .ab_flash import ENDPOINT, MODEL
    from .experiment_llm import ExperimentLLM
    from .task_budget import TaskBudgetAccount, TaskBudgetLedger
    path = os.getenv('SUPERAGENT_RESEARCH_REFLECTION_OUTBOX')
    if not path:
        return configured_flash(directory,thinking_mode='disabled' if mode in ('opening-flash','opening-feedback-flash','opening-feedback-design-flash') else None)
    if os.getenv('SUPERAGENT_OFFLINE') == '1' or os.getenv('SUPERAGENT_PLANNER_LEDGER'):
        raise ValueError('Explicit online research outbox or the team ledger must be chosen separately')
    ledger = TaskBudgetLedger(path)
    scope = os.environ.get('SUPERAGENT_TASK_BUDGET_ID','')
    arm = mode if os.getenv('SUPERAGENT_RESEARCH_PAIRED_REFLECTION') == '1' else None
    if arm == 'scheduled-flash':
        arm = 'conditional-flash'  # Both conditional timing policies share one paid slot per task.
    ledger.preflight(scope,.12,fresh_outbox_scope=True,paired_arm=arm)
    key = os.getenv('SUPERAGENT_PLANNER_KEY_FILE')
    if not key:
        raise ValueError('A campus key file is required')
    return ExperimentLLM(ENDPOINT,key,MODEL,.12,max_requests=1,allowed_served_models={MODEL},
                         thinking_mode='disabled' if mode in ('proposal-flash','delivered-flash','opening-flash','opening-feedback-flash','opening-feedback-design-flash') else None,
                         account=TaskBudgetAccount(ledger,scope,str(directory),fresh_outbox_scope=True,paired_arm=arm),streaming=True)


def command(args):
    from .autonomous_run import train
    from .planner_review import feedback_mode
    reflection=feedback_mode(args)
    if getattr(args,'policy','autonomous')=='legacy':
        if getattr(args,'self_review',False) or getattr(args,'exploration_rule',None) or reflection or getattr(args,'reflection_history',None):
            raise ValueError('Planner self-review uses the autonomous entry point')
        if (getattr(args,'task_interpretation','fallback') not in ('fallback','auto') or getattr(args,'prediction_input',None) is not None
                or getattr(args,'semantic_artifact_from',None) is not None):
            raise ValueError('Semantic interpretation and pre-fit input checks require the autonomous policy')
        from .task_constraints import package_restrictions
        if not package_restrictions(args.task_package):
            return legacy_command(args)
    issue = None
    try:
        if reflection in ('flash','conditional-flash','scheduled-flash','proposal-flash','delivered-flash','opening-flash','opening-feedback-flash','opening-feedback-design-flash'):
            flash = reflection_client(args.run_dir,reflection)
        else:
            flash = None if getattr(args,'semantic_artifact_from',None) or reflection and getattr(args,'task_interpretation','fallback') not in ('auto','proactive') else configured_flash(args.run_dir)
    except (OSError, ValueError, RuntimeError) as error:
        flash, issue = None, str(error)[:320]
    return train(args.task_package,args.run_dir,fit_limit=args.fit_limit,seed=args.seed,
                 wall=args.time_limit,timeout=args.trial_timeout,flash=flash,flash_setup_issue=issue,
                 self_review=getattr(args,'self_review',False),exploration_rule=getattr(args,'exploration_rule',None),
                 feedback_reflection=reflection,reflection_history=getattr(args,'reflection_history',None),
                 interpretation=getattr(args,'task_interpretation','fallback'),
                 prediction_package=getattr(args,'prediction_input',None),
                 semantic_artifact_from=getattr(args,'semantic_artifact_from',None),
                 await_first_delivery=getattr(args,'await_first_delivery',False))
