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


def configured_flash(directory):
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
    args=SimpleNamespace(api_url=None,model=None,api_key_file=settings[0],budget_ledger=settings[1],
                         task_budget_id=settings[2],llm_budget_rmb=None,history_policy='strict')
    ledger,allowance,_=preflight(args)
    return ExperimentLLM(args.api_url,args.api_key_file,args.model,allowance,max_requests=3,
                         allowed_served_models={args.model},account=TaskBudgetAccount(ledger,args.task_budget_id,str(directory)),
                         streaming=True)


def command(args):
    from .autonomous_run import train
    if getattr(args,'policy','autonomous')=='legacy':
        return legacy_command(args)
    return train(args.task_package,args.run_dir,fit_limit=args.fit_limit,seed=args.seed,
                 wall=args.time_limit,timeout=args.trial_timeout,flash=configured_flash(args.run_dir))
