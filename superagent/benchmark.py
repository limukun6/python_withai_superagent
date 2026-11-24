"""Controlled comparisons of random, one-shot, feedback and reviewed agents."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .data import DevelopmentTask, load_development_task
from .events import RunLogger
from .experiment_llm import ExperimentLLM, RunAllowance
from .hypothesis_agent import HypothesisAgent, STRATEGIES, check_expectation
from .evidence_agent import EVIDENCE_STRATEGIES, ROUTE_STRATEGIES, EvidenceAgent, route_board,route_of
from .models import search_candidates
from .finalization import finalize
from .confirmation import confirm_candidates
from .splits import take, inner_split, grouped_partition, effective_grouping, overlap_audit
from .profile import profile
from .pricing import DEEPSEEK_PRICE_BASIS, DEEPSEEK_PRICE_VERSION
from .runtime_budget import RuntimeBudget, PlanningTimeExhausted
from .metrics import completed, score_of
from .contracts import TaskSpec,ExperimentProposal,ExperimentResult,DecisionState,ExistingPolicyAdapter


class Policy:
    def __init__(self, strategy, seed, client, logger, limit, catalog=None):
        self.strategy, self.client, self.logger, self.limit = strategy, client, logger, limit
        self.rng = random.Random(seed)
        self.plan = None
        self.local=None
        if strategy.startswith('local_'):
            from .local_search import LocalSearchPolicy
            from .evidence_agent import route_of
            if catalog is None:raise ValueError('Local route policy requires the complete metadata catalog')
            self.local=LocalSearchPolicy(strategy.removeprefix('local_'),
                [{'id':r['id'],'params':r['params'],'route':route_of(r['id'])} for r in catalog],seed)

    def choose(self, data_profile, history, available):
        ids = [item["id"] for item in available]
        if self.local is not None:
            from .evidence_agent import route_of
            chosen,reason=self.local.choose([{**row,'route':route_of(row['candidate'])} for row in history])
            if chosen not in ids:raise ValueError('Local route decision is unavailable')
            return chosen,reason
        if self.strategy == "random":
            return self.rng.choice(ids), "Uniform random draw without replacement."
        if self.strategy == "scalar_feedback":
            metric = data_profile["primary_metric"]
            observations = [
                {"candidate": row["candidate"], "status": row["status"],
                 "search_score": score_of(row) if completed(row) else None}
                for row in history
            ]
            answer = self.client.ask({
                "primary_metric_to_maximize": metric,
                "available_candidates": available,
                "observed_trials": observations,
                "remaining_trials": self.limit + 1 - len(history),
                "instruction": "Choose one available candidate using observed search scores. "
                "Return {candidate: id, reason: a short testable hypothesis}.",
            }, self.logger, "scalar_feedback")
            candidate, reason = answer.get("candidate"), answer.get("reason")
            if candidate not in ids or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Invalid scalar-feedback decision or unavailable candidate.")
            return candidate, reason
        context = {"profile": data_profile, "available_candidates": available, "max_trials": self.limit}
        if self.strategy == "oneshot":
            if self.plan is None:
                answer = self.client.ask({**context,
                    "instruction": "Return {plan: [{candidate: id, reason: explanation}]} with exactly max_trials distinct candidates, in execution order. You will receive no experiment feedback."},
                    self.logger, "initial_plan")
                plan = answer.get("plan")
                if not isinstance(plan, list) or len(plan) != self.limit:
                    raise ValueError("One-shot plan must match the trial limit.")
                selected = [item.get("candidate") for item in plan if isinstance(item, dict)]
                if len(selected) != self.limit or len(set(selected)) != self.limit or not set(selected) <= set(ids):
                    raise ValueError("Invalid or duplicate candidates in one-shot plan.")
                self.plan = list(plan)
            answer = self.plan.pop(0)
        else:
            context["completed_experiments"] = history
            answer = self.client.ask({**context,
                "instruction": "Choose the next experiment using observed metrics and failures. Return {candidate: id, reason: a testable hypothesis and why this configuration helps}."},
                self.logger, "planner")
            if self.strategy == "reviewed":
                answer = self.client.ask({**context, "proposal": answer,
                    "instruction": "Review the proposal against observed results. Keep it or select a better available candidate. Return {candidate: id, reason: critique and final hypothesis}."},
                    self.logger, "reviewer")
        if answer.get("candidate") not in ids or not isinstance(answer.get("reason"), str) or not answer["reason"].strip():
            raise ValueError("Invalid decision schema or unavailable candidate.")
        return answer["candidate"], answer["reason"]


def train_candidate(payload_path, candidate, destination, timeout, *, fit_admission_path=None):
    started = time.monotonic()
    command = [sys.executable, "-m", "superagent.experiment_worker", str(payload_path), candidate, str(destination)]
    registry=Path(destination).parent/'candidate_registry.json'
    if registry.is_file():command.append(str(registry))
    if fit_admission_path is not None:
        command.extend(['--fit-admission', str(fit_admission_path)])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                cwd=Path(__file__).resolve().parents[1])
    except subprocess.TimeoutExpired:
        elapsed=time.monotonic()-started
        return None, {"status": "timeout", "error": "Training time limit reached", "training_seconds": elapsed,"worker_wall_seconds":elapsed}
    if result.returncode:
        elapsed=time.monotonic()-started
        return None, {"status": "failed", "error": result.stderr[-3000:], "training_seconds": elapsed,"worker_wall_seconds":elapsed}
    artifact = joblib.load(destination)
    return artifact["model"], {"status": "completed", **artifact["metrics"],"worker_wall_seconds":time.monotonic()-started}


def run_one(task, strategy, seed, directory, client, max_trials=3, time_limit=600, trial_timeout=120,
            anchor_first=False, refit_full_train=False, ensemble_selection=False,
            confirmation_repeats=1, confirmation_top_k=3, confirmation_grouping="exact_rows", search_grouping="exact_rows",
            portfolio="original",run_metadata=None,time_policy='shared',feature_cache=False,refinement='off',decision_policy=None,
            decision_architecture='single',error_feedback=False,catalog_ids=None,research_searcher=None,anchor_policy='catalog'):
    from .decision_architecture import ArchitecturePolicy,calls_per_decision
    calls_per_decision(decision_architecture)
    if strategy == 'scalar_feedback' and (client is None or not anchor_first or max_trials < 2
            or refinement != 'off' or error_feedback or decision_policy is not None):
        raise ValueError('Scalar feedback requires an LLM, a shared anchor and a fixed candidate catalog')
    if anchor_policy not in ('catalog','schema') or (anchor_policy=='schema' and not anchor_first):
        raise ValueError('Schema anchor requires an enabled first trial')
    if decision_architecture!='single' and (strategy!='budget_routes' or client is None or decision_policy is not None):
        raise ValueError('Reviewed architectures require budget_routes, a client and no custom decision policy')
    from .candidate_refinement import ALLOWED_STRATEGIES,CandidateRegistry
    if refinement not in ('off','midpoints','compose','research'):raise ValueError('Unknown refinement mode')
    if refinement=='research' and (strategy!='budget_routes' or client is None or decision_architecture!='single' or portfolio not in ('original','expanded') or time_policy!='shared' or max_trials<4):
        raise ValueError('Research needs single feedback LLM, original/expanded, shared time and at least4trials')
    if refinement in ('midpoints','compose') and (strategy not in ALLOWED_STRATEGIES or portfolio not in ('original','expanded')):
        raise ValueError('Generated feedback candidates require an approved feedback policy and original/expanded portfolio')
    if confirmation_repeats not in (1, 3) or type(confirmation_top_k) is not int or confirmation_top_k < 1:
        raise ValueError("Confirmation needs repeats 1 or 3 and positive top_k")
    confirmation_mode=effective_grouping(task.kind, confirmation_grouping)
    search_grouping=effective_grouping(task.kind, search_grouping)
    if confirmation_repeats > 1 and ensemble_selection:
        raise ValueError("Confirmation and single-split ensemble selection cannot be combined")
    task_contract=TaskSpec.from_task(task);metric=task_contract.metric
    from .task_semantics import validate_task_semantics
    semantics=validate_task_semantics(task,portfolio)
    if semantics['status']=='rejected':
        raise ValueError('Task semantics contract rejected: '+'; '.join(semantics['reasons']))
    directory.mkdir(parents=True, exist_ok=False)
    logger = RunLogger(directory)
    logger.initialize_usage_ledger()
    logger.write_json('task_contract.json',task_contract.record())
    logger.write_json('task_semantics.json',semantics)
    if run_metadata is not None:
        logger.write_json('run_manifest.json',run_metadata)
        logger.event('run_manifest_frozen',source_sha256=run_metadata['source_sha256'])
    start = time.monotonic()
    runtime=RuntimeBudget(start,time_limit,trial_timeout,confirmation_repeats,confirmation_top_k,
        search_grouping==confirmation_mode,refit_full_train,time_policy)
    if time_policy=='reserve' and ensemble_selection:
        raise ValueError('Time reservation currently supports a single final model, not ensemble refits')
    train_idx, search_idx, split_resource = grouped_partition(task, seed, search_grouping)
    evaluation_protocol={"grouping":search_grouping,"selection_data":"training_partition_only",
                         'split_resource':split_resource,
                         **overlap_audit(task,train_idx,search_idx)}
    evaluation_protocol['planned_final_selection']={
        'primary_metric':metric.name,'search_measurement':f'single_split_{metric.name}',
        'final_metric':f'mean_confirmation_{metric.name}' if confirmation_repeats>1 else f'single_split_{metric.name}',
        'confirmation_repeats':confirmation_repeats,'confirmation_top_k':confirmation_top_k if confirmation_repeats>1 else 0,
        'confirmation_grouping':confirmation_mode if confirmation_repeats>1 else None,
        'maximum_additional_confirmation_fits':min(max_trials,confirmation_top_k)*(confirmation_repeats-int(search_grouping==confirmation_mode)) if confirmation_repeats>1 else 0,
        'incomplete_confirmation_fallback':'retain_primary_best',
        'full_train_refit_after_selection':refit_full_train,'fixed_ensemble_check':ensemble_selection,
        'provided_validation_role':'score_only_after_selection_and_model_freeze'}
    logger.write_json("split.json", {"seed": seed, "train_indices": train_idx.tolist(), "search_indices": search_idx.tolist(),
                                    "evaluation_protocol":evaluation_protocol})
    payload = {"kind": task.kind, "X_train": take(task.X_train, train_idx), "y_train": task.y_train[train_idx],
               "X_search": take(task.X_train, search_idx), "y_search": task.y_train[search_idx],
               'metric_spec':metric.record(),'pair_semantics':semantics['declared_semantics']}
    if error_feedback:payload["error_feedback"]=True
    payload_path = directory / "inner_data.joblib"
    cache_directory=str(directory/'feature_cache') if feature_cache else None
    if cache_directory:payload['feature_cache_dir']=cache_directory
    joblib.dump(payload, payload_path)
    payload_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest()
    inner = DevelopmentTask(task.kind, "inner_split", payload["X_train"], payload["y_train"],
                            payload["X_search"], payload["y_search"], {'metric':metric.name})
    data_profile = profile(inner)
    data_profile["evaluation_protocol"]=evaluation_protocol
    available = [{"id": c.family, "params": c.params} for c in search_candidates(task.kind,portfolio)]
    if semantics['excluded_candidates']:
        excluded=set(semantics['excluded_candidates'])
        available=[row for row in available if row['id'] not in excluded]
        logger.event('swap_augmentation_filtered',candidates=sorted(excluded),policy=semantics['swap_policy'])
    from .recipe_registry import RecipeRegistry
    from .research_registry import ResearchRegistry
    if catalog_ids is not None:
        if not catalog_ids or len(set(catalog_ids))!=len(catalog_ids) or not set(catalog_ids)<={c['id'] for c in available}:
            raise ValueError('Resource candidate pool must be a distinct static subset')
        available=[next(c for c in available if c['id']==name) for name in catalog_ids]
    anchor_id=available[0]['id']
    from .targeted_feedback import candidate_facts
    data_profile['evaluation_protocol']['candidate_facts'] = {
        row['id']: facts for row in available
        if (facts := candidate_facts(task.kind, row)) is not None}
    if anchor_policy=='schema':
        target={'pair_text':'word_char_stats_C0.3','categorical_tabular':'catboost_depth8_reg10.0'}.get(task.kind)
        if target is not None and target not in {row['id'] for row in available}:
            raise ValueError('Requested schema anchor is absent from the candidate pool')
        anchor_id=target or anchor_id
    registry=(CandidateRegistry(task.kind,portfolio,directory/'candidate_registry.json') if refinement=='midpoints' else
              RecipeRegistry(task.kind,portfolio,directory/'candidate_registry.json') if refinement=='compose' and task.kind in ('single_text','pair_text') else
              ResearchRegistry(task.kind,portfolio,directory/'candidate_registry.json',client,logger,data_profile,metric,research_searcher) if refinement=='research' else None)
    if registry:
        data_profile['evaluation_protocol']['candidate_generation']=registry.description()
        logger.write_json('candidate_registry_initial.json',registry.document)
    logger.write_json("data_profile.json", data_profile)
    if strategy in EVIDENCE_STRATEGIES:
        policy = EvidenceAgent(client, logger, max_trials, strategy)
    elif strategy in STRATEGIES:
        policy = HypothesisAgent(client, logger, max_trials, strategy == "reflective")
    else:
        policy = Policy(strategy, seed, client, logger, max_trials - int(anchor_first),available)
    decider=(decision_policy if decision_policy is not None else
             ArchitecturePolicy(client,logger,decision_architecture) if strategy=='budget_routes' else
             ExistingPolicyAdapter(policy))
    feedback_visible=strategy not in ('random','oneshot','evidence_blind','reflective_no_feedback','local_random','local_balanced_blind')
    states=[];proposals=[];result_records=[]
    history, best_model, best_id, best_score = [], None, None, -1.0
    fallbacks, blocked_llm = 0, False
    stop_reason='trial_limit_reached'
    random_fallback = random.Random(seed)
    for index in range(max_trials):
        if not available:
            stop_reason='candidate_catalog_exhausted';break
        needs_planner=client is not None and not blocked_llm and not (anchor_first and index==0)
        window=runtime.search_window(history,needs_planner)
        logger.event('runtime_search_gate',trial=index+1,**window)
        if not window['allowed']:
            stop_reason=window['reason']
            logger.event('search_stopped_for_time',reason=stop_reason,trials_attempted=len(history))
            break
        if needs_planner and time_policy=='reserve':
            if not hasattr(client,'set_deadline'):raise ValueError('Reserved-time LLM clients must implement set_deadline')
            client.set_deadline(window['planner_deadline'])
        elif needs_planner and strategy == 'scalar_feedback':
            client.set_deadline(start + time_limit)
        if time_policy=='reserve':
            data_profile['evaluation_protocol']['runtime_budget']={**runtime.description(),**window}
        proposal = None
        declared=None
        state=DecisionState.build(data_profile,history,available,max_trials,feedback_visible,task_contract)
        states.append(state.record());logger.write_json('decision_states.json',states)
        planning_start=time.monotonic()
        try:
            if anchor_first and index == 0:
                chosen, reason = anchor_id, f"{anchor_policy} first configuration; counts toward the trial limit."
            else:
                if blocked_llm:
                    raise RuntimeError("Planner disabled after earlier failure.")
                declared=decider.propose(state)
                if not isinstance(declared,ExperimentProposal):raise ValueError('DecisionPolicy must return ExperimentProposal')
                declared.validate_available(available,history)
                chosen,reason=declared.candidate,declared.reason
                proposal=declared.raw_proposal
        except PlanningTimeExhausted:
            stop_reason='planning_window_expired_without_request'
            logger.event('search_stopped_for_time',reason=stop_reason,trials_attempted=len(history));break
        except Exception as error:
            if decision_policy is not None and getattr(decision_policy, 'stop_on_error', False):
                logger.event('decision_policy_stopped', error=str(error), trial=index+1)
                raise
            if strategy == 'scalar_feedback':
                logger.event('scalar_feedback_stopped', error=str(error), trial=index+1)
                raise
            blocked_llm = True
            fallbacks += 1
            chosen = random_fallback.choice(available)["id"]
            reason = "Random recovery after planner failure; this is not a valid pure LLM run."
            logger.event("planner_fallback", error=str(error), chosen=chosen)
        if needs_planner:runtime.observe_planner(time.monotonic()-planning_start)
        # Planning latency is paid before assigning the worker its remaining slice.
        fit_window=runtime.search_window(history,False)
        remaining_seconds=fit_window['fit_timeout_seconds']
        if remaining_seconds <= 0:
            stop_reason='planning_consumed_search_window'
            break
        configuration = next(item for item in available if item["id"] == chosen)
        declared=ExperimentProposal.from_decision(index+1,configuration,reason,proposal,history)
        proposals.append(declared.record());logger.write_json('experiment_proposals.json',proposals)
        logger.event('experiment_proposal_frozen',**declared.record())
        available = [item for item in available if item["id"] != chosen]
        explanation_audit=None
        if proposal is not None:
            from .explanation_audit import audit_explanation
            explanation_audit=audit_explanation(proposal,history,configuration)
            logger.event('decision_evidence_report',trial=index+1,**explanation_audit)
        logger.event("decision", trial=index + 1, candidate=chosen, reason=reason,
                     params=configuration["params"], proposal=proposal)
        model, metrics = train_candidate(payload_path, chosen, directory / f"trial_{index+1}.joblib",
                                         min(trial_timeout, remaining_seconds))
        row = {'primary_metric':metric.name,"trial": index + 1, "candidate": chosen, "params": configuration["params"],
               "proposal": proposal, "explanation_audit":explanation_audit, **metrics}
        # Validate objective identity before a result can replace the incumbent.
        row['metric_identity']=metric.check_identity(row.get('metric_spec'))
        if row.get('primary_metric')!=metric.name:raise ValueError('Worker result uses a different task metric')
        from .prediction_evidence import comparison_for_trial
        row['paired_error_comparison']=comparison_for_trial(row,history,directory,payload['y_search'],payload_sha256,metric)
        logger.event('paired_prediction_evidence',trial=index+1,comparison=row['paired_error_comparison'])
        row["hypothesis_check"] = check_expectation(proposal, row, history)
        logger.event("hypothesis_checked", trial=index + 1, **row["hypothesis_check"])
        from .estimator_facts import verify_configuration_claim
        row['configuration_fact_check']=verify_configuration_claim(proposal, metrics.get('estimator'))
        logger.event('configuration_fact_checked',trial=index+1,**row['configuration_fact_check'])
        if model is not None and (score_of(row) > best_score or
            (score_of(row) == best_score and chosen < best_id)):
            best_score, best_id, best_model = score_of(row), chosen, model
        best_row=next((r for r in [*history,row] if r['candidate']==best_id),None)
        row['best_search_score']=None if best_model is None else best_score
        row["best_search_accuracy"] = best_row['search_accuracy'] if best_row else None
        result_records.append(ExperimentResult.from_trial(row,declared,metric).record())
        logger.write_json('experiment_results.json',result_records)
        row["elapsed_seconds"] = time.monotonic() - start
        history.append(row)
        from .hypothesis_memory import build_memory
        logger.event("trial_result", **row)
        logger.write_json("trials.json", history)
        if registry and index+1<max_trials:
            if refinement=='research' and hasattr(client,'set_deadline'):client.set_deadline(start+time_limit-runtime.margin)
            generated=registry.propose(history)
            if refinement=='research':data_profile['evaluation_protocol']['candidate_generation']=registry.description()
            for entry in generated:
                available.append({'id':entry['id'],'params':entry['params']})
                logger.event('candidate_generated',**entry)
            if generated:
                logger.write_json(f'candidate_registry_after_trial_{index+1}.json',registry.document)
                if isinstance(policy,Policy) and policy.local is not None:
                    policy.local.add_candidates([{'id':r['id'],'params':r['params'],'route':route_of(r['id'])} for r in generated])
        if strategy in ROUTE_STRATEGIES:
            logger.write_json("route_memory.json", route_board(history, available))
        logger.write_json('hypothesis_memory.json',build_memory(history,available))
    primary_best_id = best_id
    if client is not None and hasattr(client,'set_deadline'):client.set_deadline(None)
    confirmation_limit=runtime.confirmation_limit(history)
    logger.event('runtime_finalization_budget',confirmation_deadline_elapsed_seconds=confirmation_limit,
        refit_deadline_elapsed_seconds=time_limit-runtime.margin,**runtime.tail(history))
    confirmation = confirm_candidates(task, history, seed, directory, logger, train_candidate,
        start, confirmation_limit, trial_timeout, confirmation_repeats, confirmation_top_k, confirmation_grouping, search_grouping,
        feature_cache_dir=cache_directory)
    if confirmation.get("enabled") and confirmation.get("selected_candidate"):
        confirmed = next(r for r in history if r["candidate"] == confirmation["selected_candidate"])
        if confirmed["candidate"] != best_id:
            best_model = joblib.load(directory / f"trial_{confirmed['trial']}.joblib")["model"]
        best_id, best_score = confirmed["candidate"], score_of(confirmed)
    from .task_semantics import expectations_for,probe_model_swap,declaration_consistency
    swap_probe=None
    if task.kind=='pair_text' and best_model is not None:
        try: swap_probe=probe_model_swap(task.kind,best_model,payload['X_search'])
        except Exception as error: swap_probe={'applicable':False,'reason':f'{type(error).__name__}: {error}'}
        swap_probe.update(expectations_for(semantics['declared_semantics']))
        selected=next((r for r in history if r['candidate']==best_id),None)
        swap_probe['selected_candidate_assumes_flipping']=bool(
            (selected or {}).get('params',{}).get('swap_augmentation'))
        if swap_probe.get('selected_candidate_assumes_flipping'):
            swap_probe['interpretation_note']=('This candidate trains on negated features with flipped '
                'labels, so antisymmetry follows from the construction; a flips result is not evidence '
                'that the task itself is a preference task')
        swap_probe['consistent_with_declaration']=declaration_consistency(swap_probe)
    finalized = finalize(task, history, best_model, best_id, best_score, payload, directory,
                         logger, train_candidate, start, time_limit-runtime.margin, trial_timeout,
                         refit_full_train, ensemble_selection, confirmation,feature_cache_dir=cache_directory)
    usage = [json.loads(line) for line in logger.usage_path.read_text(encoding="utf-8").splitlines()]
    usage_complete = not any(u.get("usage_known") is False for u in usage)
    course_cost = sum(u.get("course_cost_rmb", u.get("cost_rmb")) or 0 for u in usage) if usage_complete else None
    actual_cost = (sum(u["actual_cost_rmb"] for u in usage)
                   if all(u.get("actual_cost_rmb") is not None for u in usage) else None)
    summary = {"strategy": strategy, "seed": seed, "task_kind": task.kind,
        "selected_candidate": best_id, "primary_best_candidate": primary_best_id,
        "confirmation": confirmation, "portfolio":portfolio,"evaluation_protocol":evaluation_protocol,
        'primary_metric':metric.name,'search_score':best_score if best_model is not None else None,
        "search_accuracy": next((r['search_accuracy'] for r in history if r['candidate']==best_id),None),
        "official_validation_accuracy": finalized["official_validation_accuracy"],
        "completed": best_model is not None, "valid_policy_run": best_model is not None and fallbacks == 0,
        "fallback_count": fallbacks, "trials_attempted": len(history),
        "failed_trials": sum(r["status"] != "completed" for r in history),
        "elapsed_seconds": time.monotonic() - start,
        "llm_calls": sum(u["event"] != "ledger_initialized" for u in usage),
        "tokens": sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage),
        "course_cost_rmb": course_cost,
        "actual_cost_rmb": actual_cost,
        "usage_complete": usage_complete,
        "anchor_first": anchor_first,
        "anchor_policy":anchor_policy,"anchor_candidate":anchor_id if anchor_first else None,
        "hypotheses_evaluated": sum(row["hypothesis_check"]["status"] in ("supported", "not_supported") for row in history),
        "expectations_met": sum(row["hypothesis_check"]["status"] == "supported" for row in history),
        "test_accessed": False, "run_directory": str(directory)}
    summary.update(finalized)
    summary['task_semantics']=semantics
    summary['swap_behavior']=swap_probe
    fact_checks=[row.get('configuration_fact_check') or {} for row in history]
    summary['configuration_facts']={
        'declared_and_confirmed':sum(c.get('status')=='confirmed' for c in fact_checks),
        'declared_and_contradicted':sum(c.get('status')=='contradicted' for c in fact_checks),
        'declared_but_unverifiable':sum(c.get('status')=='unverified' for c in fact_checks),
        'no_claim_made':sum(c.get('status')=='not_declared' for c in fact_checks),
        'scope':'Declared classifier compared with the estimator actually fitted by the worker'}
    architecture=(decider.summary() if isinstance(decider,ArchitecturePolicy) else
                  {'architecture':'custom' if decision_policy is not None else 'single','enabled':False,
                   'review_fallback_count':0,'all_requested_decisions_valid':True})
    summary['decision_architecture']=architecture
    summary['valid_policy_run']=summary['valid_policy_run'] and architecture['all_requested_decisions_valid']
    summary['refinement']=registry.description() if registry else {'enabled':False,'mode':'off','requested_mode':refinement,'generated_candidates':0,'reason':'not applicable to modality' if refinement=='compose' else None}
    if refinement=='research':
        summary['refinement']['generated_candidates_executed']=sum(r['candidate'] in registry.document['entries'] for r in history)
        summary['refinement']['generated_candidates_completed']=sum(r['candidate'] in registry.document['entries'] and r['status']=='completed' for r in history)
        summary['valid_policy_run']=summary['valid_policy_run'] and summary['refinement']['all_requested_research_valid']
    if registry:summary['refinement']['registry_sha256']=hashlib.sha256(registry.path.read_bytes()).hexdigest()
    comparisons=[row['paired_error_comparison'] for row in history]
    summary['prediction_evidence_audit']={'comparisons_completed':sum(r['status']=='completed' for r in comparisons),
        'comparisons_unavailable':sum(r['status']=='unavailable' for r in comparisons),
        'invalid_comparisons':sum(r['status']=='invalid' for r in comparisons),
        'integrity_valid':not any(r['status']=='invalid' for r in comparisons)}
    audits=[row['explanation_audit'] for row in history if row.get('explanation_audit')]
    summary['explanation_review']={'audited_decisions':len(audits),
        'decisions_flagged':sum(bool(a['warnings']) for a in audits),
        'warning_count':sum(len(a['warnings']) for a in audits),
        'freeform_prose_fully_verified':False,
        'valid_policy_run_scope':'Completed execution without planner fallback; not certification of all prose claims'}
    summary["route_actions"] = {
        action: sum((r.get("proposal") or {}).get("route_update", {}).get("action") == action for r in history)
        for action in ("start", "continue", "revisit")}
    summary["search_training_seconds"] = sum(r.get("training_seconds", 0.) for r in history)
    summary['runtime_budget']={**runtime.description(),'search_stop_reason':stop_reason,
        'requested_primary_trials':max_trials,'actual_primary_trials':len(history),
        'confirmation_deadline_elapsed_seconds':confirmation_limit,
        'elapsed_exceeded_requested_limit':summary['elapsed_seconds']>time_limit}
    summary['feature_cache']={'enabled':feature_cache,'directory':cache_directory,
        'scope':'same run, unsupervised paired TF-IDF; all classifiers still fitted and logged separately'}
    unknown_component_attempts=(summary['failed_trials']+confirmation.get('failed_attempts_with_unknown_component_count',0)
        +summary['full_refits_attempted']-summary['full_refits_completed'])
    summary['classifier_fit_accounting']={
        'completed_classifier_fits':sum(r.get('classifier_fits',1) for r in history if r['status']=='completed')
            +confirmation.get('classifier_fits_completed',0)+summary['refit_classifier_fits_completed'],
        'failed_candidate_attempts_with_unknown_component_count':unknown_component_attempts,
        'complete_count_known':unknown_component_attempts==0,
        'scope':'Counts each fitted classifier in fixed blends; failed workers may have fitted partial components'}
    logger.write_json("summary.json", summary)
    return summary


def write_report(destination, rows):
    lines = ["# Agent architecture comparison (development)", "",
        "Official validation is scored only after inner-search selection. Previously used official validation is not a pristine holdout.",
        "Only valid policy runs enter primary-score averages; fallback runs remain visible in summaries. Course costs use frozen official DeepSeek off-peak rates without cache discounts. Account debits are separate.", "",
        "| Task | Policy | Valid / attempted | Mean validation primary score | Sample SD | Mean seconds (all) | Tokens (all) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for task, strategy, name in sorted({(r["task_kind"], r["strategy"],r.get("primary_metric","accuracy")) for r in rows}):
        group = [r for r in rows if r["task_kind"] == task and r["strategy"] == strategy and r.get("primary_metric","accuracy")==name]
        scores = [r.get("official_validation_score",r["official_validation_accuracy"] if name=="accuracy" else None) for r in group if r["valid_policy_run"]]
        scores = [value for value in scores if value is not None]
        mean = f"{statistics.mean(scores):.4f}" if scores else "N/A"
        sd = f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "N/A"
        lines.append(f"| {task} ({name}) | {strategy} | {len(scores)} / {len(group)} | {mean} | {sd} | {statistics.mean(r['elapsed_seconds'] for r in group):.1f} | {sum(r['tokens'] for r in group)} |")
    lines += ["", "## Interpretation limits", "",
        "One repeat is a feasibility pilot, not evidence of architecture superiority. Compare matched seeds, report all failures, and repeat before selecting an architecture.",
        "Reviewed agents use extra calls; this trial-matched comparison does not isolate architecture from token expenditure. A token-matched follow-up is required.",
        "The fixed search space contains 14 configurations per task. These experiments test search policy, not unrestricted code generation or generality to unseen modalities.",
        "All classifiers retain model seed 42; repeat seeds control the shared data split and random search. This does not estimate independent model-training variance."]
    (destination / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+")
    parser.add_argument("--strategies", nargs="+", choices=["random", "oneshot", "feedback", "reviewed", *STRATEGIES, *EVIDENCE_STRATEGIES], default=["random"])
    parser.add_argument("--refit-full-train", action="store_true")
    parser.add_argument("--ensemble-selection", action="store_true")
    parser.add_argument("--confirmation-repeats", type=int, choices=(1,3), default=1)
    parser.add_argument("--confirmation-top-k", type=int, default=3)
    parser.add_argument("--confirmation-grouping", choices=("exact_rows", "shared_text", "near_text"), default="exact_rows")
    parser.add_argument("--search-grouping", choices=("exact_rows", "shared_text", "near_text"), default="exact_rows")
    parser.add_argument("--portfolio",choices=("original","expanded","research","nonlinear","adaptive"),default="original")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--trial-timeout", type=float, default=120)
    parser.add_argument("--output-dir", default="runs/benchmarks")
    parser.add_argument("--api-url", default="https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions")
    parser.add_argument("--api-key-file")
    parser.add_argument("--model", default="DeepSeek-V4-Pro")
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"))
    parser.add_argument("--disable-template-thinking", action="store_true")
    parser.add_argument("--batch-budget-rmb", type=float, default=2.0)
    parser.add_argument("--per-run-budget-rmb", type=float,
                        help="Equal course-cost allowance per LLM run; all allowances must fit in the batch ceiling")
    parser.add_argument("--anchor-first", action="store_true",
                        help="Use the same first configuration for all policies, within the existing trial count")
    args = parser.parse_args()
    if not 1 <= args.max_trials < 14 or not 0 < args.time_limit < float("inf") or not 0 < args.trial_timeout < float("inf"):
        parser.error("Use 1..13 trials and positive finite time limits.")
    if args.confirmation_top_k < 1 or (args.confirmation_repeats > 1 and args.ensemble_selection):
        parser.error("Use positive confirmation top_k and do not combine confirmation with single-split ensembles.")
    strategies = list(dict.fromkeys(args.strategies))
    seeds = list(dict.fromkeys(args.seeds))
    if args.per_run_budget_rmb is not None:
        if not math.isfinite(args.per_run_budget_rmb) or not 0 < args.per_run_budget_rmb <= 20:
            parser.error("Per-run budget must be finite and in (0, 20].")
        llm_runs = len(args.packages) * len(seeds) * sum(s != "random" for s in strategies)
        if llm_runs * args.per_run_budget_rmb > args.batch_budget_rmb + 1e-12:
            parser.error("Preallocate equal allowances for every LLM run within the batch budget.")
    tasks=[load_development_task(package) for package in args.packages]
    for task in tasks:TaskSpec.from_task(task)
    client = None
    if any(s != "random" for s in strategies):
        if not args.api_key_file:
            parser.error("LLM strategies require --api-key-file.")
        client = ExperimentLLM(args.api_url, args.api_key_file, args.model, args.batch_budget_rmb,
                               reasoning_effort=args.reasoning_effort, disable_template_thinking=args.disable_template_thinking)
    destination = Path(args.output_dir).resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination.mkdir(parents=True)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    config = {k: v for k, v in vars(args).items() if k != "api_key_file"}
    config.update({"source_sha256": source_hash.hexdigest(), "python": sys.version,
                   "price_status": DEEPSEEK_PRICE_BASIS if client is not None else "no_model_calls",
                   "price_version": DEEPSEEK_PRICE_VERSION if client is not None else None,
                   "pricing": client.price.metadata() if client is not None else None,
                   "test_accessed": False})
    (destination / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for task_index, package in enumerate(args.packages):
        task = tasks[task_index]
        for seed in seeds:
            order = strategies.copy()
            random.Random(seed).shuffle(order)
            for strategy in order:
                directory = destination / f"task{task_index}_{task.kind}_{strategy}_{seed}"
                run_client = (RunAllowance(client, args.per_run_budget_rmb)
                              if strategy != "random" and args.per_run_budget_rmb is not None else client)
                summary = run_one(task, strategy, seed, directory, run_client, args.max_trials,
                                  args.time_limit, args.trial_timeout, args.anchor_first, args.refit_full_train, args.ensemble_selection,
                                  args.confirmation_repeats, args.confirmation_top_k, args.confirmation_grouping,
                                  args.search_grouping,args.portfolio)
                summary["course_run_allowance_rmb"] = args.per_run_budget_rmb if strategy != "random" else 0
                (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                rows.append(summary)
                (destination / "summaries.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                write_report(destination, rows)
                print(json.dumps(summary), flush=True)
    print(f"Comparison: {destination / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
