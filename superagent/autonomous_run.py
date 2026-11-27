"""Course entry point for anonymous classification. Only training data reach the planner."""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .ab_methods import PREFERENCE_CATALOG_ENV
from .ab_run import Executor, explicit_windows, registered_capabilities, sha, sources
from .autonomous_checkpoint import frozen_spec, interrupted, promote, standby
from .contracts import TaskSpec
from .data import TrainingTask, load_training_task
from .events import RunLogger
from .experiment_policy import actions, identity
from .exploration_rule import apply_rule, load_rule
from .metrics import MetricSpec
from .models import SoftVoteModel, predict_with_model, probability_with_model
from .open_methods import MethodSpace
from .prediction_evidence import fingerprint, _checked_predictions
from .splits import sample_groups, take
from .task_profile import training_facts
from .task_semantics import declared_semantics


@contextmanager
def installed_methods():
    previous = os.environ.get(PREFERENCE_CATALOG_ENV)
    os.environ[PREFERENCE_CATALOG_ENV] = 'all-installed'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PREFERENCE_CATALOG_ENV, None)
        else:
            os.environ[PREFERENCE_CATALOG_ENV] = previous


def partitions(task, seed):
    declared = explicit_windows(task, seed)
    if declared is not None:
        return declared
    protocol = (task.spec.get('evaluation_protocol') or {}).get('kind','grouped_stratified')
    if protocol != 'grouped_stratified':
        raise ValueError('Unsupported evaluation protocol: '+str(protocol))
    groups = sample_groups(task, 'shared_text')
    y = task.y_train
    labels = set(y.tolist())
    most = min(5, *(len(set(groups[y==c])) for c in labels))
    for count in range(most, 1, -1):
        folds = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
        usable = [(a,b) for a,b in folds.split(np.zeros(len(y)),y,groups)
                  if set(y[a].tolist())==labels and set(y[b].tolist())==labels]
        if usable:
            return {name:(a,b,dict(protocol=protocol,fold_count=count,shared_groups=0))
                    for name,(a,b) in zip(('primary','confirmation_1','confirmation_2'),usable)}
    raise ValueError('No class-preserving holdout of independent training groups is available')


def planner_observation(bundle, facts, record, metric):
    labels = record['labels']
    evidence = bundle['prediction_record'] if 'prediction_record' in bundle else bundle
    if evidence.get('evaluation_payload_sha256')!=record['sha256']:
        raise ValueError('Worker predictions belong to a different training partition')
    pred = _checked_predictions(evidence,labels)
    probs = None
    scored = evidence.get('score_record')
    if scored:
        from .score_evidence import checked_score_record
        probs, classes = checked_score_record(scored, evidence, metric, facts['resolved_configuration']['id'], labels)
        probs = probs[:,[classes.index(c) for c in metric.labels]]
    scored = metric.evaluate(labels, pred, probs, list(metric.labels) if probs is not None else None)
    if abs(facts['search_score']-scored['score'])>1e-9:
        raise ValueError('Reported search score and row-level predictions disagree')
    return dict(score=scored['score'], labels=labels, probabilities=probs, predictions=pred,
                row_key=evidence['evaluation_payload_sha256'])


def choose_with_flash(client, logger, menu, caps, profile, metric, history, remaining, recovery_context=None):
    # The provider sees anonymous capabilities, training scores and feasible actions. It never sees raw rows.
    brief = [{k:a[k] for k in ('action','members','split','evidence')} for a in menu]
    payload = dict(instruction='Choose the next experiment, corroboration or final model, considering uncertainty, '
                   'generalization and remaining resources. Options include software-generated pipelines. '
                   'Return JSON with integer option (zero-based), reason, and evidence_trials (a list of observed '
                   'trial numbers). Use only supplied observations. Never assume a particular dataset or winner.',
                   metric=metric.name, profile=profile, fits_remaining=remaining,
                   observations=[{k:r.get(k) for k in ('trial','candidate','split_id','status','train_score',
                                                        'search_score','class_recall','paired_errors')} for r in history],
                   methods={m:dict(estimator=caps[m]['estimator'],representation=caps[m]['representation'],
                                   resource=caps[m]['resource'],configuration=caps[m]['params'])
                            for m in sorted({n for action in menu for n in action['members']})},options=brief)
    if recovery_context is not None:payload['recovery_state']=recovery_context
    proposal=client.ask(payload,logger,'experiment_planner')
    index=proposal.get('option')
    refs=proposal.get('evidence_trials')
    if (type(index) is not int or not 0<=index<len(menu) or not isinstance(proposal.get('reason'),str)
            or not proposal['reason'].strip() or not isinstance(refs,list)
            or any(type(r) is not int or r not in {h['trial'] for h in history} for r in refs)):
        raise ValueError('Planner returned an ungrounded or infeasible action')
    return dict(menu[index],reason=proposal['reason'][:1000],cited_trials=refs)


def freeze_model(task, spec, metric, choice, executor, directory, log, timeout, *, challenger=False, challenger_prefix='challenger_', provisional_total=None):
    members = choice['members']
    prefix = 'core_ready_' if provisional_total is not None else challenger_prefix if challenger else ''
    log.write_json(prefix + 'selection.json', dict(choice=choice, scope='frozen before full training and external evaluation'))
    refit = directory/('core_ready_refit_data.joblib' if provisional_total is not None else 'refit_data.joblib')
    if not challenger:
        joblib.dump(dict(kind=task.kind,mode='refit',X_train=task.X_train,y_train=task.y_train,
                         X_search=take(task.X_train,[0]),metric_spec=metric.record(),
                         pair_semantics=declared_semantics(task)),refit)
    models = []
    for i,name in enumerate(members):
        reserve = sum(executor.caps[m]['refit_classifier_fits'] for m in members[i+1:])
        model,_ = executor.worker(refit,name,directory/f'{prefix}refit_component_{i}.joblib',timeout,reserve=reserve)
        models.append(model)
    family = members[0] if len(members)==1 else 'equal_soft_vote'
    model = models[0] if len(models)==1 else SoftVoteModel(task.kind,models,np.asarray(metric.labels))
    destination = directory/(prefix + 'selected_model.joblib')
    joblib.dump(dict(kind=task.kind,candidate=family,model=model),destination)
    if not challenger:
        log.write_json('task_spec.json',frozen_spec(task,spec,metric))
    selection = dict(selected_experiment='stop_'+identity(members)[:12],family=family,
                     ensemble_components=members if len(members)>1 else [],single_candidate=members[0],
                     primary_metric=metric.name,selection_metric='training folds',selection_score=choice['utility'],
                     fold_scores=choice['evidence']['scores'],test_accessed=False,refit=True,training_rows=len(task.y_train),
                     model_format='candidate_bundle_v1',model_file=destination.name,model_sha256=sha(destination),
                     task_spec_sha256=sha(directory/'task_spec.json'))
    if provisional_total is not None:
        probe=take(task.X_train,np.linspace(0,len(task.y_train)-1,min(8,len(task.y_train)),dtype=int))
        restored=joblib.load(destination)['model']
        np.testing.assert_array_equal(predict_with_model(task.kind,model,probe),predict_with_model(task.kind,restored,probe))
        if metric.requires_scores:
            before,after=[probability_with_model(task.kind,m,probe) for m in (model,restored)]
            np.testing.assert_array_equal(before[0],after[0])
            np.testing.assert_allclose(before[1],after[1])
        selection.update(refit=False,refit_scope='all search rows',delivery_status='partial',contract_unresolved=False,
                         requested_planner_completed=False,partial_reason='Optional experiment and final training pending',
                         full_training_rows=provisional_total,model_replay_rows=len(probe))
    log.write_json(prefix + 'final_selection.json',selection)
    if provisional_total is not None:
        log.write_json('final_selection.json',selection)
    return selection


def api_accounting(flash, log):
    import json
    entries = [json.loads(s) for s in log.usage_path.read_text(encoding='utf-8').splitlines()]
    unknown = sum(e.get('usage_known') is False for e in entries)
    known = flash.spent if flash else 0
    return dict(new_api_requests=flash.request_count if flash else 0, known_course_cost_rmb=known,
                unresolved_api_requests=unknown, complete_run_course_cost_rmb=None if unknown else known)


def train(package, directory, **options):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError('Training requires a new run directory: '+str(directory))
    try:
        return _train(package,directory,**options)
    except FileExistsError:
        raise
    except Exception as error:
        if interrupted(directory,type(error).__name__+': '+str(error)[:500]):
            return 0
        raise


def _train(package, directory, *, fit_limit=10, seed=18643, wall=1200., timeout=300., flash=None,
          flash_setup_issue=None, self_review=False, exploration_rule=None,feedback_reflection=None,reflection_history=None,
          interpretation='fallback',prediction_package=None,semantic_artifact_from=None):
    from . import feedback_reflection as feedback
    prior_feedback = feedback.load_memory(reflection_history)
    if type(fit_limit) is not int or not 2<=fit_limit<=16 or not all(math.isfinite(t) and t>0 for t in (wall,timeout)):
        raise ValueError('Need 2–16 physical fits including refits and positive finite deadlines')
    directory = Path(directory).resolve()
    rule = load_rule(exploration_rule) if exploration_rule is not None else None
    directory.mkdir(parents=True,exist_ok=False)
    log = RunLogger(directory)
    log.initialize_usage_ledger()
    if flash_setup_issue:
        log.event('planner_preflight_unavailable', reason=flash_setup_issue)
    start = time.monotonic()
    from .recovery import RecoveryController
    recovery=RecoveryController(log,client=flash,fit_limit=fit_limit,deadline=start+wall)
    if flash is not None:
        flash.set_deadline(start+wall)
    manifest = dict(policy='anonymous-sequential-v1',self_review=self_review,fit_limit=fit_limit,seed=seed,wall_seconds=wall,
                    timeout_seconds=timeout,controller='bounded-flash-and-local' if flash else 'local evidence policy',
                    training_only=True,source_files=sources(),task_interpretation=interpretation,
                    semantic_artifact_reuse=semantic_artifact_from is not None)
    if rule is not None:
        log.write_json('exploration_rule.json',rule)
        manifest['exploration_rule_digest'] = identity(rule)
    if feedback_reflection:
        manifest.update(feedback_reflection=feedback_reflection,reflection_history_digest=identity(prior_feedback))
    log.write_json('run_manifest.json',manifest)
    from .partial_classification import record_input_failure
    from .format_intake import DECODE_ERRORS
    from .task_opening import MODES as opening_modes, TaskOpening
    from .opening_evidence import instructions, verified_windows
    opening_enabled=feedback_reflection in opening_modes
    if opening_enabled and flash is not None:
        ceiling=3 if interpretation in ('proactive','auto') else 1
        flash.max_requests=ceiling if flash.max_requests is None else min(flash.max_requests,ceiling)
    opening=None
    phase='training schema'
    task = None
    try:
        from .semantic_contract import resolve_training_task
        if interpretation=='auto' and flash_setup_issue:
            from .format_sources import UnsupportedFormat
            raise UnsupportedFormat('Automatic semantic review unavailable; deployment preflight did not pass')
        intake_client=flash if interpretation=='proactive' or opening_enabled or not feedback_reflection else None
        task=resolve_training_task(package,intake_client,log,mode=interpretation,prediction_package=prediction_package,
                                   artifact_from=semantic_artifact_from,recovery=recovery)
        if opening_enabled:
            task,lines,origin=instructions(package,task)
        spec, metric = TaskSpec.from_task(task).record(), MetricSpec.for_task(task)
        log.event('selection_metric_resolved', **spec['goal_resolution']['fields']['primary_metric'])
    except DECODE_ERRORS as error:
        recovery.exhausted('Task interpretation did not yield a verified contract within the available recovery route and allowance')
        return record_input_failure(package,directory,log,error,phase,
                                    api_usage=api_accounting(flash,log), resolved_task=task)
    recovery.advance('training_method',goal=dict(task_kind=task.kind,metric=metric.record()))
    from .proposal_value import MODES as proposal_modes, DELIVERED_MODES, ProposalStudy, split_search
    pilot=None
    core_limit=min(10,fit_limit) if opening_enabled else fit_limit
    if feedback_reflection in proposal_modes+DELIVERED_MODES:
        pilot=ProposalStudy(feedback_reflection,directory,log,flash,start,wall)
        core_limit=fit_limit-min(5,fit_limit//3)
        try:
            inner,folds,sealed,split_record=split_search(task,seed,partitions)
        except ValueError:
            standby(task,spec,metric,directory,log)
            raise
        log.write_json('proposal_partition.json',split_record)
        checkpoint=standby(inner,spec,metric,directory,log)
        pilot.snapshot(checkpoint,0)
    else:
        checkpoint = standby(task,spec,metric,directory,log)
        folds=verified_windows(task,seed,partitions) if opening_enabled else partitions(task,seed)
    payloads = {}
    for name,(fit,held,resource) in folds.items():
        path = directory/(name+'.joblib')
        labels=task.y_train[held]
        joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,fit),y_train=task.y_train[fit],
                         X_search=take(task.X_train,held),y_search=labels,metric_spec=metric.record(),
                         pair_semantics=declared_semantics(task),capture_probabilities=True),path)
        payloads[name]=dict(path=path,labels=labels,sha256=sha(path))
        log.write_json(name+'_split.json',dict(fit=fit.tolist(),held=held.tolist(),resource=resource,
                       row_sha256=fingerprint(held),label_sha256=fingerprint(labels),payload_sha256=sha(path)))
    fit,_,_ = folds['primary']
    profile=training_facts(TrainingTask(task.kind,take(task.X_train,fit),task.y_train[fit],task.spec))
    profile['computed_from']='primary fitting rows only'
    log.write_json('training_profile.json',profile)
    log.write_json('task_contract.json',spec)
    with installed_methods():
        widths=task.X_train.shape[1] if task.kind=='numeric_tabular' else None
        static=registered_capabilities(spec,len(task.y_train),numeric_kernel_width=widths)
        generator=MethodSpace(spec,profile,directory)
        executor=Executor(directory,{},metric,start,wall,timeout,fit_limit,log)
        history,decisions,blocked=[],[],set()
        observed={f:{} for f in folds}
        flash_available=flash is not None and not feedback_reflection
        reflection = None
        scheduled = None
        if feedback_reflection=='scheduled-flash':
            from .scheduled_review import ScheduledReview
            scheduled=ScheduledReview(log,flash,start+wall)
        from .planner_review import PlannerReview
        review = PlannerReview() if self_review else None
        if review is not None:
            log.write_json('policy_review.json',review.record())
        try:
            if opening_enabled:
                opening=TaskOpening(feedback_reflection,task,metric,folds,lines,origin,log,flash,start,wall)
                opening.launch()
            while True:
                recovery.advance('training_method',accounting=executor.accounting(),artifacts=['selected_model.joblib','task_spec.json'])
                caps, generation=generator.refresh(history,static)
                from .output_recipe import needs_scores
                if needs_scores(task.spec.get('output_recipe')):
                    excluded=[name for name,cap in caps.items() if not cap.get('supports_probabilities')]
                    caps={name:cap for name,cap in caps.items() if cap.get('supports_probabilities')}
                    if excluded:
                        log.event('output_capability_filter',requires='predict_proba',excluded=excluded,scoring_metric=metric.name)
                executor.caps=caps
                remaining=core_limit-executor.accounting()['attempted_classifier_fits']
                menu=actions(caps,profile,history,observed,metric,list(folds),remaining,blocked)
                if not menu:
                    raise ValueError('No admissible action with known fit cost or deployable observation')
                base_first = menu[0]
                if rule is not None:
                    menu = apply_rule(menu, rule['weights'], len(observed['primary']))
                initial_first = menu[0]
                if review is not None:
                    menu=review.menu(menu)
                    rule_first = menu[0]
                    if self_review in ('active', 'protected'):
                        menu = review.seek_evidence(menu, remaining, caps)
                choice,source=menu[0],'local evidence policy'
                if pilot is not None:
                    pilot.launch(choice,menu,generator,caps,observed,history,profile,metric,
                                 remaining+fit_limit-core_limit)
                if feedback_reflection in ('control','local','flash') and reflection is None and feedback.eligible(history,menu):
                    packet=feedback.evidence_packet(menu,caps,profile,history,metric.name,remaining,prior_feedback)
                    reflection=feedback.decide(feedback_reflection,packet,flash,log)
                    advice=reflection['decision']
                    choice=menu[advice['option']]
                    source=feedback_reflection+' feedback reflection'
                    reflection['step']=len(decisions)+1
                    log.write_json('reflection.json',reflection)
                if flash_available and flash.request_count<flash.max_requests and len(menu)>1:
                    try:
                        choice=choose_with_flash(flash,log,menu,caps,profile,metric,history,remaining,
                                                 recovery_context=recovery.context())
                        source='DeepSeek planner'
                    except (ValueError,RuntimeError,TimeoutError,OSError) as err:
                        flash_available=False
                        log.event('planner_unavailable',error_type=type(err).__name__,reason=str(err)[:320])
                decisions.append(dict(step=len(decisions)+1,source=source,choice=choice,remaining_fits=remaining,
                     observed_trials=[r['trial'] for r in history],registry=generation['file'],
                     registry_sha256=generation['sha256'],alternatives=menu))
                if rule is not None:
                    decisions[-1]['inherited_rule'] = dict(digest=manifest['exploration_rule_digest'],
                        prior_primary=len(observed['primary']), changed_first=any(
                            base_first[k] != initial_first[k] for k in ('action','members','split')))
                if review is not None:
                    keys = ('action','members','split')
                    decisions[-1].update(policy_generation=review.generation,
                        initial_policy_first={k:initial_first[k] for k in keys},
                        review_changed_first=any(initial_first[k]!=menu[0][k] for k in keys))
                    if self_review in ('active', 'protected'):
                        decisions[-1].update(first_before_information={k:rule_first[k] for k in keys},
                            information_changed_first=any(rule_first[k]!=menu[0][k] for k in keys))
                if self_review == 'protected' and pilot is None and choice['action'] == 'stop':
                    decisions[-1]['after_refit'] = 'stop_review.json'
                log.write_json('decisions.json',decisions)
                if choice['action']=='stop':
                    break
                name,fold=choice['members'][0],choice['split']
                destination=directory/f'trial_{len(history)+1}.joblib'
                monitor=dict(trial=len(history)+1,candidate=name,split_id=fold)
                payload=payloads[fold]
                if scheduled is not None:
                    scheduled.before(choice,monitor['trial'],checkpoint,observed,folds,caps,metric,remaining)
                if review is not None:
                    review.before_fit(monitor['trial'],choice)
                    log.write_json('policy_review.json',review.record())
                try:
                    _,facts=executor.worker(payload['path'],name,destination,timeout,
                                            reserve=choice['evidence']['reserved_refits'])
                except RuntimeError as error:
                    if executor.accounting()['unknown_failed_attempts']:
                        raise
                    blocked.add(name)
                    recovery.failure('training_method',dict(code=type(error).__name__,
                        message='Selected candidate failed in the worker; blocked for this run. Choose another executable method.',
                        candidate=name),action=dict(candidate=name,split=fold))
                    if review is not None:
                        review.ticket=None
                        log.write_json('policy_review.json',review.record())
                    history.append(dict(monitor,status='failed',error=str(error)[-800:]))
                    log.write_json('history.json',history)
                    if scheduled is not None:
                        scheduled.after(monitor['trial'],checkpoint,observed,task,metric,history,payloads['primary'],directory,failed=True)
                    continue
                bundle=joblib.load(destination)
                recovery.accepted('method_executed',stage='training_method')
                observation=planner_observation(bundle,facts,payload,metric)
                if review is not None:
                    review.observed(monitor['trial'],observation['score'])
                    log.write_json('policy_review.json',review.record())
                if reflection and reflection['step']==monitor['trial'] and reflection['decision']['verdict']=='experiment':
                    threshold=reflection['decision']['minimum_score']
                    reflection['observed_check']=dict(trial=monitor['trial'], fold=fold, score=observation['score'],
                                                      minimum_score=threshold, passed=observation['score']>=threshold)
                    log.write_json('reflection.json',reflection)
                if fold == 'primary':
                    try:
                        checkpoint = promote(task,metric,checkpoint,monitor['trial'],name,destination,payload,observation,log)
                    except (ValueError, OSError, RuntimeError, AssertionError) as error:
                        log.event('checkpoint_not_promoted',trial=monitor['trial'],reason=str(error)[:320])
                bank=observed[fold]
                best=max(bank.values(),key=lambda o:o['score'],default=None)
                paired=None
                if best is not None:
                    hit=observation['predictions']==payload['labels']
                    prior=best['predictions']==payload['labels']
                    paired=dict(newly_correct=int(np.sum(hit & ~prior)),newly_wrong=int(np.sum(prior & ~hit)),
                                both_wrong=int(np.sum(~prior & ~hit)))
                bank[name]=observation
                history.append(dict(monitor,status='completed',primary_metric=metric.name,
                    train_score=facts['train_score'],search_score=facts['search_score'],search_accuracy=facts['search_accuracy'],
                    class_recall=facts['class_recall'],paired_errors=paired,training_seconds=facts['training_seconds'],
                    resolved_configuration=facts['resolved_configuration']))
                log.write_json('history.json',history)
                if scheduled is not None:
                    scheduled.after(monitor['trial'],checkpoint,observed,task,metric,history,payloads['primary'],directory)
                if opening is not None:
                    opening.observe(monitor['trial'],observed)
                if pilot is not None:
                    pilot.snapshot(checkpoint,sum(r['status']=='completed' for r in history))
            if pilot is not None:
                if feedback_reflection in DELIVERED_MODES:
                    cost=sum(executor.caps[n]['refit_classifier_fits'] for n in choice['members'])
                    remaining=fit_limit-executor.accounting()['attempted_classifier_fits']
                    if remaining>=2*cost:
                        ready=freeze_model(inner,spec,metric,choice,executor,directory,log,timeout,provisional_total=len(task.y_train))
                        pilot.protected_selection=ready
                        pilot.snapshot(ready,sum(r['status']=='completed' for r in history))
                        pilot.record.update(protected_model=ready['model_sha256'],protected_rows=ready['training_rows'],
                                            protected_seconds=pilot.snapshots[-1]['elapsed'])
                        pilot.save()
                choice=pilot.tail(task,spec,metric,choice,checkpoint,history,observed,payloads,sealed,generator,
                                  executor,profile,timeout,blocked)
                log.write_json('pre_refit_proposal_selection.json',choice)
                if sum(executor.caps[n]['refit_classifier_fits'] for n in choice['members'])>fit_limit-executor.accounting()['attempted_classifier_fits']:
                    raise ValueError('Retain the published checkpoint; selected refit exceeds remaining fits')
            selection=freeze_model(task,spec,metric,choice,executor,directory,log,timeout)
            if self_review == 'protected' and pilot is None:
                from .protected_stop import review_stop
                selection, choice = review_stop(task, spec, metric, menu, choice, selection, executor,
                                                directory, log, timeout, observed, payloads)
            if opening is not None:
                selection,choice=opening.after_delivery(spec,choice,selection,history,observed,payloads,profile,executor,timeout,blocked)
            if feedback_reflection == 'conditional-flash':
                from .conditional_review import review_delivery
                selection, choice = review_delivery(task, spec, metric, selection, choice, executor, directory,
                    log, timeout, observed, payloads, folds, flash, prior_feedback)
            if scheduled is not None:
                scheduled.finish(selection)
            if pilot is not None:
                pilot.snapshot(selection,sum(r['status']=='completed' for r in history)+pilot.tail_completed,final=True)
            outcome=dict(status='completed',model_available=True,winner=selection['family'],
                         members=choice['members'],training_rows=len(task.y_train),training_only=True)
            recovery.advance('model_exported',accounting=executor.accounting(),
                             artifacts=['selected_model.joblib','task_spec.json','final_selection.json'],status='training_completed')
        except Exception as error:
            outcome=dict(status='failed',model_available=False,error=type(error).__name__+': '+str(error))
            recovery.exhausted('Training did not complete; retain separately verified checkpoints if available')
            raise
        finally:
            if opening is not None:
                opening.finish()
            if pilot is not None:
                pilot.finish()
            if 'outcome' in locals():
                log.write_json('run_outcome.json',dict(outcome,**executor.accounting(),wall_seconds=time.monotonic()-start,
                               **api_accounting(flash,log),formal_test_accessed=False))
    return 0
