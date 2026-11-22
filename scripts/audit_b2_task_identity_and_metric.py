"""Replay identical real observations under renamed inputs and independently rescored objectives."""
import argparse
from dataclasses import replace
import math
import os
from pathlib import Path

import joblib

from scripts import run_b1_ab_acceptance as record
from scripts.replay_b1_observed_actions import key, trusted_input
from superagent import ab_run, ab_text_recipes
from superagent.ab_scheduler import next_action, selection_evidence
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.scheduler_bridge import fingerprint
from tools.ab_registered_rows import content_digest


def read_renamed(case, out, task, weights):
    registrations = {}
    src_task = load_development_task(case/'train.zip')
    identity = content_digest(src_task.X_train,src_task.y_train)
    for label in ('anonymous','wrong_hint'):
        directory = out/(task+'_'+label)
        directory.mkdir()
        name = 'dataset.zip' if label=='anonymous' else 'satellite-image-classification.zip'
        path = directory/name
        os.link(case/'train.zip',path)
        changed = load_development_task(path)
        if (changed.X_val is not None or changed.y_val is not None or content_digest(changed.X_train,changed.y_train)!=identity
                or ab_run.sha(path)!=ab_run.sha(case/'train.zip')):
            raise ValueError('Filename-only negative control changed the actual archive, split or rows')
        contract = TaskSpec.from_task(changed).record()
        menu = ab_run.registered_capabilities(contract,len(changed.y_train),'full')
        registrations[label]=(contract,menu)
    return registrations


def menus(caps,policy):
    return {k:v for k,v in caps.items() if policy!='original' or k not in ab_text_recipes.ROUTES}


def bind_metric(run,work,original,metric):
    result = {}
    for split,receipt in original.items():
        path = run/(split+'.joblib')
        if ab_run.sha(path)!=receipt['evaluation_payload_sha256']:
            raise ValueError('Trusted real split payload changed')
        data = joblib.load(path)
        dest = work/(split+'.joblib')
        # All training and held rows remain in their original order. This is a new evaluation identity.
        joblib.dump(dict(data,metric_spec=metric.record()),dest)
        result[split]=dict(receipt,evaluation_payload_sha256=ab_run.sha(dest))
    return result


def remeasure(run,history,evidence,trusted,metric):
    by_trial = {r['trial']:r for r in evidence}
    projected = {}
    for r in history:
        if r['status']!='completed' or r['trial'] not in by_trial:
            raise ValueError('Replay requires complete published worker observations')
        raw = joblib.load(run/f"trial_{r['trial']}.joblib")['metrics']
        if raw['resolved_configuration']['id']!=r['candidate'] or raw['train_metrics']!=r['train_metrics']:
            raise ValueError('Raw worker training metrics differ from history')
        proof = by_trial[r['trial']]
        split = r['split_id']
        held = metric.evaluate(trusted[split]['labels'],proof['predictions'])
        for name in ('accuracy','macro_f1'):
            if not math.isclose(held[name],r['search_metrics'][name],abs_tol=1e-12,rel_tol=0):
                raise ValueError('Recomputed worker search metrics differ on actual labels')
        train = dict(raw['train_metrics'],primary_metric=metric.name,score=raw['train_metrics'][metric.name])
        fields = {k:r[k] for k in ('trial','candidate','params','status','split_id','classifier_fits')}
        fields.update(primary_metric=metric.name,metric_spec=metric.record(),search_score=held['score'],train_score=train['score'],
                      train_metrics=train,search_metrics=held)
        projected[r['trial']]=fields
    return projected


def projected_state(state,contract,observations,trusted):
    return dict(state,task=contract,history=[observations[r['trial']] for r in state['history']],
        prediction_evidence=[dict(r,evaluation_payload_sha256=trusted[r['split_id']]['evaluation_payload_sha256'])
                             for r in state['prediction_evidence']])


def audit_case(case,folder,aliases,pre,proto,expected,policy):
    folder.mkdir()
    run=case/'run'
    recorded=record.read(run/'actions.json')
    if recorded!=expected['actions']:
        raise ValueError('Frozen actions changed')
    capref = pre['base_sha256' if policy=='original' else 'expanded_sha256']
    entry = dict(files={'train.zip':pre['train_sha256']},catalog_sha256=capref,splits=pre['splits'])
    caps,trusted = trusted_input(case,entry,proto['source'])
    dev=load_development_task(case/'train.zip')
    old=MetricSpec.for_task(dev)
    new_name='macro_f1' if old.name=='accuracy' else 'accuracy'
    changed_spec = dict(dev.spec,metric=new_name,goal_sources={**dev.spec.get('goal_sources',{}),'primary_metric':'declared_spec'})
    changed=replace(dev,spec=changed_spec)
    new=MetricSpec.for_task(changed)
    old_contract,new_contract=[TaskSpec.from_task(d).record() for d in (dev,changed)]
    new_caps=menus(ab_run.registered_capabilities(new_contract,len(dev.y_train),'full'),policy)
    if old_contract!=pre['contract'] or list(new_caps.items())!=list(caps.items()):
        raise ValueError('Declared metric altered the executable menu or prior contract')
    callback=ab_text_recipes.propose if policy=='composed' else None
    def choose(state,tr):
        return next_action(state,capabilities=caps,trusted_partitions=tr,on_local_recipe=callback)
    history=record.read(run/'history.json')
    evidence=record.read(run/'prediction_evidence.json')
    alternate_trusted=bind_metric(run,folder,trusted,new)
    observed=remeasure(run,history,evidence,trusted,old)
    rescored=remeasure(run,history,evidence,trusted,new)
    effects=[]
    aliases_checked=0
    final_old=final_new=None
    for index,action in enumerate(recorded,1):
        state=record.read(run/f'state_{index}.json')
        original_receipt=record.read(run/'bounded_recipes'/f'step_{index}.json')
        if fingerprint(state)!=original_receipt['state_sha256'] or original_receipt['action']!=action:
            raise ValueError('Source state fingerprint differs from real execution')
        for contract,alias_menu in aliases.values():
            if contract!=state['task'] or list(menus(alias_menu,policy).items())!=list(caps.items()):
                raise ValueError('Changing an exterior filename changed an inner contract or menu')
            if choose(dict(state,task=contract),trusted)!=action:
                raise ValueError('Changing an exterior filename changed a real local decision')
            aliases_checked+=1
        old_state=projected_state(state,old_contract,observed,trusted)
        n_state=projected_state(state,new_contract,rescored,alternate_trusted)
        if choose(old_state,trusted)!=action:
            raise ValueError('Reduced history failed to reproduce the original complete action')
        counter=choose(n_state,alternate_trusted)
        if counter['action']=='stop' and 'reconciliation' in counter['reason']:
            raise ValueError('Rescored row-level evidence did not reconcile')
        if key(counter)!=key(action):
            effects.append(dict(step=index,observed=key(action),alternative=key(counter),
                                observed_refs=action['evidence_trials'],alternative_refs=counter['evidence_trials']))
        if index==len(recorded):
            final_old=selection_evidence(old_state,trusted_partitions=trusted)['incumbent']
            final_new=selection_evidence(n_state,trusted_partitions=alternate_trusted)['incumbent']
    if final_old!=expected['selected']:
        raise ValueError('Original incumbent did not reproduce')
    return dict(states=len(recorded),filename_decisions_checked=aliases_checked,catalog_sha256=capref,verified_trials=len(history),
        original_metric=old.name,alternative_metric=new.name,changed_actions=effects,old_incumbent=final_old,
        alternative_incumbent_on_same_history=final_new,alternative_payloads={k:v['evaluation_payload_sha256'] for k,v in alternate_trusted.items()})


def study(args):
    out,origin=args.output.resolve(),args.previous.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    signed=record.ROOT/'research/b2_bounded_text_composition'
    for n in ('preflight','results','protocol'):
        if ab_run.sha(origin/(n+'.json'))!=ab_run.sha(signed/(n+'.json')):
            raise ValueError('B2 parent no longer matches repository evidence')
    proto,pre,result=[record.read(signed/(n+'.json')) for n in ('protocol','preflight','results')]
    if record.digest(ab_run.sources())!=proto['source']:
        raise ValueError('B2 implementation changed')
    record.write(out/'protocol.json',dict(batch='B2-04',parent_results_sha256=ab_run.sha(signed/'results.json'),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B2_04_file_identity_metric_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),filename_variants=2,new_fit_limit=0,new_model_api_limit=0))
    try:
        aliases={name:read_renamed(origin/(name+'_original'),out,name,args.weights) for name in pre}
        reports={}
        for case,expected in result['paths'].items():
            task,policy=case.split('_')
            reports[case]=audit_case(origin/case,out/case,aliases[task],pre[task],proto,expected,policy)
            record.write(out/'checkpoint.json',reports)
        if record.trace(out):
            raise ValueError('Unexpected fitting')
        record.write(out/'results.json',reports)
        record.write(out/'completion.json',dict(status='completed',states=sum(r['states'] for r in reports.values()),
            unchanged_filename_actions=sum(r['filename_decisions_checked'] for r in reports.values()),
            changed_after_metric_switch=sum(len(r['changed_actions']) for r in reports.values()),
            changed_end_recommendations=sum(r['old_incumbent']!=r['alternative_incumbent_on_same_history'] for r in reports.values()),
            verified_worker_trials=sum(r['verified_trials'] for r in reports.values()),new_fits=0,new_model_predictions=0,
            new_model_api_requests=0,new_course_rmb=0,outer_grades=0))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    study(parser.parse_args())
