"""Replay a frozen primary decision and collect only the two independent confirmation folds."""
import argparse
import os
from pathlib import Path
import time

import joblib

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env, read, require, sha
from superagent.ab_run import Executor, registered_capabilities
from superagent.analogy_review import blend_counts, nominate, promoted
from superagent.autonomous_run import installed_methods, planner_observation
from superagent.events import RunLogger
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.preference_analogy import derived_candidate


PLAN=ROOT/'research/paired_encoder_search_1127/analogy_near_tie_protocol.json'


def run(root):
    out=root/'analogy-close-confirm'
    out.mkdir(exist_ok=False)
    agent=root/'analogy-raw/revised/run/agent'
    prefix=read(agent/'analogy_review.json')
    require(prefix['status']=='retained_primary' and prefix['locked_weight']==0 and prefix['extra_fits']==1,
            'Unexpected earlier stop')
    confirmations=('confirmation_1','confirmation_2')
    require(all(not prefix['already_observed_folds'][k] for k in confirmations),'Confirmation fold was used already')
    source=read(agent/'final_selection.json')
    base_name=source['single_candidate']
    baseline=agent/source['model_file']
    require(sha(baseline)==prefix['incumbent_model']==source['model_sha256'],'Changed deployed incumbent')
    weights=root/'discovery-library'
    physical=out/'physical.jsonl'
    settings=env(weights,physical)
    for k in tuple(settings):
        if k.startswith('SUPERAGENT_') and (k.endswith('_WEIGHTS') or k in ('SUPERAGENT_WEIGHTS_DIR','SUPERAGENT_PLANNER_KEY_FILE')):
            settings.pop(k)
    settings.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/'feature_bank'),HF_HUB_OFFLINE='1')
    os.environ.update(settings)
    activate_resource_directory(weights)
    spec=read(agent/'task_contract.json')
    metric=MetricSpec.from_payload(dict(metric_spec=spec['metric']))
    panels={}
    for fold in ('primary',*confirmations):
        payload=agent/(fold+'.joblib')
        stamp=read(agent/(fold+'_split.json'))
        require(stamp['payload_sha256']==sha(payload),'Changed original fold')
        panels[fold]=dict(path=payload,sha256=sha(payload),labels=joblib.load(payload)['y_search'])
    with installed_methods():
        candidate=derived_candidate(joblib.load(baseline)['model'])
        require(candidate.family==prefix['candidate'] and candidate.params==prefix['derived_configuration'],'Different executable adviser')
        name=candidate.family
        cap=dict(id=name,kind='pair_text',params=candidate.params,estimator='preference_analogy',origin='b_port',
                 resource='local_frozen_encoder_cpu',classifier_fits=1,refit_classifier_fits=1)
        caps=registered_capabilities(spec,source['training_rows'])
        caps[name]=cap
        def observation(path,fold):
            trial=joblib.load(path)
            return planner_observation(trial,trial['metrics'],panels[fold],metric)
        history=read(agent/'history.json')
        row=next(v for v in history if v['candidate']==base_name and v['split_id']=='primary' and v['status']=='completed')
        first_base=agent/f"trial_{row['trial']}.joblib"
        first_near=agent/f'analogy_primary_{name}.joblib'
        truth=panels['primary']['labels']
        decision=nominate(observation(first_base,'primary'),observation(first_near,'primary'),truth)
        require(decision['scores']==prefix['primary']['scores'],'Cannot reproduce the exact earlier primary comparison')
        lock(out/'nomination.json',dict(protocol=sha(PLAN),source=source_identity(ROOT),initial_decision=sha(agent/'analogy_review.json'),
             evidence=[sha(first_base),sha(first_near)],remaining_unmeasured=list(confirmations),nomination=decision))
        scores=[]
        if decision['weight']>0:
            executor=Executor(out,caps,metric,time.monotonic(),1200,300,4,RunLogger(out),restrictions=spec.get('method_restrictions',()))
            for i,fold in enumerate(confirmations):
                pair=[]
                for j,member in enumerate((base_name,name)):
                    path=out/f'{fold}_{member}.joblib'
                    executor.worker(panels[fold]['path'],member,path,300,reserve=3-2*i-j)
                    pair.append(observation(path,fold))
                truth2=panels[fold]['labels']
                before,after=blend_counts(*pair,truth2,[0.,decision['weight']])
                scores.append(dict(fold=fold,rows=len(truth2),row_key=pair[0]['row_key'],anchor=before,chosen=after,gain=after-before))
                lock(out/(fold+'_decision.json'),scores[-1])
                print(fold,before,after,flush=True)
            require(executor.accounting()['attempted_classifier_fits']==len(physical.read_text().splitlines())==4,'Unknown physical fits')
        passed=bool(decision['weight'] and promoted(decision['primary_gain'],scores,primary_rows=len(truth)))
    result=dict(protocol=sha(PLAN),nomination=sha(out/'nomination.json'),decision=decision,confirmations=scores,advance=passed,
         additional_classifier_fits=2*len(scores),reused_prefix_fits=prefix['before_fits']+prefix['extra_fits'],
         new_model_requests=0,course_validation_opened=False,H2_opened=False,formal_test_opened=False)
    lock(out/'results.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
