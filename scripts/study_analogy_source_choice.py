"""Select a retrieval representation by paired improvements on training rows."""
import argparse
import os
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT,env,read,require,sha
from superagent.ab_run import Executor
from superagent.analogy_review import blend_counts,incumbent_evidence,nominate,promoted
from superagent.autonomous_run import planner_observation
from superagent.events import RunLogger
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.preference_analogy import derived_candidate


PLAN=ROOT/'research/paired_encoder_search_1127/analogy_source_choice_protocol.json'


def run(root):
    out=root/'analogy-source-choice'
    out.mkdir(exist_ok=False)
    base=root/'analogy-raw/original/run/agent'
    legacy=root/'analogy-close-raw/close/run/agent'
    weights=root/'discovery-library'
    physical=out/'physical.jsonl'
    settings=env(weights,physical)
    for k in list(settings):
        if k.startswith('SUPERAGENT_') and (k.endswith('_WEIGHTS') or k in ('SUPERAGENT_WEIGHTS_DIR','SUPERAGENT_PLANNER_KEY_FILE')):
            settings.pop(k)
    os.environ.update(settings)
    os.environ['SUPERAGENT_PAIRED_FEATURE_CACHE']=str(root/'feature_bank')
    activate_resource_directory(weights)
    election=read(base/'finalist_result.json')
    old=read(legacy/'analogy_review.json')
    require(old['status']=='accepted','Missing independently accepted incumbent source')
    fitted=joblib.load(base/read(base/'final_selection.json')['model_file'])['model']
    members=election['selected']
    history=read(base/'history.json')
    infos={}
    for n,m in zip(members,fitted.models,strict=True):
        c=derived_candidate(m)
        if c:
            infos[c.family]=(n,c)
    require(len(infos)==2 and old['candidate'] in infos,'Two distinct source representations required')
    chosen_before=old['candidate']
    spec=read(base/'task_contract.json')
    metric=MetricSpec.from_payload(dict(metric_spec=spec['metric']))
    caps={key:dict(id=key,params=c.params,estimator='preference_analogy',resource='local_frozen_encoder_cpu',origin='b_port',
                   classifier_fits=1,refit_classifier_fits=1) for key,(_,c) in infos.items()}
    executor=Executor(out,caps,metric,time.monotonic(),1800,540,3,RunLogger(out),restrictions=spec.get('method_restrictions',()))
    locks={};panels={}
    for fold in ('primary','confirmation_1','confirmation_2'):
        path=base/(fold+'.joblib')
        digest=sha(path)
        require(digest==read(legacy/(fold+'_split.json'))['payload_sha256']==read(base/(fold+'_split.json'))['payload_sha256'],'Row identities differ')
        entries={n:next(r['trial'] for r in history if r['candidate']==n and r['split_id']==fold and r['status']=='completed') for n in members}
        trials=[base/f'trial_{entries[n]}.joblib' for n in members]
        old_path=legacy/f'analogy_{fold}_{chosen_before}.joblib'
        locks[fold]=dict(payload=digest,fitted_members=[sha(v) for v in trials],old_adviser=sha(old_path))
        panels[fold]=(path,trials,old_path)
    lock(out/'inputs.json',dict(plan=sha(PLAN),actual_election=sha(base/'finalist_result.json'),previous_election=sha(legacy/'analogy_review.json'),
         supported_sources={k:dict(member=n,params=c.params) for k,(n,c) in infos.items()},folds=locks))
    def panel(fold):
        payload,paths,old_path=panels[fold]
        truth=joblib.load(payload)['y_search']
        stamp=dict(sha256=locks[fold]['payload'],labels=truth)
        def observation(path):
            item=joblib.load(path)
            return planner_observation(item,item['metrics'],stamp,metric)
        anchor=incumbent_evidence([observation(p) for p in paths],[1/len(paths)]*len(paths),metric)
        return payload,truth,anchor,observation,observation(old_path)
    def actual(name,fold,reserve):
        path,truth,anchor,observe,prior=panel(fold)
        if name==chosen_before:
            candidate=prior
        else:
            artifact=out/f'{fold}_{name}.joblib'
            executor.worker(path,name,artifact,540,reserve=reserve)
            candidate=observe(artifact)
        return truth,anchor,candidate,prior
    nominations={}
    for name in infos:
        truth,base_obs,near,_=actual(name,'primary',2)
        nominations[name]=nominate(base_obs,near,truth)
    eligible=[n for n,d in nominations.items() if d['weight']>0]
    selected=min(eligible,key=lambda n:(-nominations[n]['primary_gain'],n!=chosen_before,n)) if eligible else None
    lock(out/'source_choice.json',dict(inputs=sha(out/'inputs.json'),nominations=nominations,chosen=selected,prior=chosen_before))
    comparisons=[]
    replace=False
    if selected and selected!=chosen_before:
        proposed=nominations[selected]
        for i,fold in enumerate(('confirmation_1','confirmation_2')):
            truth,anchor,challenger,old_adviser=actual(selected,fold,1-i)
            n,k=blend_counts(anchor,challenger,truth,[0,proposed['weight']])
            prior_count=blend_counts(anchor,old_adviser,truth,[nominations[chosen_before]['weight']])[0]
            comparisons.append(dict(fold=fold,rows=len(truth),anchor=n,candidate=k,existing_adviser=prior_count,
                                    gain=k-n,vs_existing=k-prior_count))
            lock(out/(fold+'.json'),comparisons[-1])
        replace=bool(promoted(proposed['primary_gain'],comparisons,primary_rows=nominations[selected]['paired']['rows'])
                     and min(v['vs_existing'] for v in comparisons)>=0 and sum(v['vs_existing'] for v in comparisons)>0)
    fits=executor.accounting()['attempted_classifier_fits']
    require(fits==executor.accounting()['completed_classifier_fits']==len(physical.read_text().splitlines())<=3,'Unmatched supervised fits')
    result=dict(lock=sha(out/'source_choice.json'),primary=nominations,prior=chosen_before,nominee=selected,confirmation=comparisons,
                promote_different_source=replace,new_index_fits=fits,new_base_fits=0,new_model_requests=0,
                designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/'results.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
