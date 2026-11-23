"""Repeat the opt-in third-split policy on a separate film holdout and fresh inner seed."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.check_b_coverage_feedback import trusted_parts
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, packets, read, save
from scripts.run_b_quote_adaptation import movie_groups
from scripts.run_b_quote_transfer import movie_pairs, normalized
from superagent import ab_adaptive_opening, ab_run
from superagent.ab_methods import JOINT_PREFERENCE
from superagent.data import load_development_task
from superagent.splits import sample_groups

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_single_conflict_replication_1123_zh.md'
SEED=27183


def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    old=read(a.previous/'protocol.json')
    check(ab_run.sha(a.previous/'scoring_only.json')==old['scoring'],'Original film boundary changed')
    rows,films,_=movie_pairs(a.archive)
    groups=movie_groups(rows,films)
    historical=set(read(a.previous/'scoring_only.json')['ids'])
    check(historical=={i for i,g in enumerate(groups) if int(g,16)%5==0},'Cannot reproduce original 450')
    pool=[i for i in range(len(rows)) if i not in historical]
    keys={g:int.from_bytes(hashlib.sha256(('film-conflict-replication-1123-v1:'+g).encode()).digest(),'big')%5
          for g in {groups[i] for i in pool}}
    train=[i for i in pool if keys[groups[i]]!=0];held=[i for i in pool if keys[groups[i]]==0]
    partitions=[train,held,sorted(historical)]
    for j,ids in enumerate(partitions):
        for other in partitions[j+1:]:
            check({groups[i] for i in ids}.isdisjoint(groups[i] for i in other) and
                  {films[i] for i in ids}.isdisjoint(films[i] for i in other) and
                  {normalized(t) for i in ids for t in rows[i][:2]}.isdisjoint(
                      normalized(t) for i in other for t in rows[i][:2]),'An outer film, group or text crosses the boundary')
    check(len(pool)==1744 and len(historical)==450 and train and held,'Unexpected corpus or empty partition')
    packets(rows,train,True,a.out/'anonymous.zip')
    packets(rows,held,False,a.out/'features.zip')
    save(a.out/'scoring_only.json',dict(labels=[rows[i][2] for i in held],groups=[groups[i] for i in held]))
    save(a.out/'protocol.json',dict(plan=ab_run.sha(PLAN),code=ab_run.sha(Path(__file__)),sources=ab_run.sources(),
        training=ab_run.sha(a.out/'anonymous.zip'),features=ab_run.sha(a.out/'features.zip'),scoring=ab_run.sha(a.out/'scoring_only.json'),
        historical_outer=len(historical),available=len(pool),train_rows=len(train),outer_rows=len(held),
        train_films=len({films[i] for i in train}),held_films=len({films[i] for i in held}),
        held_components=len({groups[i] for i in held}),outer_shared_films=0,outer_shared_texts=0,
        seed=SEED,fit_cap_per_path=11,model_api_cap=0))


def native(arm,out,plan,env):
    folder=out/arm
    physical=out/(arm+'.physical.jsonl')
    log=out/(arm+'.log')
    if not any(f.exists() for f in (folder,physical,log)):
        args=['run',str(out/'anonymous.zip'),'--scheduler','ab-adaptive-complement','--portfolio','ab-v1','--planner','local',
              '--preference-catalog',JOINT_PREFERENCE,'--seed',str(SEED),'--cover-untried-encoder','--fit-limit','11',
              '--trial-timeout','300','--time-limit','1200','--run-dir',str(folder)]
        if arm=='new':
            args.append('--resolve-single-conflict')
        rc,_=command(args,log,dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(rc==0,'Incomplete native '+arm+' path; review its recorded fit attempts')
    outcome,sel,m,ledger=[read(folder/n) for n in ('run_outcome.json','final_selection.json','run_manifest.json','classifier_fits.json')]
    usage=[json.loads(line) for line in (folder/'llm_usage.jsonl').read_text().splitlines()]
    n=len(physical.read_text().splitlines())
    check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['source_unchanged']
          and not outcome['formal_test_accessed'] and outcome['new_api_requests']==0 and len(usage)==1 and
          usage[0]['event']=='ledger_initialized' and usage[0]['cost_rmb']==0 and (arm=='new')==m.get('resolve_single_conflict',False)
          and m['seed']==SEED and m['fit_limit']==11 and m['task_archive_sha256']==plan['training'] and
          n==ledger['completed_classifier_fits']==ledger['attempted_classifier_fits']<=11 and
          sel['training_rows']==plan['train_rows'] and sel['model_sha256']==ab_run.sha(folder/'selected_model.joblib'),'Native receipt differs')
    return dict(model=sel['model_sha256'],selected=sel['ensemble_components'] or [sel['family']],fits=n,physical=ab_run.sha(physical))


def check_old(folder,inputs):
    groups=sample_groups(load_development_task(inputs),'shared_text')
    trusted=trusted_parts(folder,groups)
    caps=read(folder/'run_manifest.json')['capabilities']
    actual=read(folder/'actions.json')
    for i, action in enumerate(actual,1):
        state=read(folder/f'state_{i}.json')
        old=ab_adaptive_opening.action(state,caps,trusted,cover_untried_encoder=True)
        if action!=old:
            return dict(identical=False,first_difference=i,new=action,old=old)
    d=ab_adaptive_opening.review(state,trusted)
    new=read(folder/'confirmation.json')['b_scheduler_review']
    return dict(identical=d==new,actions=len(actual),old_selected=d['members'],new_selected=new['members'])


def run(a):
    if not a.out.exists():
        prepare(a)
    p=read(a.out/'protocol.json')
    check(p['plan']==ab_run.sha(PLAN) and p['code']==ab_run.sha(Path(__file__)) and p['sources']==ab_run.sources()
          and p['training']==ab_run.sha(a.out/'anonymous.zip') and p['features']==ab_run.sha(a.out/'features.zip'),'Registered source changed')
    env=offline_environment(a.workspace)
    current=native('new',a.out,p,env)
    comparison=check_old(a.out/'new',a.out/'anonymous.zip')
    frozen_save(a.out/'old_replay.json',comparison)
    shared=comparison['identical']
    previous=current if shared else native('old',a.out,p,env)
    if not shared:
        for split in ('primary','confirmation_1','confirmation_2'):
            check(ab_run.sha(a.out/'new'/(split+'.joblib'))==ab_run.sha(a.out/'old'/(split+'.joblib')),'Partitions differ')
    frozen_save(a.out/'decisions.json',dict(new=current,old=previous,old_reused=shared))
    new=verified_export(a.out/'new',a.out,env,a.out)
    old=new if shared else verified_export(a.out/'old',a.out,env,a.out)
    frozen_save(a.out/'exports.json',dict(new=new,old=old))
    check(ab_run.sha(a.out/'scoring_only.json')==p['scoring'],'New holdout changed')
    truth=read(a.out/'scoring_only.json');y=np.asarray(truth['labels'])
    n=classes(a.out/'new_export_0/rows.csv',new,len(y))==y
    o=n if shared else classes(a.out/'old_export_0/rows.csv',old,len(y))==y
    delta=n.astype(int)-o.astype(int)
    decision=read(a.out/'new/confirmation.json')['b_scheduler_review']
    result=dict(rows=len(y),films=p['held_films'],correct_new=int(n.sum()),correct_old=int(o.sum()),
                gained=int((delta>0).sum()),lost=int((delta<0).sum()),paired=paired(delta,np.asarray(truth['groups'])),
                selected_new=current['selected'],selected_old=previous['selected'],old_reused=shared,
                conflict=decision.get('single_conflict',{}).get('status','untriggered'),actual_new_fits=current['fits']+(0 if shared else previous['fits']),
                model_api_calls=0,new_rmb=0,course_test_or_designated_opened=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(results=ab_run.sha(a.out/'results.json'),protocol=ab_run.sha(a.out/'protocol.json'),
        decisions=ab_run.sha(a.out/'decisions.json'),exports=ab_run.sha(a.out/'exports.json'),old_replay=ab_run.sha(a.out/'old_replay.json'),
        fits=result['actual_new_fits'],model_api_calls=0))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','archive','previous','out'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args())
