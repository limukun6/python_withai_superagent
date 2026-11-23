"""Verify unchanged old title decisions and try third-split conflict resolution in the native film run."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.check_b_coverage_feedback import trusted_parts
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, read
from scripts.run_b_preference_policy_controls import audit
from superagent import ab_adaptive_opening, ab_run, ab_single_conflict
from superagent.ab_methods import JOINT_PREFERENCE
from superagent.data import load_development_task
from superagent.splits import sample_groups

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_single_conflict_1123_zh.md'


def same_titles(inputs, folders):
    groups=sample_groups(load_development_task(inputs/'anonymous.zip'),'shared_text')
    unchanged={}
    for condition, folder in folders.items():
        m=read(folder/'run_manifest.json')
        actions=read(folder/'actions.json')
        trusted=trusted_parts(folder,groups)
        for i, action in enumerate(actions,1):
            state=read(folder/f'state_{i}.json')
            proposal=ab_single_conflict.action(state,m['capabilities'],trusted,cover_untried_encoder=True)
            check(proposal==action,'Historic title action changed for '+condition)
        baseline=read(folder/'confirmation.json')['b_scheduler_review']
        check(ab_single_conflict.review(state,trusted)==ab_adaptive_opening.review(state,trusted)==baseline,
              'Historic title selection changed')
        unchanged[condition]=dict(seed=m['seed'],actions=len(actions),members=baseline['members'],promoted=baseline['promoted'],
                                  source=ab_run.sha(folder/'actions.json'),new_fits=0)
    return unchanged


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    env=offline_environment(a.workspace)
    old=a.quotes/'autonomous'
    held=read(a.quotes/'protocol.json')
    old_policy=read(a.controls/'results.json')
    check(old_policy['task']=='quotes' and held['train_rows']==1744 and
          held['train_packet']==ab_run.sha(a.quotes/'anonymous.zip'),'Unexpected film control')
    frozen_save(a.out/'protocol.json',dict(runner=ab_run.sha(Path(__file__)),plan=ab_run.sha(PLAN),source=ab_run.sources(),
        reference=ab_run.sha(a.controls/'completion.json'),training=held['train_packet'],seed=18643,fit_cap=11,model_api_cap=0))
    no_change=same_titles(a.title_inputs,{'promoted_pair':a.title_promoted,'uncontested':a.title_uncontested})
    frozen_save(a.out/'title_replay.json',no_change)
    folders=a.out/'autonomous'
    physical=a.out/'physical.jsonl'
    if not any(f.exists() for f in (folders,physical,a.out/'run.log')):
        argv=['run',str(a.quotes/'anonymous.zip'),'--scheduler','ab-adaptive-complement','--portfolio','ab-v1',
              '--planner','local','--preference-catalog',JOINT_PREFERENCE,'--cover-untried-encoder','--resolve-single-conflict',
              '--seed','18643','--fit-limit','11','--trial-timeout','300','--time-limit','1200','--run-dir',str(folders)]
        rc,_=command(argv,a.out/'run.log',dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(rc==0,'Native conflict run stopped; inspect durable fit evidence')
    new=audit('dynamic',folders,physical,held['train_packet'],held['train_rows'])
    m=read(folders/'run_manifest.json')
    before=read(old/'run_manifest.json')
    check(m['resolve_single_conflict'] and all(before[k]==m[k] for k in
          ('capabilities','task_contract','task_archive_sha256','fit_limit','seed','preference_catalog')),'Unmatched paths')
    for split in ('primary','confirmation_1','confirmation_2'):
        check(ab_run.sha(folders/(split+'.joblib'))==ab_run.sha(old/(split+'.joblib')),'Different training or search data')
    history=read(old/'prediction_evidence.json')
    all_new=read(folders/'prediction_evidence.json')
    check(history==all_new[:len(history)],'Previous per-row predictions, scores or IDs differ')
    decision=read(folders/'confirmation.json')['b_scheduler_review']
    check(decision['single_conflict']['status'] in ('retain','switch') and
          [r['split_id'] for r in all_new[len(history):]]==['confirmation_2','confirmation_2'] and
          new['selected']==decision['members'] and new['fits']==len(all_new)+1, 'Incomplete third comparison')
    frozen_save(a.out/'decision.json',dict(decision=decision,actual=new,matched_prior_rows_and_scores=len(history),
                third_observations=[{'candidate':r['candidate_id'],'predictions':r['prediction_sha256'],
                                     'rows':len(r['predictions'])} for r in all_new[len(history):]]))
    digest=verified_export(folders,a.quotes,env,a.out)
    expected=read(a.controls/'exports.json')
    frozen_save(a.out/'exports.json',dict(new=digest,prior=expected['dynamic'],fixed=expected['fixed']))
    check(ab_run.sha(a.quotes/'scoring_only.json')==held['scoring'],'Holdout changed')
    truth=read(a.quotes/'scoring_only.json')
    y=np.array(truth['labels']); groups=np.array(truth['components'])
    new_hits=classes(a.out/'autonomous_export_0/rows.csv',digest,len(y))==y
    old_hits=classes(a.quotes/'autonomous_export_0/rows.csv',expected['dynamic'],len(y))==y
    fixed=classes(a.controls/'fixed/autonomous_export_0/rows.csv',expected['fixed'],len(y))==y
    delta=new_hits.astype(int)-old_hits.astype(int)
    result=dict(rows=len(y),new=int(new_hits.sum()),prior=int(old_hits.sum()),fixed=int(fixed.sum()),new_fits=new['fits'],
                gained=int((delta>0).sum()),lost=int((delta<0).sum()),paired=paired(delta,groups),selected=new['selected'],
                status=decision['single_conflict']['status'],identical_to_fixed=bool(np.array_equal(new_hits,fixed)),
                prior_titles_replayed=len(no_change),model_api_calls=0,new_rmb=0,known_development_task=True)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(result=ab_run.sha(a.out/'results.json'),protocol=ab_run.sha(a.out/'protocol.json'),
        decision=ab_run.sha(a.out/'decision.json'),exports=ab_run.sha(a.out/'exports.json'),title_replay=ab_run.sha(a.out/'title_replay.json'),
        physical_fits=new['fits'],model_api_calls=0))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','quotes','controls','title-inputs','title-promoted','title-uncontested','out'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args())
