"""Compare actual native runs on one anonymous split with and without encoder coverage."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, read
from superagent import ab_run
from superagent.ab_methods import JOINT_PREFERENCE


def run(workspace, inputs, control, frozen, out):
    out.mkdir(parents=True,exist_ok=True)
    env=offline_environment(workspace)
    prior=control/'autonomous'
    old_protocol=read(control/'protocol.json')
    seed=old_protocol['seed']
    check(seed==40531 and old_protocol['method_menu']==JOINT_PREFERENCE and old_protocol['fit_limit']==10
          and ab_run.sha(inputs/'anonymous.zip')==old_protocol['training'],'Wrong matched control')
    frozen_save(out/'protocol.json',dict(plan=ab_run.sha(ROOT/'docs/batches/B_encoder_coverage_1123_zh.md'),
        runner=ab_run.sha(Path(__file__)),production=ab_run.sources(),control=ab_run.sha(control/'completion.json'),
        seed=seed,fit_limit=10,model_api_cap=0,training=old_protocol['training'],data_protocol=ab_run.sha(inputs/'protocol.json')))
    d,physical=out/'autonomous',out/'physical.jsonl'
    if not d.exists() and not physical.exists():
        args=['run',str(inputs/'anonymous.zip'),'--scheduler','ab-adaptive-complement','--portfolio','ab-v1','--planner','local',
              '--preference-catalog',JOINT_PREFERENCE,'--cover-untried-encoder','--seed',str(seed),'--fit-limit','10',
              '--trial-timeout','300','--time-limit','1200','--run-dir',str(d)]
        code,_=command(args,out/'run.log',dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(code==0,'Incomplete coverage run; inspect the recorded budget')
    m=read(d/'run_manifest.json')
    old_m=read(prior/'run_manifest.json')
    check(m['cover_untried_encoder'] and m['adaptive_opening_active'] and not old_m.get('cover_untried_encoder',False)
          and all(m[n]==old_m[n] for n in ('seed','capabilities','fit_limit','task_contract','preference_catalog')),
          'Control and treatment differ beyond the intended policy')
    for s in ('primary','confirmation_1','confirmation_2'):
        check(ab_run.sha(d/(s+'.joblib'))==ab_run.sha(prior/(s+'.joblib')) and
              read(d/(s+'_split.json'))==read(prior/(s+'_split.json')),'Partition bytes or row ordering changed')
    old_obs=read(prior/'prediction_evidence.json')[:4]
    new_obs=read(d/'prediction_evidence.json')[:4]
    check(len(new_obs)==len(old_obs)==4 and new_obs==old_obs,'First four measured row-level scores or predictions differ')
    decision=read(d/'confirmation.json')['b_scheduler_review']
    outcome,selection,ledger=[read(d/n) for n in ('run_outcome.json','final_selection.json','classifier_fits.json')]
    history=read(d/'history.json')
    primary=[r['candidate'] for r in history if r['split_id']=='primary']
    original_encoders={m['capabilities'][n]['params'].get('encoder') for n in primary[:4]}
    extra=m['capabilities'][primary[4]]['params'].get('encoder') if len(primary)==5 else None
    check(len(primary)==5 and extra and extra not in original_encoders and read(d/'actions.json')[4]['action']=='explore',
          'An eligible untested encoder was not explored on this registered state')
    members=decision['members']
    count=len(physical.read_text().splitlines())
    calls=[json.loads(line) for line in (d/'llm_usage.jsonl').read_text().splitlines()]
    check(len(calls)==1 and calls[0]['event']=='ledger_initialized' and calls[0]['cost_rmb']==0
          and outcome['requested_planner_completed'] and outcome['source_unchanged'] and outcome['full_refit_completed']
          and not outcome['formal_test_accessed'] and outcome['new_api_requests']==0 and selection['training_rows']==6608
          and count==ledger['attempted_classifier_fits']==ledger['completed_classifier_fits']==len(history)+len(members)<=10
          and selection['model_sha256']==ab_run.sha(d/'selected_model.joblib')
          and (selection['ensemble_components'] or [selection['family']])==members,'Run or physical accounting incomplete')
    frozen_save(out/'model_frozen.json',dict(decision=decision,opening=primary,extra_encoder=extra,
        original_opening_evidence_matched=True,model=selection['model_sha256'],physical_fits=count,physical_sha=ab_run.sha(physical),
        actions=ab_run.sha(d/'actions.json')))
    export=verified_export(d,inputs,env,out)
    control_hash=read(control/'exports_frozen.json')['new']
    fixed_hash=read(frozen/'exports_frozen.json')['new_vote']
    frozen_save(out/'exports_frozen.json',dict(treatment=export,control=control_hash,fixed=fixed_hash,decision=ab_run.sha(out/'model_frozen.json')))
    data_protocol=read(inputs/'protocol.json')
    check(data_protocol['training_sha']==old_protocol['training'] and all(ab_run.sha(inputs/n)==data_protocol[k]
          for k,n in [('feature_sha','features.zip'),('locked_scores','scoring_only.json'),('locked_groups','groups_only.json')]),
          'Training-scoped development reserve changed')
    labels=read(inputs/'scoring_only.json')['labels']
    groups=np.asarray(read(inputs/'groups_only.json'))
    y=np.asarray(labels)
    hard=classes(out/'autonomous_export_0/rows.csv',export,len(y))
    old=classes(control/'autonomous_export_0/rows.csv',control_hash,len(y))
    reference=classes(frozen/'e5_familiarity_export_0/rows.csv',fixed_hash,len(y))
    new_hits,old_hits,ref_hits=(a==y for a in (hard,old,reference))
    check(len(y)==len(groups)==1693 and old_hits.sum()==read(control/'results.json')['new_correct']
          and ref_hits.sum()==1237,'Wrong scoring reference')
    result=dict(seed=seed,opening=primary,additional_encoder=extra,selected=members,rows=len(y),new_correct=int(new_hits.sum()),
        control_correct=int(old_hits.sum()),fixed_research_correct=int(ref_hits.sum()),gained=int((new_hits&~old_hits).sum()),
        lost=int((~new_hits&old_hits).sum()),delta=paired(new_hits.astype(int)-old_hits,groups),
        actual_new_fits=count,control_fits=read(control/'completion.json')['physical_fits'],model_api=0,new_rmb=0,
        eligible_for_designated_validation=int(new_hits.sum())>=1235,official_validation_used=False,formal_test_used=False)
    frozen_save(out/'results.json',result)
    frozen_save(out/'completion.json',dict(status='complete',result=ab_run.sha(out/'results.json'),physical_fits=count,model_api=0))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','control','frozen','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    run(*(getattr(a,k).resolve() for k in ('workspace','inputs','control','frozen','output')))
