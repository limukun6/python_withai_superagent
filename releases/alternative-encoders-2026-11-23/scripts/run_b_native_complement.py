"""Run one fresh native complement planner, verify the budget, and score sealed development exports."""
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
from scripts.run_b_encoder_vote_followup import classes,verified_export
from scripts.run_b_local_component_pair import check,command,read
from superagent import ab_run
from superagent.ab_methods import TEXT_DEEP

SEED=18643


def run(workspace,inputs,old,out):
    out.mkdir(parents=True,exist_ok=True)
    env=offline_environment(workspace)
    older=read(old/'protocol.json')
    check(older['seed']!=SEED and all(ab_run.sha(inputs/n)==v for n,v in older['identities']['packets'].items()),
          'Wrong training or reserve data')
    frozen_save(out/'protocol.json',dict(plan=ab_run.sha(ROOT/'docs/batches/B_native_complement_protocol_1123_zh.md'),
        runner=ab_run.sha(Path(__file__)),production=ab_run.sources(),previous=ab_run.sha(old/'results.json'),
        packets=older['identities']['packets'],seed=SEED,fit_cap=8,model_api_cap=0))
    folder=out/'autonomous'
    physical=out/'physical.jsonl'
    if not folder.exists() and not physical.exists():
        opts=['run',str(inputs/'anonymous.zip'),'--portfolio','ab-v1','--scheduler','ab-complement','--planner','local',
              '--preference-catalog','deep-e5','--seed',str(SEED),'--fit-limit','8','--trial-timeout','300',
              '--time-limit','1200','--run-dir',str(folder)]
        with_env=dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1')
        rc,_=command(opts,out/'run.log',with_env,1260)
        check(rc==0,'Native run failed. Inspect the recorded stage; no automatic rerun.')
    outcome=read(folder/'run_outcome.json')
    confirmation=read(folder/'confirmation.json')
    decision=confirmation['b_scheduler_review']
    ledger=read(folder/'classifier_fits.json')
    selection=read(folder/'final_selection.json')
    manifest=read(folder/'run_manifest.json')
    used=len(physical.read_text().splitlines())
    requested=4+(2 if decision['confirmation_rows'] else 0)+len(decision['members'])
    check(used==ledger['attempted_classifier_fits']==ledger['completed_classifier_fits']==requested<=8,
          'Wrong physical fit accounting')
    check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['source_unchanged']
          and outcome['new_api_requests']==0 and not outcome['formal_test_accessed'] and outcome['selected_candidate']==selection['family'],
          'Run incomplete or changed')
    check(manifest['complement_active'] and manifest['seed']==SEED and len(manifest['capabilities'])==34,'Wrong catalog')
    if decision['promoted']:
        check(selection['ensemble_components']==decision['members'] and selection['family']=='equal_soft_vote','Vote delivery mismatch')
    else:
        check(selection['family']==decision['anchor'] and not selection['ensemble_components'],'Single delivery mismatch')
    check(selection['model_sha256']==ab_run.sha(folder/'selected_model.joblib') and selection['training_rows']==6608,'Refit changed')
    log=[json.loads(x) for x in (folder/'llm_usage.jsonl').read_text().splitlines()]
    check(len(log)==1 and log[0]['event']=='ledger_initialized' and
          all(log[0][k]==0 for k in ('input_tokens','output_tokens','cost_rmb')),'Nonzero model usage')
    splits={k:read(folder/(k+'_split.json')) for k in ('primary','confirmation_1','confirmation_2')}
    check(all(s['search_rows_sharing_text']==0 and s['row_ids_sha256']!=older['splits'][k]['row_ids_sha256']
              for k,s in splits.items()),'New seed or group isolation not verified')
    frozen_save(out/'model_frozen.json',dict(candidate=selection['family'],members=decision['members'],model=selection['model_sha256'],
        primary_and_confirmation=decision,actual_fits=used,classification_refits=outcome['finalization']['full_refits_completed'],
        physical_log=ab_run.sha(physical),actions=ab_run.sha(folder/'actions.json'),training_rows=selection['training_rows']))
    export=verified_export(folder,inputs,env,out)
    baseline=read(old/'exports_frozen.json')['exports']['deep-e5']
    frozen_save(out/'exports_frozen.json',dict(new=export,original=baseline,model=ab_run.sha(out/'model_frozen.json')))
    y=np.asarray(read(inputs/'scoring_only.json')['labels'])
    g=np.asarray(read(inputs/'groups_only.json'))
    check(len(y)==len(g)==1693,'Wrong reserve')
    p=classes(out/'autonomous_export_0/rows.csv',export,len(y))==y
    base=classes(old/'deep-e5_export_0/rows.csv',baseline,len(y))==y
    check(int(base.sum())==1224,'Earlier full comparison changed')
    result=dict(seed=SEED,selected=selection['family'],members=decision['members'],source='fresh standard CLI',
        original_fixed_candidate=TEXT_DEEP,correct=int(p.sum()),rows=len(y),baseline_correct=int(base.sum()),
        gained=int((p&~base).sum()),lost=int((base&~p).sum()),delta=paired(p.astype(int)-base,g),
        actual_new_fits=used,model_api=0,new_course_rmb=0,official_validation=False,formal_test=False,
        export_seal=ab_run.sha(out/'exports_frozen.json'))
    frozen_save(out/'results.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','old','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    run(*(getattr(a,n).resolve() for n in ('workspace','inputs','old','output')))
