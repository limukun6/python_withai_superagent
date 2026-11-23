"""Finish only the missing refit of a frozen replay decision, then check exported predictions."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_fixed_encoder_vote import PreferenceSoftVote,canonical_frozen
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_encoder_agreement import compare
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes,verified_export
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.external_assets import model_resources,verify_resources


def export_model(out,name,model,task_spec):
    folder=out/name
    folder.mkdir(exist_ok=True)
    target=folder/'selected_model.joblib'
    if not target.exists():
        shutil.copyfile(task_spec,folder/'task_spec.json')
        joblib.dump(dict(kind='pair_text',candidate=name,model=model),target)
    externals,_=model_resources(model)
    verify_resources(externals)
    expected={'bge','minilm_l12'} if name=='bge_vote' else {'bge'}
    check({r['name'] for r in externals}==expected,'Wrong frozen encoder resources')
    seal=dict(selected_experiment=name,family=name,model_file=target.name,model_format='candidate_bundle_v1',
         model_sha256=ab_run.sha(target),task_spec_sha256=ab_run.sha(task_spec),training_rows=6608,refit=True,
         test_accessed=False,resources=externals)
    frozen_save(folder/'final_selection.json',seal)
    return seal['model_sha256']


def run(inputs,batch,follow,replay,out,env):
    out.mkdir(parents=True,exist_ok=True)
    replay_result=read(replay/'results.json')
    decisions=replay/'decisions_before_score_lookup.json'
    check(replay_result['decision_seal']==ab_run.sha(decisions),'Replay altered')
    record=read(decisions)['deep-bge']
    ref=batch/'deep-bge'
    original=read(ref/'run_outcome.json')['selected_candidate']
    check(record['status']=='replayed' and record['fit_equivalents']==8 and record['anchor']==original
          and record['selected']==[original,record['requested_confirmation']],'No registered portfolio for this replay')
    contender=record['requested_confirmation']
    prior=read(follow/'results.json')
    check(prior['encoding_study_total_fits']==22 and read(follow/'fits.json')['deep-bge']['candidate']==contender,
          'Incorrect resource handoff')
    source=ref/'refit_data.joblib'
    anchor=ref/'selected_model.joblib'
    check(ab_run.sha(source)==read(follow/'protocol.json')['payloads']['deep-e5'] and
          ab_run.sha(anchor)==read(batch/'deep-bge_receipt.json')['model'],'Full refit inputs changed')
    frozen_save(out/'protocol.json',dict(plan=ab_run.sha(ROOT/'docs/batches/B_replayed_vote_materialization_1123_zh.md'),
        runner=ab_run.sha(Path(__file__)),replay=ab_run.sha(decisions),candidate=contender,payload=ab_run.sha(source),
        retained_model=ab_run.sha(anchor),prior_physical_fits=22,fit_equivalents=8,reused_components=7,
        additional_physical_fit_cap=1,model_api_cap=0))
    admission=out/'admission.json'
    ledger=out/'worker_fits.json'
    physical=out/'physical.jsonl'
    worker=out/'worker.joblib'
    if not admission.exists():
        save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)))
        with (out/'worker.log').open('x') as log:
            rc=subprocess.run([sys.executable,'-m','superagent.experiment_worker',str(source),contender,str(worker),
                '--fit-admission',str(admission)],cwd=ROOT,env=dict(env,**{PREFERENCE_CATALOG_ENV:'deep-bge'},
                AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),stdout=log,stderr=subprocess.STDOUT,timeout=300)
        check(rc.returncode==0,'Missing refit failed')
    count=read(ledger)
    check(len(physical.read_text().splitlines())==count['attempted_classifier_fits']==count['completed_classifier_fits']==1
          and count['candidate']==contender,'Incorrect resource accounting')
    with patch.dict(os.environ,env):
        base=canonical_frozen(joblib.load(anchor)['model'])
        candidate=canonical_frozen(joblib.load(worker)['model'])
        ensemble=PreferenceSoftVote('pair_text',[base,candidate],np.array([0,1]))
        frozen_save(out/'models_frozen.json',{n:export_model(out,n,m,ref/'task_spec.json')
                    for n,m in [('bge_full',candidate),('bge_vote',ensemble)]})
        seals={n:verified_export(out/n,inputs,env,out) for n in ('bge_full','bge_vote')}
    seals['deep']=read(batch/'exports_frozen.json')['exports']['deep-bge']
    seals['e5_vote']=read(follow/'exports_frozen.json')['soft_vote']
    frozen_save(out/'exports_frozen.json',seals)
    inputs_signed=read(batch/'protocol.json')['identities']['packets']
    check(all(ab_run.sha(inputs/n)==inputs_signed[n] for n in ('features.zip','scoring_only.json','groups_only.json')),'Reserve mismatch')
    y=np.array(read(inputs/'scoring_only.json')['labels'])
    g=np.array(read(inputs/'groups_only.json'))
    check(len(y)==len(g)==1693,'Wrong reserve size')
    paths={n:out/(n+'_export_0')/'rows.csv' for n in ('bge_full','bge_vote')}
    paths.update(deep=batch/'deep-bge_export_0/rows.csv',e5_vote=follow/'soft_vote_export_0/rows.csv')
    preds={n:classes(path,seals[n],len(y)) for n,path in paths.items()}
    base=preds['deep']
    metrics={n:dict(**compare(v,base,y),delta=paired((v==y).astype(int)-(base==y),g)) for n,v in preds.items()}
    summary=dict(reserve=metrics,paired_against_e5_vote=compare(preds['bge_vote'],preds['e5_vote'],y),
        physical_fits_this_stage=1,physical_fits_across_batches=23,path_fit_equivalents=8,
        path_reused_fits=7,model_api=0,course_validation=False,formal_test=False,replayed_decision=True,
        export_seal=ab_run.sha(out/'exports_frozen.json'))
    frozen_save(out/'results.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','batch','followup','replay','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(*(getattr(a,n).resolve() for n in ('inputs','batch','followup','replay','output')),offline_environment(a.workspace.resolve()))
