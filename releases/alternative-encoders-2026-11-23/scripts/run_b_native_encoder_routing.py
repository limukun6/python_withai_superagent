"""Exercise optional confidence routing through a cold native run and fresh prediction processes."""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, command, read
from scripts.run_b_pair_surface_checks import pairs
from superagent.ab_encoder_routing import ConfidenceRoutedPreferenceVote
from superagent.ab_methods import JOINT_PREFERENCE
from superagent.ab_run import sha, sources
from superagent.prediction_evidence import fingerprint

PLAN=Path(__file__).resolve().parents[1]/'docs/batches/B_native_encoder_routing_1123_zh.md'


def observations(folder):
    return [(r['candidate_id'],r['split_id'],r['prediction_sha256'],r['score_record']['scores_sha256'],
             r['evaluation_payload_sha256']) for r in read(folder/'prediction_evidence.json')]


def verify_predictor(model, rows, predictions):
    check(isinstance(model,ConfidenceRoutedPreferenceVote),'Expected routed predictor')
    empty=model.predict_proba([])
    check(empty.shape==(0,2) and len(model.predict([]))==0,'Empty batch handling')
    sample=rows[:75]
    ordinary=model.predict_proba(sample)
    swapped=model.predict_proba([(b,a) for a,b in sample])
    np.testing.assert_allclose(ordinary,swapped[:,::-1],rtol=0,atol=1e-12)
    np.testing.assert_allclose(ordinary.sum(axis=1),1.,rtol=0,atol=1e-12)
    np.testing.assert_allclose(model.predict_proba([('same','same')]),[[.5,.5]],rtol=0,atol=0)
    np.testing.assert_array_equal(model.predict(sample),predictions[:len(sample)])
    return dict(swapped=len(sample),empty_rows=0,tied_probability=.5,
                maximum_swap_error=float(np.max(np.abs(ordinary-swapped[:,::-1]))))


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    reference=a.control/'autonomous'
    old=read(reference/'run_manifest.json')
    previous=read(a.study/'protocol.json')['tasks']['titles']
    prereq=read(a.replication/'completion.json')
    inputs=read(a.inputs/'protocol.json')
    training=sha(a.inputs/'anonymous.zip')
    check(read(a.replication/'results.json')['eligible'] and prereq['results']==sha(a.replication/'results.json')
          and old['task_archive_sha256']==training==previous['training']==inputs['training_sha']
          and old['seed']==18643 and old['fit_limit']==10 and old['planner_mode']=='local','Baseline or input mismatch')
    frozen_save(a.out/'protocol.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),sources=sources(),seed=18643,
        training=training,control=sha(a.control/'completion.json'),development_features=previous['features'],
        study=sha(a.study/'completion.json'),replication=sha(a.replication/'completion.json'),fit_cap=10,api_cap=0))
    env=offline_environment(a.workspace)
    d,physical,log=a.out/'autonomous',a.out/'physical.jsonl',a.out/'run.log'
    if not any(f.exists() for f in (d,physical,log)):
        args=['run',str(a.inputs/'anonymous.zip'),'--portfolio','ab-v1','--scheduler','ab-adaptive-complement',
              '--cover-untried-encoder','--confidence-cascade','--planner','local','--preference-catalog',JOINT_PREFERENCE,
              '--seed','18643','--fit-limit','10','--trial-timeout','300','--time-limit','1200','--run-dir',str(d)]
        rc,_=command(args,log,dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(rc==0,'Native route stopped; inspect its original durable log')
    start=read(d/'state_1.json')
    sel=read(d/'final_selection.json')
    outcome=read(d/'run_outcome.json')
    fits=read(d/'classifier_fits.json')
    manifest=read(d/'run_manifest.json')
    check(not start['history'] and not start['prediction_evidence'] and manifest['confidence_cascade']
          and manifest['capabilities']==old['capabilities'] and outcome['source_unchanged']
          and outcome['requested_planner_completed'] and not outcome['formal_test_accessed'] and outcome['new_api_requests']==0,
          'Wrong initial state, catalog, or outcome')
    check(fits['attempted_classifier_fits']==fits['completed_classifier_fits']==len(physical.read_text().splitlines())==10,
          'Physical fits differ')
    a_now=read(d/'actions.json')
    check(a_now==read(reference/'actions.json') and a_now[-1]['action']=='stop' and
          observations(d)==observations(reference),'Training path differs')
    chosen=sel['inference_routing']
    expected=read(a.study/'decisions.json')['titles']['policy']
    check(all(chosen[k]==v for k,v in expected.items()) and chosen['adopted'] and
          sel['ensemble_components']==read(reference/'final_selection.json')['ensemble_components'] and
          outcome['selected_candidate']==sel['family']=='confidence_routed_vote' and sel['model_sha256']==sha(d/'selected_model.joblib'),
          'Routed delivery differs from the frozen rule')
    frozen_save(a.out/'model_frozen.json',dict(model=sel['model_sha256'],members=sel['ensemble_components'],routing=chosen,
        actions=sha(d/'actions.json'),observations=observations(d),fit_count=10,physical=sha(physical),source=sha(d/'run_manifest.json')))
    check(sha(a.inputs/'features.zip')==previous['features'],'Wrong feature input')
    exported=verified_export(d,a.inputs,env,a.out)
    native=classes(a.out/'autonomous_export_0/rows.csv',exported,1693)
    baseline=read(a.control/'exports_frozen.json')['development']
    np.testing.assert_array_equal(classes(a.control/'autonomous_export_0/rows.csv',baseline,1693),native)
    check(fingerprint(native)==read(a.study/'exports.json')['titles']['selected'],'Research/deployed row differences')
    os.environ.update(env)
    check(sha(d/'selected_model.joblib')==sel['model_sha256'],'Saved predictor changed')
    technical=verify_predictor(joblib.load(d/'selected_model.joblib')['model'],pairs(a.inputs/'features.zip'),native)
    frozen_save(a.out/'exports.json',dict(native=exported,native_repeats=2,baseline=baseline,prediction=fingerprint(native),
                                          technical=technical,rows=len(native)))
    check(sha(a.inputs/'scoring_only.json')==previous['scorer'],'Wrong development scorer')
    y=np.asarray(read(a.inputs/'scoring_only.json')['labels'])
    result=dict(correct=int((native==y).sum()),rows=len(y),components=sel['ensemble_components'],first=chosen['first'],
        threshold=chosen['threshold'],actual_fits=10,actions=len(a_now),identical_trial_records=len(observations(d)),
        model_api_calls=0,new_rmb=0,course_validation_or_test=False)
    frozen_save(a.out/'results.json',result)
    frozen_save(a.out/'completion.json',dict(fits=10,model_api_calls=0,physical=sha(physical),
        **{k:sha(a.out/(k+'.json')) for k in ('protocol','model_frozen','exports','results')}))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','control','study','replication','out'):
        p.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args())
