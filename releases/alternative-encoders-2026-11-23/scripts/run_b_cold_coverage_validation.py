"""Cold native replay and a gated one-shot designated validation of its exact frozen model."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import ARCHIVE_SHA, check, command, packets, read
from superagent import ab_run
from superagent.ab_methods import JOINT_PREFERENCE
from superagent.prediction_evidence import fingerprint

VAL='Released_Task2-Headlines_CTR/val.csv'


def columns(archive, fields):
    with zipfile.ZipFile(archive) as z, z.open(VAL) as f:
        rows=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''))
        check(set(fields)<=set(rows.fieldnames or ()),'Invalid course validation schema')
        return [tuple(row[k] for k in fields) for row in rows]


def designated(archive, run, model_sha, env, out):
    check(ab_run.sha(archive)==ARCHIVE_SHA and ab_run.sha(run/'selected_model.joblib')==model_sha,'Released source or frozen model changed')
    official=out/'designated'
    official.mkdir(exist_ok=True)
    source=official/'inputs'
    source.mkdir(exist_ok=True)
    features=columns(archive,('headline_A','headline_B'))
    check(len(features)==940,'Unexpected designated validation size')
    file=source/'features.zip'
    if not file.exists():
        packets(features,range(len(features)),False,file)
    with zipfile.ZipFile(file) as z,z.open('rows.csv') as stream:
        rows=csv.reader(io.TextIOWrapper(stream,encoding='utf-8',newline=''))
        check(next(rows)==['option_a','option_b'] and [tuple(r) for r in rows]==features,'Feature-only packet changed rows')
    digest=verified_export(run,source,env,official)
    frozen_save(official/'prediction_seal.json',dict(model=model_sha,predictions=digest,feature_packet=ab_run.sha(file),
        rows=len(features),source_member=VAL,release_archive=ARCHIVE_SHA,label_column_in_prediction_packet=False))
    truth=np.asarray([int(r[0]) for r in columns(archive,('label_pairwise',))])
    hard=classes(official/(run.name+'_export_0')/'rows.csv',digest,len(truth))
    check(set(np.unique(truth))==set(np.unique(hard))=={0,1} and len(truth)==940,'Invalid label or prediction coding')
    table=[[int(((truth==i)&(hard==j)).sum()) for j in (0,1)] for i in (0,1)]
    n=int((hard==truth).sum())
    result=dict(scope='course designated validation',rows=len(truth),correct=n,accuracy=n/len(truth),
        frozen_model_training_rows=6608,training_archive_rows_available=8301,confusion_true_by_predicted=table,
        validation_target_sha256=fingerprint(truth),prediction_seal=ab_run.sha(official/'prediction_seal.json'),
        new_supervised_fits=0,model_api=0,course_test_member_opened=False)
    frozen_save(official/'results.json',result)
    return result


def run(workspace, inputs, opened, previous, joint, reconstructed, archive, out):
    out.mkdir(parents=True,exist_ok=True)
    env=offline_environment(workspace)
    replay=read(reconstructed/'results.json')
    expected=read(opened/'protocol.json')['packets']
    check(replay['eligible_for_cold_run'] and replay['seed']==18643 and ab_run.sha(inputs/'anonymous.zip')==expected['anonymous.zip'],
          'Incomplete prerequisite or training data changed')
    frozen_save(out/'protocol.json',dict(plan=ab_run.sha(ROOT/'docs/batches/B_cold_coverage_validation_1123_zh.md'),
        code=ab_run.sha(Path(__file__)),production=ab_run.sources(),replay=ab_run.sha(reconstructed/'completion.json'),
        training=expected['anonymous.zip'],seed=18643,fit_cap=10,api_cap=0,designated_gate=1235,
        designated_rows=940,training_rows=6608,source_archive=ARCHIVE_SHA,allowed_designated_member=VAL))
    d,physical=out/'autonomous',out/'physical.jsonl'
    if not d.exists() and not physical.exists():
        args=['run',str(inputs/'anonymous.zip'),'--portfolio','ab-v1','--scheduler','ab-adaptive-complement',
              '--cover-untried-encoder','--planner','local','--preference-catalog',JOINT_PREFERENCE,'--seed','18643',
              '--fit-limit','10','--trial-timeout','300','--time-limit','1200','--run-dir',str(d)]
        code,_=command(args,out/'run.log',dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),1260)
        check(code==0,'Cold native run ended early. Inspect its durable fit record.')
    hist=read(d/'history.json')
    decision=read(d/'confirmation.json')['b_scheduler_review']
    sel,outcome,counts=[read(d/n) for n in ('final_selection.json','run_outcome.json','classifier_fits.json')]
    manifest=read(d/'run_manifest.json')
    check(manifest['cover_untried_encoder'] and manifest['adaptive_opening_active']
          and manifest['capabilities']==read(joint/'autonomous/run_manifest.json')['capabilities'] and
          all(ab_run.sha(d/(s+'.joblib'))==ab_run.sha(opened/'autonomous'/(s+'.joblib')) for s in ('primary','confirmation_1','confirmation_2')),
          'Historical and cold contracts differ')
    members=decision['members']
    actual=len(physical.read_text().splitlines())
    usage=[json.loads(l) for l in (d/'llm_usage.jsonl').read_text().splitlines()]
    check(len(usage)==1 and usage[0]['event']=='ledger_initialized' and usage[0]['cost_rmb']==0 and outcome['new_api_requests']==0
          and outcome['requested_planner_completed'] and outcome['source_unchanged'] and outcome['full_refit_completed']
          and not outcome['formal_test_accessed'] and actual==len(hist)+len(members)==counts['attempted_classifier_fits']
          ==counts['completed_classifier_fits']<=10 and (sel['ensemble_components'] or [sel['family']])==members
          and sel['training_rows']==6608 and sel['model_sha256']==ab_run.sha(d/'selected_model.joblib'),'Native completion not verified')
    historical={}
    for folder in (opened/'autonomous',previous/'autonomous'):
        for r in read(folder/'prediction_evidence.json'):
            historical[r['candidate_id'],r['split_id']]=r['prediction_sha256']
    extra=next(reconstructed.glob('confirmation_*/worker.joblib'))
    import joblib
    supplemental=joblib.load(extra)
    historical[supplemental['metrics']['resolved_configuration']['id'],'confirmation_1']=supplemental['prediction_record']['prediction_sha256']
    observations=read(d/'prediction_evidence.json')
    agreed=sum(historical.get((r['candidate_id'],r['split_id']))==r['prediction_sha256'] for r in observations)
    frozen_save(out/'model_frozen.json',dict(model=sel['model_sha256'],decision=decision,actions=ab_run.sha(d/'actions.json'),
        actual_fits=actual,physical_sha256=ab_run.sha(physical),matched_prior_predictions=agreed,
        actual_new_observations=len(observations),model_api_requests=0))
    export=verified_export(d,inputs,env,out)
    prior_export=read(reconstructed/'exports_frozen.json')['new']
    frozen_save(out/'exports_frozen.json',dict(model=sel['model_sha256'],development=export,prior=prior_export))
    check(all(ab_run.sha(inputs/n)==expected[n] for n in ('features.zip','groups_only.json','scoring_only.json')),'Development reserve changed')
    y=np.asarray(read(inputs/'scoring_only.json')['labels'])
    check(len(y)==1693,'Wrong development reserve size')
    correct=int((classes(out/'autonomous_export_0/rows.csv',export,len(y))==y).sum())
    stable=members==replay['members'] and agreed==len(hist) and export==prior_export
    result=dict(seed=18643,selected=members,development_rows=len(y),development_correct=correct,
        actual_new_fits=actual,model_api_requests=0,new_rmb=0,cold_start=True,all_replay_predictions_and_decisions_match=stable,
        designated_eligible=correct>=1235,model=sel['model_sha256'],course_test_opened=False)
    frozen_save(out/'development_results.json',result)
    official=designated(archive,d,sel['model_sha256'],env,out) if result['designated_eligible'] else None
    frozen_save(out/'results.json',dict(**result,designated=official))
    frozen_save(out/'completion.json',dict(status='complete',result=ab_run.sha(out/'results.json'),physical_fits=actual,model_api=0))
    print(dict(development=correct,designated=official,physical_fits=actual,replayed_exactly=stable),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','opened','previous','joint','reconstructed','archive','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    run(*(getattr(a,k).resolve() for k in ('workspace','inputs','opened','previous','joint','reconstructed','archive','output')))
