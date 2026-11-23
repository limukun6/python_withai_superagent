"""Use one spare fit on each closed comparison to test a qualified vote and a missing replication."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_fixed_encoder_vote import PreferenceSoftVote,canonical_frozen
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_encoder_agreement import compare,predictions
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check,command,read,save
from superagent import ab_run
from superagent.ab_methods import PREFERENCE_CATALOG_ENV,TEXT_DEEP,TRANSFER_METHODS
from superagent.external_assets import model_resources, verify_resources
from superagent.prediction_evidence import fingerprint


def one_fit(arm,payload,batch,out,env):
    name=TRANSFER_METHODS[arm][1]
    folder=out/arm
    folder.mkdir(exist_ok=True)
    physical=folder/'physical.jsonl'
    fits=folder/'fits.json'
    admission=folder/'admission.json'
    target=folder/'worker.joblib'
    count=len((batch/(arm+'.physical.jsonl')).read_text().splitlines())
    check(count==read(batch/(arm+'_receipt.json'))['actual_fits']==10,'Wrong previous resource count')
    if not admission.exists():
        save(admission,dict(maximum_classifier_fits=1,ledger_path=str(fits)))
        cmd=[sys.executable,'-m','superagent.experiment_worker',str(payload),name,str(target),'--fit-admission',str(admission)]
        with (folder/'worker.log').open('x') as log:
            status=subprocess.run(cmd,cwd=ROOT,env=dict(env,**{PREFERENCE_CATALOG_ENV:arm},
                AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1'),
                stdout=log,stderr=subprocess.STDOUT,timeout=300)
        check(status.returncode==0,'Worker failed. The same fit is not attempted again.')
    calls=[json.loads(line) for line in physical.read_text().splitlines()]
    record=read(fits)
    check(len(calls)==record['attempted_classifier_fits']==record['completed_classifier_fits']==1 and
          record['candidate']==calls[0]['candidate']==name and record['status']=='completed','Fit receipt mismatch')
    return dict(candidate=name,model=ab_run.sha(target),physical=ab_run.sha(physical),worker_ledger=ab_run.sha(fits),
                new_fits=1,prior_fits=count,total=count+1)


def bundle(out,name,model,task_spec):
    folder=out/name
    folder.mkdir(exist_ok=True)
    path=folder/'selected_model.joblib'
    if not path.exists():
        shutil.copyfile(task_spec,folder/'task_spec.json')
        joblib.dump(dict(kind='pair_text',candidate=name,model=model),path)
    assets,_=model_resources(model)
    verify_resources(assets)
    check({r['name'] for r in assets}==({'e5'} if name=='e5_full' else {'e5','minilm_l12'}),'Wrong fitted resources')
    frozen_save(folder/'final_selection.json',dict(selected_experiment=name,family=name,model_file=path.name,
        model_format='candidate_bundle_v1',model_sha256=ab_run.sha(path),task_spec_sha256=ab_run.sha(task_spec),
        training_rows=6608,test_accessed=False,refit=True,choice_source='separate matched evidence follow-up',
        resources=assets))
    return ab_run.sha(path)


def verified_export(folder,inputs,env,out):
    seals=[]
    for repeat in range(2):
        where=out/(folder.name+'_export_'+str(repeat))
        file=where/'rows.csv'
        if not where.exists():
            where.mkdir()
            rc,_=command(['predict',str(inputs/'features.zip'),'--run-dir',str(folder),'--output',str(file)],
                         where/'predict.log',env,600)
            check(rc==0,'Fresh-process prediction failed')
        check(not read(where/'prediction_manifest.json')['test_labels_read'],'Unexpected labels')
        seals.append(ab_run.sha(file))
    check(seals[0]==seals[1],'Fresh-process predictions differ')
    return seals[0]


def classes(file,digest,n):
    check(ab_run.sha(file)==digest,'Prediction changed')
    with file.open(newline='') as f:
        rows=list(csv.DictReader(f))
    check([int(r['row_id']) for r in rows]==list(range(n)),'Row alignment drift')
    return np.array([int(r['prediction']) for r in rows])


def bge_confirmation(batch,out,evidence):
    part='confirmation_1'
    source=batch/'deep-e5'
    trials=read(source/'history.json')
    old=next(r for r in trials if r['candidate']==TEXT_DEEP and r['split_id']==part)
    first=joblib.load(source/f"trial_{old['trial']}.joblib")
    new=joblib.load(out/'deep-bge/worker.joblib')
    payload=joblib.load(source/(part+'.joblib'))
    check(fingerprint(payload['y_search'])==old['label_sha256']==new['prediction_record']['label_sha256'],'Labels differ')
    left,p=predictions(first['model'],payload['X_search'])
    right,q=predictions(new['model'],payload['X_search'])
    check(np.array_equal(left,first['search_predictions']) and np.array_equal(right,new['search_predictions']),
          'Replicated underlying predictions differ')
    prob=(p+q)/2
    ordered=np.array([l<r for l,r in payload['X_search']])
    average=((prob>.5)|((prob==.5)&ordered)).astype(int)
    y=payload['y_search']
    prior=evidence['primary']
    ticks=[prior['observed_correct']['deep'],int((left==y).sum())]
    challenger=[prior['observed_correct']['bge'],int((right==y).sum())]
    votes=[prior['pair_mean']['bge']['correct'],int((average==y).sum())]
    total=prior['rows']+len(y)
    gain=sum(votes)-max(sum(ticks),sum(challenger))
    wins=sum(v>max(l,r) for l,r,v in zip(ticks,challenger,votes))
    return dict(deep=ticks,bge=challenger,mean=votes,first_confirmation=compare(average,left,y),
        paired_row_observations=total,pooled_gain=gain,pooled_delta_pp=gain/total*100,
        wins=wins,diagnostic_gate=wins>=2 and gain/total>=.002)


def main(workspace,inputs,batch,diagnosis,out):
    import os
    from unittest.mock import patch
    env=offline_environment(workspace)
    protocol=read(batch/'protocol.json')
    old=read(batch/'results.json')
    evidence=read(diagnosis/'inner.json')
    source=batch/'deep-e5'
    weights=source/'selected_model.joblib'
    payloads={'deep-e5':source/'refit_data.joblib','deep-bge':batch/'deep-bge/confirmation_1.joblib'}
    check(ab_run.sha(weights)==read(source/'final_selection.json')['model_sha256']==read(batch/'deep-e5_receipt.json')['model'],
          'Full-training anchor changed')
    check(ab_run.sha(payloads['deep-bge'])==ab_run.sha(source/'confirmation_1.joblib')==
          protocol['splits']['confirmation_1']['evaluation_payload_sha256'],'Confirmation split differs')
    check(ab_run.sha(payloads['deep-e5'])==ab_run.sha(batch/'deep-bge/refit_data.joblib'),'Full-training packets differ')
    check(old['actual_total_new_fits']==20 and read(diagnosis/'results.json')['pair_average_gate']['e5']['passed']
          and 'bge' not in evidence.get('confirmation_1',{}).get('observed_correct',{}),'Unregistered follow-up')
    out.mkdir(parents=True,exist_ok=True)
    signed=dict(plan=ab_run.sha(ROOT/'docs/batches/B_encoder_followup_protocol_1123_zh.md'),
        code=ab_run.sha(Path(__file__)),vote=ab_run.sha(ROOT/'scripts/b_fixed_encoder_vote.py'),
        source=ab_run.sources(),earlier=ab_run.sha(batch/'results.json'),evidence=ab_run.sha(diagnosis/'inner.json'),
        payloads={k:ab_run.sha(v) for k,v in payloads.items()},previous_fits=20,additional_fit_cap=2,model_api_cap=0)
    frozen_save(out/'protocol.json',signed)
    receipts={k:one_fit(k,v,batch,out,env) for k,v in payloads.items()}
    frozen_save(out/'fits.json',receipts)
    with patch.dict(os.environ,env):
        anchor=canonical_frozen(joblib.load(weights)['model'])
        e5=canonical_frozen(joblib.load(out/'deep-e5/worker.joblib')['model'])
        vote=PreferenceSoftVote('pair_text',[anchor,e5],np.array([0,1]))
        frozen_save(out/'models_frozen.json',{n:bundle(out,n,m,source/'task_spec.json')
            for n,m in [('e5_full',e5),('soft_vote',vote)]})
        exports={n:verified_export(out/n,inputs,env,out) for n in ('e5_full','soft_vote')}
        exports['deep']=read(batch/'exports_frozen.json')['exports']['deep-e5']
        frozen_save(out/'exports_frozen.json',exports)
        bge=bge_confirmation(batch,out,evidence)
    check(all(ab_run.sha(inputs/n)==protocol['identities']['packets'][n] for n in ('features.zip','scoring_only.json','groups_only.json')),
          'Reserve identity differs')
    y=np.asarray(read(inputs/'scoring_only.json')['labels'])
    groups=np.asarray(read(inputs/'groups_only.json'))
    check(len(y)==len(groups)==1693,'Wrong reserve')
    paths={n:out/(n+'_export_0')/'rows.csv' for n in ('e5_full','soft_vote')}
    paths['deep']=batch/'deep-e5_export_0/rows.csv'
    preds={n:classes(file,exports[n],len(y)) for n,file in paths.items()}
    base=preds['deep']
    stats={n:dict(**compare(v,base,y),delta=paired((v==y).astype(int)-(base==y),groups)) for n,v in preds.items()}
    result=dict(reserve=stats,bge_replication=bge,new_fits=2,encoding_study_total_fits=22,model_api_requests=0,
        new_model_api_rmb=0,course_designated_validation_used=False,formal_test_used=False,autonomous_original_terminal_choice=TEXT_DEEP,
        separate_followup=True,export_seal=ab_run.sha(out/'exports_frozen.json'))
    frozen_save(out/'results.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','batch','diagnosis','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    main(*(getattr(a,n).resolve() for n in ('workspace','inputs','batch','diagnosis','output')))
