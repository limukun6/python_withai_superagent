"""Reconstruct the four-opening control using verified predictions and reusable full fits."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, read, save
from superagent import ab_complement, ab_run
from superagent.ab_methods import JOINT_PREFERENCE, PREFERENCE_CATALOG_ENV
from superagent.benchmark import train_candidate
from superagent.external_assets import model_resources, verify_resources
from superagent.metrics import MetricSpec


def lookup_confirmation(name, source, cap, out):
    split = 'confirmation_1'
    actual = next((r for r in read(source/'history.json') if r['candidate']==name and r['split_id']==split),None)
    if actual:
        path = source/f"trial_{actual['trial']}.joblib"
        artifact = joblib.load(path)
        return artifact, dict(source='same-run matched observation',artifact=ab_run.sha(path),physical_new_fits=0)
    folder=out/('confirmation_'+name)
    folder.mkdir(exist_ok=True)
    admission,file,ledger,physical=[folder/n for n in ('admission.json','worker.joblib','fits.json','physical.jsonl')]
    if not admission.exists():
        save(admission,dict(maximum_classifier_fits=1,ledger_path=str(ledger)))
        with patch.dict(os.environ,{PREFERENCE_CATALOG_ENV:JOINT_PREFERENCE,'AB_B1_PER_WORKER_FIT_LIMIT':'1',
                                    'AB_B1_FIT_LOG':str(physical)}):
            model,facts=train_candidate(source/(split+'.joblib'),name,file,360,fit_admission_path=admission)
        check(model is not None,'Missing control confirmation failed: '+str(facts.get('error','unknown')))
    account=read(ledger)
    check(account['status']=='completed' and account['candidate']==name and
          account['attempted_classifier_fits']==account['completed_classifier_fits']==len(physical.read_text().splitlines())==1,
          'Incorrect new confirmation accounting')
    artifact=joblib.load(file)
    check(artifact['metrics']['resolved_configuration']['params']==cap['params'],'Wrong newly trained control')
    return artifact,dict(source='additional physical confirmation',artifact=ab_run.sha(file),physical_new_fits=1,
        fit_ledger=ab_run.sha(ledger),physical_log=ab_run.sha(physical))


def full_model(name, dirs, source, cap):
    for folder in dirs:
        final=read(folder/'final_selection.json')
        members=final['ensemble_components'] or [final['family']]
        if name not in members:
            continue
        check(final['training_rows']==6608 and final['refit'] and final['task_spec_sha256']==ab_run.sha(source/'task_spec.json')
              and ab_run.sha(folder/'refit_data.joblib')==ab_run.sha(source/'refit_data.joblib')
              and read(folder/'run_manifest.json')['capabilities'][name]['params']==cap['params']
              and final['model_sha256']==ab_run.sha(folder/'selected_model.joblib'),'Wrong historical complete model')
        model=joblib.load(folder/'selected_model.joblib')['model']
        if final['ensemble_components']:
            model=model.models[members.index(name)]
        return model,dict(model_bundle=final['model_sha256'],component=name,index=members.index(name),
            full_training_payload=ab_run.sha(folder/'refit_data.joblib'),training_rows=6608,additional_refits=0)
    raise ValueError('No verified full checkpoint for '+name)


def run(inputs, batch, old, earlier, deeper, out, env):
    out.mkdir(parents=True,exist_ok=True)
    source=batch/'autonomous'
    protocol=read(batch/'protocol.json')
    stage=source/'state_5.json'
    state=read(stage)
    caps=read(source/'run_manifest.json')['capabilities']
    check(protocol['fit_limit']==state['budget']['fit_limit']==10 and len(state['history'])==4 and
          all(r['split_id']=='primary' for r in state['history']),'Expected only the four common opening observations')
    frozen_save(out/'protocol.json',dict(plan=ab_run.sha(ROOT/'docs/batches/B_same_seed_opening_control_1123_zh.md'),
        runner=ab_run.sha(Path(__file__)),native=ab_run.sha(batch/'completion.json'),
        source_state=ab_run.sha(stage),fit_limit=10,additional_confirmation_fit_cap=2,model_api_cap=0))
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        s=read(source/(split+'_split.json'))
        data=joblib.load(source/(split+'.joblib'))
        trusted[split]=dict(labels=data['y_search'].tolist(),row_ids=s['search_indices'],
            evaluation_payload_sha256=ab_run.sha(source/(split+'.joblib')),tie_label_one=[a<b for a,b in data['X_search']])
    receipts={}
    actions=[]
    while True:
        action=ab_complement.action(state,caps,trusted)
        actions.append(action)
        if action['action']=='stop':
            break
        name,split=action['candidate_id'],action['split_id']
        check(action['action']=='confirm' and split=='confirmation_1' and name not in receipts and len(receipts)<2,
              'Unregistered old-scheduler action')
        artifact,receipts[name]=lookup_confirmation(name,source,caps[name],out)
        raw=artifact['prediction_record']
        facts=artifact['metrics']
        t=trusted[split]
        check(facts['resolved_configuration']['id']==name and facts['resolved_configuration']['params']==caps[name]['params']
              and facts['classifier_fits']==1 and raw['evaluation_payload_sha256']==t['evaluation_payload_sha256'],
              'Reused observation metadata differs')
        number=len(state['history'])+1
        state['history'].append(dict(trial=number,candidate=name,params=caps[name]['params'],split_id=split,
            proposal=None,status='completed',**facts,evaluation_payload_sha256=raw['evaluation_payload_sha256'],
            label_sha256=raw['label_sha256'],row_ids_sha256=ab_run.fingerprint(t['row_ids'])))
        state['prediction_evidence'].append(dict(trial=number,candidate_id=name,split_id=split,row_id_scope=ab_run.ROW_SCOPE,
            row_ids=t['row_ids'],predictions=raw['predictions'].tolist(),
            **{k:raw[k] for k in ('prediction_sha256','label_sha256','evaluation_payload_sha256','score_record')}))
        for k in ('attempted_classifier_fits','completed_classifier_fits'):
            state['budget'][k]+=1
        state['budget']['remaining_fit_allowance']-=1
    proof=ab_complement.review(state,trusted)
    members=proof['members']
    fits_equivalent=4+len(receipts)+len(members)
    newly=sum(r['physical_new_fits'] for r in receipts.values())
    check(fits_equivalent<=10 and newly<=2,'Excess resources')
    frozen_save(out/'decision_frozen.json',dict(original_decision=proof,actions=actions,
        confirmations=receipts,fit_equivalents=fits_equivalent,physical_new_fits=newly,unchanged_first_four=True))
    dirs=[source,old/'autonomous',earlier/'autonomous',deeper/'deep-e5']
    pairs=[full_model(n,dirs,source,caps[n]) for n in members]
    model=pairs[0][0] if len(pairs)==1 else ab_complement.PreferenceProbabilityVote('pair_text',[m for m,r in pairs],np.array([0,1]))
    name=members[0] if len(members)==1 else 'equal_soft_vote'
    where=out/'control'
    where.mkdir(exist_ok=True)
    file=where/'selected_model.joblib'
    if not file.exists():
        shutil.copyfile(source/'task_spec.json',where/'task_spec.json')
        joblib.dump(dict(model=model,kind='pair_text',candidate=name),file)
    resources,_=model_resources(model)
    verify_resources(resources)
    frozen_save(where/'final_selection.json',dict(family=name,selected_experiment=name,model_file=file.name,
        model_format='candidate_bundle_v1',refit=True,test_accessed=False,task_spec_sha256=ab_run.sha(where/'task_spec.json'),
        model_sha256=ab_run.sha(file),training_rows=6608,choice_source='four-opening matched checkpoint replay',
        resources=resources,component_sources=[r for m,r in pairs]))
    old_export=verified_export(where,inputs,env,out)
    new_export=read(batch/'exports_frozen.json')['new']
    frozen_save(out/'exports_frozen.json',dict(control=old_export,adaptive=new_export,
        decision=ab_run.sha(out/'decision_frozen.json')))
    y=np.asarray(read(inputs/'scoring_only.json')['labels'])
    g=np.asarray(read(inputs/'groups_only.json'))
    expected=read(old/'protocol.json')['packets']
    check(len(y)==len(g)==1693 and all(ab_run.sha(inputs/n)==expected[n] for n in ('features.zip','groups_only.json','scoring_only.json')),
          'Unexpected evaluation packet')
    control=classes(out/'control_export_0/rows.csv',old_export,len(y))==y
    current=classes(batch/'autonomous_export_0/rows.csv',new_export,len(y))==y
    result=dict(original_members=members,adaptive_members=read(batch/'results.json')['members'],rows=len(y),
        control_correct=int(control.sum()),adaptive_correct=int(current.sum()),gained=int((current&~control).sum()),
        lost=int((~current&control).sum()),delta=paired(current.astype(int)-control,g),
        original_equivalent_fits=fits_equivalent,physical_new_fits=newly,model_api=0,course_designated_validation=False,
        formal_test=False,source='archived four-opening controller and exact matched observations')
    frozen_save(out/'results.json',result)
    frozen_save(out/'completion.json',dict(status='complete',result=ab_run.sha(out/'results.json'),physical_fits=newly,model_api=0))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','batch','old','earlier','deeper','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    env=offline_environment(a.workspace.resolve())
    with patch.dict(os.environ,env):
        run(*(getattr(a,k).resolve() for k in ('inputs','batch','old','earlier','deeper','output')),env)
