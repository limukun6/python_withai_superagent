"""Supply the single missing real confirmation in an archived joint-catalog decision."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_b_coverage_feedback import next_step, trusted_parts
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.replay_b_same_seed_opening import lookup_confirmation, full_model
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, read
from superagent import ab_adaptive_opening, ab_complement, ab_run
from superagent.ab_methods import TRANSFER_METHODS
from superagent.data import load_development_task
from superagent.external_assets import model_resources, verify_resources
from superagent.splits import sample_groups


def run(inputs, previous, original, audit, encoders, vote, matched, out, env):
    out.mkdir(parents=True,exist_ok=True)
    e5=TRANSFER_METHODS['deep-e5'][1]
    proof=read(audit/'results.json')['additional_seed']
    source,other=previous/'autonomous',original/'autonomous'
    check(proof['missing']==dict(candidate=e5,split='confirmation_1') and not proof['complete'],'Wrong or already complete archived state')
    frozen_save(out/'protocol.json',dict(runner=ab_run.sha(Path(__file__)),plan=ab_run.sha(ROOT/'docs/batches/B_missing_coverage_confirmation_1123_zh.md'),
        prior=ab_run.sha(audit/'results.json'),primary=ab_run.sha(source/'primary.joblib'),confirmation=ab_run.sha(source/'confirmation_1.joblib'),
        fit_cap=1,model_api_cap=0))
    task=load_development_task(inputs/'anonymous.zip')
    t=trusted_parts(source,sample_groups(task,'shared_text'))
    caps=read(source/'run_manifest.json')['capabilities']
    caps[e5]=read(other/'run_manifest.json')['capabilities'][e5]
    state=read(source/'state_5.json')
    state['available_candidates'].append(caps[e5])
    state['budget'].update(fit_limit=10,remaining_fit_allowance=10-state['budget']['attempted_classifier_fits'])
    receipts,actions={},[]
    for _ in range(6):
        action=next_step(state,caps,t)
        actions.append(action)
        if action['action']=='stop':
            break
        name,s=action['candidate_id'],action['split_id']
        number=len(state['history'])+1
        if action['action']=='explore':
            check(name==e5 and number==5 and s=='primary' and ab_run.sha(source/(s+'.joblib'))==ab_run.sha(other/(s+'.joblib')),
                  'Wrong fifth archived observation')
            row=next(r for r in read(other/'history.json') if (r['candidate'],r['split_id'])==(name,s))
            obs=next(r for r in read(other/'prediction_evidence.json') if (r['candidate_id'],r['split_id'])==(name,s))
            receipts[s+'_'+name]=dict(source='prior native exploratory trial',physical_new_fits=0,predictions=obs['prediction_sha256'])
        else:
            check(s=='confirmation_1' and name in proof['decision']['comparison_members'] and s+'_'+name not in receipts,
                  'Unexpected confirmation')
            art,receipts[s+'_'+name]=lookup_confirmation(name,source,caps[name],out)
            facts,raw=art['metrics'],art['prediction_record']
            check(facts['resolved_configuration']['params']==caps[name]['params'] and
                  raw['evaluation_payload_sha256']==t[s]['evaluation_payload_sha256'],'Wrong matched worker')
            row=dict(candidate=name,params=caps[name]['params'],split_id=s,proposal=None,status='completed',**facts,
                evaluation_payload_sha256=raw['evaluation_payload_sha256'],label_sha256=raw['label_sha256'],
                row_ids_sha256=ab_run.fingerprint(t[s]['row_ids']))
            obs=dict(candidate_id=name,split_id=s,row_id_scope=ab_run.ROW_SCOPE,row_ids=t[s]['row_ids'],
                predictions=raw['predictions'].tolist(),**{k:raw[k] for k in ('prediction_sha256','label_sha256','evaluation_payload_sha256','score_record')})
        state['history'].append(dict(row,trial=number))
        state['prediction_evidence'].append(dict(obs,trial=number))
        state['available_candidates']=[cap for cap in state['available_candidates'] if cap['id']!=name]
        state['budget']['remaining_fit_allowance']-=1
        for field in ('attempted_classifier_fits','completed_classifier_fits'):
            state['budget'][field]+=1
    decision=ab_adaptive_opening.review(state,t)
    members=decision['members']
    check(actions[-1]['action']=='stop' and not decision.get('pending') and sum(r['physical_new_fits'] for r in receipts.values())==
        sum(len(f.read_text().splitlines()) for f in out.glob('*/physical.jsonl'))==1,'Incomplete or incorrect physical count')
    equivalents=len(state['history'])+len(members)
    check(equivalents<=10,'Exceeded per-path budget')
    frozen_save(out/'decision_frozen.json',dict(decision=decision,actions=actions,receipts=receipts,fit_equivalents=equivalents,
                                                actual_new_fits=1,source_state=ab_run.sha(source/'state_5.json')))
    models=[]
    provenance=[]
    for name in members:
        if name==e5:
            folder=vote/'e5_full'
            file=folder/'selected_model.joblib'
            spec=read(folder/'final_selection.json')
            worker_file=vote/'deep-e5/worker.joblib'
            facts=joblib.load(worker_file)['metrics']
            check(spec['model_sha256']==ab_run.sha(file)==read(vote/'models_frozen.json')['e5_full']
                  and spec['task_spec_sha256']==ab_run.sha(source/'task_spec.json')
                  and facts['resolved_configuration']['params']==caps[name]['params'] and facts['training_rows']==6608
                  and read(vote/'protocol.json')['payloads']['deep-e5']==ab_run.sha(source/'refit_data.joblib'),
                  'Incompatible frozen E5 fit')
            model=joblib.load(file)['model']
            receipt=dict(component=name,model_bundle=ab_run.sha(file),full_training_payload=ab_run.sha(source/'refit_data.joblib'))
        else:
            model,receipt=full_model(name,[source,other,encoders/'deep-e5'],source,caps[name])
        models.append(model)
        provenance.append(receipt)
    model=models[0] if len(models)==1 else ab_complement.PreferenceProbabilityVote('pair_text',models,np.array([0,1]))
    family=members[0] if len(models)==1 else 'equal_soft_vote'
    folder=out/'delivered'
    folder.mkdir(exist_ok=True)
    file=folder/'selected_model.joblib'
    if not file.exists():
        shutil.copyfile(source/'task_spec.json',folder/'task_spec.json')
        joblib.dump(dict(model=model,kind='pair_text',candidate=family),file)
    resources,_=model_resources(model)
    verify_resources(resources)
    frozen_save(folder/'final_selection.json',dict(family=family,selected_experiment=family,model_file=file.name,
        model_format='candidate_bundle_v1',refit=True,test_accessed=False,task_spec_sha256=ab_run.sha(folder/'task_spec.json'),
        model_sha256=ab_run.sha(file),training_rows=6608,choice_source='completed archived joint decision',
        resources=resources,components=provenance))
    export=verified_export(folder,inputs,env,out)
    fixed_hash=read(matched/'exports_frozen.json')['new_vote']
    frozen_save(out/'exports_frozen.json',dict(new=export,fixed_combo=fixed_hash,decision=ab_run.sha(out/'decision_frozen.json')))
    expected=read(previous/'protocol.json')['packets']
    check(all(ab_run.sha(inputs/n)==expected[n] for n in ('features.zip','groups_only.json','scoring_only.json')),'Scoring data changed')
    y=np.asarray(read(inputs/'scoring_only.json')['labels'])
    pred=classes(out/'delivered_export_0/rows.csv',export,len(y))
    fixed=classes(matched/'e5_familiarity_export_0/rows.csv',fixed_hash,len(y))
    hits=int((pred==y).sum())
    check(len(y)==1693 and int((fixed==y).sum())==1237,'Fixed reference differs')
    result=dict(seed=18643,members=members,proposed=decision['proposed_pair'],matched_on_confirmation=decision['confirmation_correct'],
        confirmation_vote=decision['confirmation_combination'],primary_single_leader=decision['anchor'],
        rows=len(y),correct=hits,identical_to_frozen_research_vote=export==fixed_hash,eligible_for_cold_run=hits>=1235,
        physical_new_fits=1,equivalent_path_fits=equivalents,model_api_calls=0,course_designated_validation=False,
        formal_test=False,cold_native_run=False)
    frozen_save(out/'results.json',result)
    frozen_save(out/'completion.json',dict(result=ab_run.sha(out/'results.json'),status='complete',physical_fits=1,model_api=0))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','previous','original','audit','encoders','vote','matched','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    env=offline_environment(a.workspace.resolve())
    with patch.dict(os.environ,env):
        run(*(getattr(a,k).resolve() for k in ('inputs','previous','original','audit','encoders','vote','matched','output')),env)
