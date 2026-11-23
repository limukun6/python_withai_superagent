"""Probe feedback and resource dependence; replay an additional archived joint opening without fitting."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read
from superagent import ab_run, ab_adaptive_opening
from superagent.ab_methods import TRANSFER_METHODS
from superagent.data import load_development_task
from superagent.splits import sample_groups


def trusted_parts(folder, groups):
    ret={}
    for s in ('primary','confirmation_1','confirmation_2'):
        source=folder/(s+'.joblib')
        rec=read(folder/(s+'_split.json'))
        data=joblib.load(source)
        ids=rec['search_indices']
        check(rec['search_rows_sharing_text']==0 and ab_run.sha(source)==rec['evaluation_payload_sha256']
              and rec['label_sha256']==ab_run.fingerprint(data['y_search']) and rec['row_ids_sha256']==ab_run.fingerprint(ids),
              'Invalid local split')
        ret[s]=dict(row_ids=ids,labels=data['y_search'].tolist(),groups=groups[ids].tolist(),
            evaluation_payload_sha256=rec['evaluation_payload_sha256'],tie_label_one=[a<b for a,b in data['X_search']])
    return ret


def next_step(state,caps,trusted):
    return ab_adaptive_opening.action(state,caps,trusted,cover_untried_encoder=True)


def interventions(run,groups):
    state=read(run/'state_5.json')
    caps=read(run/'run_manifest.json')['capabilities']
    t=trusted_parts(run,groups)
    actual=next_step(state,caps,t)
    check(actual==read(run/'actions.json')[4] and actual['action']=='explore','Original fifth action not reproduced')
    mapping={name:f'candidate-{i:02d}' for i,name in enumerate(reversed(caps))}
    renamed=deepcopy(state)
    for entry in renamed['available_candidates']:
        entry['id']=mapping[entry['id']]
    for row in renamed['history']:
        row['candidate']=mapping[row['candidate']]
    for rec in renamed['prediction_evidence']:
        name=mapping[rec['candidate_id']]
        rec['candidate_id']=rec['score_record']['candidate_id']=name
    renamed_caps={mapping[n]:dict(c,id=mapping[n]) for n,c in caps.items()}
    blind=next_step(renamed,renamed_caps,t)
    check(blind==dict(actual,candidate_id=mapping[actual['candidate_id']]),'Rename changed route selection')
    unseen={e['id'] for e in ab_adaptive_opening.unseen_encoder_candidates(state,caps)}
    unavailable=deepcopy(state)
    unavailable['available_candidates']=[e for e in unavailable['available_candidates'] if e['id'] not in unseen]
    absent=next_step(unavailable,{n:c for n,c in caps.items() if n not in unseen},t)
    reduced=deepcopy(state)
    reduced['budget']['fit_limit']-=1
    reduced['budget']['remaining_fit_allowance']-=1
    limited=next_step(reduced,caps,t)
    check(absent['action']==limited['action']=='stop','A fixed result was requested despite missing capacity or asset')
    damaged=deepcopy(state)
    damaged['prediction_evidence'][0]['score_record']['scores_sha256']='0'*64
    try:
        next_step(damaged,caps,t)
    except ValueError as exc:
        rejected=str(exc)
    else:
        raise ValueError('Unbound feedback accepted')
    return dict(observed=actual,renamed_action=blind,renamed_equivalent=True,missing_asset=absent,
                reduced_budget=limited,corrupt_feedback_rejection=rejected,source_state=ab_run.sha(run/'state_5.json'))


def archived_joint(prior, old, groups):
    source, other=prior/'autonomous',old/'autonomous'
    t=trusted_parts(source,groups)
    caps=read(source/'run_manifest.json')['capabilities']
    e5=TRANSFER_METHODS['deep-e5'][1]
    other_caps=read(other/'run_manifest.json')['capabilities']
    check(e5 not in caps and all(t[s]['evaluation_payload_sha256']==ab_run.sha(other/(s+'.joblib'))
          for s in t),'Cannot combine distinct partitions')
    state=read(source/'state_5.json')
    caps[e5]=other_caps[e5]
    state['available_candidates'].append(caps[e5])
    state['budget']['fit_limit']=10
    state['budget']['remaining_fit_allowance']=10-state['budget']['attempted_classifier_fits']
    rows={}
    for directory in (source,other):
        hist=read(directory/'history.json')
        existing=read(directory/'prediction_evidence.json')
        for row,obs in zip(hist,existing,strict=True):
            name,s=row['candidate'],row['split_id']
            key=name,s
            check(obs['candidate_id']==name and obs['split_id']==s and row['params']==caps[name]['params']
                  and obs['evaluation_payload_sha256']==t[s]['evaluation_payload_sha256'] and row['status']=='completed',
                  'Archived history differs from its trusted capability or split')
            if key in rows:
                check(obs['prediction_sha256']==rows[key][1]['prediction_sha256'] and obs['score_record']['scores_sha256']
                      ==rows[key][1]['score_record']['scores_sha256'],'Shared archived results differ')
            rows.setdefault(key,(row,obs))
    actions=[]
    unavailable=None
    for _ in range(6):
        action=next_step(state,caps,t)
        actions.append(action)
        if action['action']=='stop':
            break
        n,s=action['candidate_id'],action['split_id']
        if (n,s) not in rows:
            unavailable={'candidate':n,'split':s}
            break
        row,obs=rows[n,s]
        number=len(state['history'])+1
        state['history'].append(dict(row,trial=number))
        state['prediction_evidence'].append(dict(obs,trial=number))
        state['available_candidates']=[e for e in state['available_candidates'] if e['id']!=n]
        for key in ('attempted_classifier_fits','completed_classifier_fits'):
            state['budget'][key]+=1
        state['budget']['remaining_fit_allowance']-=1
    check(actions[0]['action']=='explore' and actions[0]['candidate_id']==e5 and (unavailable or actions[-1]['action']=='stop'),
          'Missing replay endpoint')
    proof=ab_adaptive_opening.review(state,t)
    return dict(seed=18643,replayed_actions=actions,verified_observations=len(state['prediction_evidence']),
                decision=proof,missing=unavailable,complete=unavailable is None,development_reserve_scored=False)


def run(inputs,current,prior,old,out):
    out.mkdir(parents=True,exist_ok=True)
    packet=inputs/'anonymous.zip'
    check(ab_run.sha(packet)==read(current/'protocol.json')['training'],'Wrong training groups')
    groups=sample_groups(load_development_task(packet),'shared_text')
    frozen_save(out/'protocol.json',dict(code=ab_run.sha(Path(__file__)),plan=ab_run.sha(ROOT/'docs/batches/B_coverage_feedback_checks_1123_zh.md'),
        current_action_source=ab_run.sha(current/'autonomous/state_5.json'),earlier=ab_run.sha(prior/'results.json'),
        alternate=ab_run.sha(old/'results.json'),fit_cap=0,model_api_cap=0))
    checks=interventions(current/'autonomous',groups)
    replay=archived_joint(prior,old,groups)
    result=dict(measured_and_synthetic_checks=checks,additional_seed=replay,actual_new_fits=0,model_api=0,
        development_reserve_scored=False,official_validation=False,formal_test=False)
    frozen_save(out/'results.json',result)
    frozen_save(out/'completion.json',dict(status='complete',result=ab_run.sha(out/'results.json'),physical_fits=0,model_api=0))
    print('resource interventions',checks['missing_asset']['action'],checks['reduced_budget']['action'],
          'older seed',replay['complete'],replay['missing'],replay['decision']['members'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','current','prior','old','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    run(*(getattr(a,k).resolve() for k in ('inputs','current','prior','old','output')))
