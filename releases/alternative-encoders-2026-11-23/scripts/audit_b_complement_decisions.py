"""Probe state-dependent decisions using measured actions and separately identified synthetic controls."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run,ab_complement
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.score_evidence import make_score_record


def mapped(state,names):
    result=deepcopy(state)
    for r in result['history']:
        r['candidate']=names[r['candidate']]
    for p in result['prediction_evidence']:
        p['candidate_id']=names[p['candidate_id']]
        p['score_record']['candidate_id']=p['candidate_id']
    return result


def synthetic(state,trusted,mode):
    transformed=deepcopy(state)
    y=np.asarray(trusted['primary']['labels'])
    ranks=np.arange(len(y))%10
    for index,(row,p) in enumerate(zip(transformed['history'],transformed['prediction_evidence'],strict=True)):
        anchor=mode=='identical' or index in (0,2,3)
        wrong=(ranks<2) if anchor else ((ranks>=2)&(ranks<5))
        confidence=np.where(wrong,.55,.97) if anchor else np.where(wrong,.55,.96)
        hard=np.where(wrong,1-y,y)
        positive=np.where(hard==1,confidence,1-confidence)
        p['predictions']=hard.tolist()
        p['prediction_sha256']=fingerprint(hard)
        scores=np.column_stack([1-positive,positive])
        p['score_record']=make_score_record(p,scores,np.array([0,1]),MetricSpec.from_payload(
            {'metric_spec':state['task']['metric']}),p['candidate_id'])
        row['search_score']=row['search_accuracy']=float((hard==y).mean())
    return transformed


def run(folder,out):
    stage=folder/'state_5.json'
    initial_state=ab_run.sha(stage)
    state=read(stage)
    check(len(state['history'])==len(state['prediction_evidence'])==4,'Requires first four actual results')
    caps=read(folder/'run_manifest.json')['capabilities']
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        data=joblib.load(folder/(split+'.joblib'))
        rec=read(folder/(split+'_split.json'))
        trusted[split]=dict(labels=data['y_search'].tolist(),row_ids=rec['search_indices'],
            evaluation_payload_sha256=ab_run.sha(folder/(split+'.joblib')),
            tie_label_one=[l<r for l,r in data['X_search']])
    observed=ab_complement.action(state,caps,trusted)
    check(observed==read(folder/'actions.json')[4],'Observed next action did not reproduce')
    names={p['candidate_id']:f'method_{i}' for i,p in enumerate(reversed(state['prediction_evidence']))}
    blind=ab_complement.action(mapped(state,names),caps,trusted)
    check(blind==dict(observed,candidate_id=names.get(observed['candidate_id'])),'Candidate ID sensitivity')
    same=ab_complement.action(synthetic(state,trusted,'identical'),caps,trusted)
    check(same['action']=='stop','Identical evidence must retain a single model')
    artificial=synthetic(state,trusted,'complement')
    proof=ab_complement.review(artificial,trusted)
    choice=ab_complement.action(artificial,caps,trusted)
    check(choice['action']=='confirm' and proof['partner']==state['history'][1]['candidate'],'Missed complementary errors')
    rejected={}
    for key in ('probability_fingerprint','row_order'):
        damaged=deepcopy(state)
        entry=damaged['prediction_evidence'][0]
        if key=='row_order':
            entry['row_ids'][0],entry['row_ids'][1]=entry['row_ids'][1],entry['row_ids'][0]
        else:
            entry['score_record']['scores_sha256']='0'*64
        try:
            ab_complement.review(damaged,trusted)
        except ValueError as exc:
            rejected[key]=str(exc)
        else:
            raise ValueError('Corrupt evidence accepted')
    check(ab_run.sha(stage)==initial_state,'Original state changed')
    save(out,dict(source_state=initial_state,observed_action=observed,anonymous_replay=blind,
        mapped_exactly=True,synthetic_identical_action=same['action'],synthetic_complement_action=choice['action'],
        synthetic_partner=proof['partner'],corrupt_evidence_rejected=rejected,new_classifier_fits=0,model_api_requests=0,
        synthetic_feedback_used_for_performance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.run.resolve(),a.output.resolve())
