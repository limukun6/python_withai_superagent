"""Replay the full native menus until a policy actually requests an unavailable fit."""
import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_shared_preference_pair import environment
from superagent import ab_run
from superagent.ab_catalog import untried_candidates
from superagent.ab_methods import PREFERENCE_CATALOG_ENV,TEXT_DEEP
from superagent.ab_scheduler import next_action,selection_evidence
from superagent.prediction_evidence import fingerprint


def replay(inputs,root,out):
    old=inputs/'original'
    evidence=read(old/'prediction_evidence.json')
    predictions={(v['candidate_id'],v['split_id']):v['predictions'] for v in evidence}
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        part=read(old/(split+'_split.json'))
        payload=old/(split+'.joblib')
        values=joblib.load(payload)
        check(ab_run.sha(payload)==part['evaluation_payload_sha256'] and
              fingerprint(values['y_search'])==part['label_sha256'],'Historical split does not match its payload')
        trusted[split]=dict(row_ids=part['search_indices'],evaluation_payload_sha256=part['evaluation_payload_sha256'],
                            labels=values['y_search'].tolist())
    deep=read(inputs/'components/inner_evidence.json')
    delivery=read(REPO/'research/b_accuracy_frontier_1123/native_delivery.json')
    check(deep['held_row_ids']==trusted['primary']['row_ids'] and fingerprint(deep['observed_predictions']['deep'])==
          delivery['checks'][TEXT_DEEP]['prediction_sha256'],'Deep bank differs from native CLI')
    predictions[TEXT_DEEP,'primary']=deep['observed_predictions']['deep']
    rec={}
    for mode in ('standard','deep','deep-edit'):
        state=read(old/'state_1.json')
        with patch.dict(os.environ,environment(root)|{PREFERENCE_CATALOG_ENV:mode}):
            caps=ab_run.registered_capabilities(state['task'],6608)
            actions=[]
            selected=None
            while len(actions)<=11:
                state['available_candidates']=untried_candidates(caps,state['history'])
                action=next_action(state,capabilities=caps,trusted_partitions=trusted)
                actions.append(action)
                if action['action']=='stop':
                    selected=selection_evidence(state,trusted_partitions=trusted)['incumbent']
                    break
                name,s=action['candidate_id'],action['split_id']
                if (name,s) not in predictions:
                    break
                trial=len(actions)
                t=trusted[s]
                p=predictions[name,s]
                score=float(np.mean(np.array(p)==t['labels']))
                state['history'].append(dict(trial=trial,candidate=name,split_id=s,params=caps[name]['params'],
                    status='completed',primary_metric='accuracy',metric_spec=state['task']['metric'],
                    classifier_fits=1,search_score=score,search_accuracy=score))
                state['prediction_evidence'].append(dict(trial=trial,candidate_id=name,split_id=s,
                    row_id_scope='positions within this run training input',row_ids=t['row_ids'],predictions=p,
                    evaluation_payload_sha256=t['evaluation_payload_sha256'],label_sha256=fingerprint(t['labels']),
                    prediction_sha256=fingerprint(p)))
                state['budget'].update(attempted_classifier_fits=trial,completed_classifier_fits=trial,
                                       remaining_fit_allowance=11-trial)
            rec[mode]=dict(size=len(caps),actions=actions,selected=selected,
                           missing_observation=None if selected else [actions[-1]['candidate_id'],actions[-1]['split_id']])
    keys=('action','candidate_id','split_id')
    signatures=lambda rows:[tuple(row[k] for k in keys) for row in rows]
    check(signatures(rec['standard']['actions'])==signatures(read(old/'actions.json')),'Baseline replay differs')
    check(rec['standard']['selected']==read(old/'run_outcome.json')['selected_candidate'],'Mismatch in baseline winner')
    save(out,dict(seed=707,menus=rec,actual_new_fits=0,model_api=0,standard_actions_reproduced=True,
         input_history=ab_run.sha(old/'prediction_evidence.json'),native_preflight=ab_run.sha(
             REPO/'research/b_accuracy_frontier_1123/native_delivery.json'),source_sha256=ab_run.sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=REPO/'runs/local_components_02')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    replay(a.inputs.resolve(),a.workspace.resolve(),a.output)
