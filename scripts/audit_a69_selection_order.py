"""Posthoc same-evidence order counterexample, no training, requests or new validation scoring."""
import argparse
from copy import deepcopy
from pathlib import Path
import joblib
from scripts import a38_predict_guard
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_scheduler import selection_evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);run=a.study/'bean_1707_ab-fixed'
    history=read(run/'history.json');predictions=read(run/'prediction_evidence.json');manifest=read(run/'run_manifest.json')
    save(a.output/'protocol.json',dict(batch='A69-selection-order-audit',runner_sha256=sha(Path(__file__)),history_sha256=sha(run/'history.json'),
        plan_sha256=sha(Path(__file__).resolve().parents[1]/'docs/batches/A69_selection_order_audit_plan.md'),posthoc=True,actual_classifier_fits=0,real_api_requests=0))
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        payload=run/(split+'.joblib');trusted[split]=dict(row_ids=read(run/(split+'_split.json'))['search_indices'],
            labels=joblib.load(payload)['y_search'].tolist(),evaluation_payload_sha256=sha(payload))
    bykey={(r['candidate'],r['split_id']):r for r in history};pre='numeric_tabular_'
    order=[(pre+'logistic_C0.1','primary'),(pre+'extra_trees_leaf1','primary'),(pre+'extra_trees_leaf3','primary'),
        (pre+'extra_trees_leaf1','confirmation_1'),(pre+'logistic_C0.1','confirmation_1'),
        (pre+'extra_trees_leaf1','confirmation_2'),(pre+'logistic_C0.1','confirmation_2'),
        (pre+'logistic_C1.0','primary'),(pre+'logistic_C1.0','confirmation_1'),(pre+'logistic_C1.0','confirmation_2')]
    if set(order)!=set(bykey) or len(order)!=len(history):raise ValueError('Alternative order changes observations')
    reviews={}
    for label,rows in (('actual',history),('alternative',[bykey[k] for k in order])):
        hs=[];ps=[]
        for number,row in enumerate(rows,1):
            item=deepcopy(row);pred=deepcopy(next(p for p in predictions if p['trial']==row['trial']))
            item['trial']=number;pred['trial']=number;hs.append(item);ps.append(pred)
        state=dict(ab_schema_version=1,task=manifest['task_contract'],history=hs,prediction_evidence=ps,
            registered_confirmation_splits=['confirmation_1','confirmation_2'])
        reviews[label]=selection_evidence(state,trusted_partitions=trusted)
    result=dict(status='completed',same_observation_set=True,observation_count=len(history),primary_opening_same=True,
        actual_review=reviews['actual'],alternative_review=reviews['alternative'],
        winner_changed=reviews['actual']['incumbent']!=reviews['alternative']['incumbent'],
        actual_classifier_fits=0,real_api_requests=0,new_validation_scoring=False,
        interpretation='Sequential promotion policy depends on observation order; not a newly executed budget policy or efficacy proof')
    save(a.output/'completion.json',result);print(result['winner_changed'])
