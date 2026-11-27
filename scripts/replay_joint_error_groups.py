"""Test the selected error group against frozen, validated training-fold predictions with zero new fits."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.compare_joint_feedback import CASES
from superagent.autonomous_run import planner_observation
from superagent.conditional_review import compare, group_observation
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.opening_evidence import subset


def observable(check):
    return check['kind']=='slice' or check['kind']=='all_of' and all(observable(part) for part in check['parts'])


def observations(directory, fold, task, metric, tested):
    split=read(directory/(fold+'_split.json'))
    ids=np.asarray(split['held'],dtype=int)
    payload=directory/(fold+'.joblib')
    record=dict(labels=task.y_train[ids],sha256=sha(payload))
    require(record['sha256']==split['payload_sha256'],'Wrong fold payload')
    bank={}
    def add(name, path):
        bundle=joblib.load(path)
        meta=bundle['metrics']
        require(meta['resolved_configuration']['id']==name, 'Mismatched candidate identity')
        bank[name]=planner_observation(bundle,meta,record,metric)
    for row in read(directory/'history.json'):
        if row['status']=='completed' and row['split_id']==fold:
            add(row['candidate'], directory/f"trial_{row['trial']}.joblib")
    probe=directory/('opening_probe_'+fold+'.joblib')
    if probe.exists():add(tested,probe)
    return ids,bank


def switch(base,other,inside,metric):
    require(base['row_key']==other['row_key'] and np.array_equal(base['labels'],other['labels']), 'Unmatched rows')
    values=np.where(inside,other['predictions'],base['predictions'])
    scores=None
    if base['probabilities'] is not None and other['probabilities'] is not None:
        scores=np.where(inside[:,None],other['probabilities'],base['probabilities'])
    value=metric.evaluate(base['labels'],values,scores,list(metric.labels) if scores is not None else None)['score']
    return dict(base,predictions=values,probabilities=scores,score=value)


def replay(folder):
    directory=folder/'feedback-local/run/agent'
    opening=read(directory/'task_opening.json')
    result=dict(check=opening.get('check'),verdict=opening['check_status'])
    if result['verdict']!='supported': return dict(result,outcome='no_corroborated_group')
    brief=read(directory/'opening_brief.json')
    key=result['check']
    definition=brief['checks'].get(key)
    if definition is None:
        from superagent.feedback_hypothesis import joint_check
        actual,definition=joint_check(key.split('&'),brief['checks'])
        require(actual==key,'Unmatched composite check')
    if not observable(definition):return dict(result,outcome='group_requires_unknown_label')
    task=load_training_task(folder/'task.zip',read_declarations=True)
    metric=MetricSpec.for_task(task)
    base=opening['incumbent_members']
    primary,confirm=opening['confirmation']
    def evidence(fold):
        ids,bank=observations(directory,fold,task,metric,opening['tested_candidate'])
        return set(ids.tolist()),group_observation(bank,base,metric),subset(task,metric,ids,definition),bank
    primary_ids,reference,inside,bank=evidence(primary)
    wrong=inside & (reference['predictions']!=reference['labels'])
    rescued=np.zeros_like(wrong)
    candidates={}
    for name,row in bank.items():
        candidates[name]=compare(primary,reference,switch(reference,row,inside,metric))
        rescued |= wrong & (row['predictions']==row['labels'])
    result.update(primary_candidates=len(candidates),group_rows=int(sum(inside)),incorrect=int(sum(wrong)),
                  rescued_somewhere=int(sum(rescued)),primary_deltas={n:r['delta'] for n,r in sorted(candidates.items())})
    positive=[n for n,r in candidates.items() if r['delta']>1e-12]
    if not positive:return dict(result,outcome='no_primary_improvement')
    winner=min(positive,key=lambda n:(-candidates[n]['delta'],n))
    result.update(primary_winner=winner,primary=candidates[winner])
    confirm_ids,reference,inside,bank=evidence(confirm)
    require(primary_ids.isdisjoint(confirm_ids),'Confirmation rows overlap')
    if winner not in bank:return dict(result,outcome='missing_independent_observation')
    checked=compare(confirm,reference,switch(reference,bank[winner],inside,metric))
    gain=checked['delta']
    passed=gain>0 and (gain+candidates[winner]['delta'])/2>=.003
    return dict(result,outcome='positive_on_both' if passed else 'failed_confirmation',confirmation=checked)


def main(root):
    study=root/'development'
    output=study/'slice_replay.json'
    results={case:replay(study/case) for case in CASES}
    save(output,dict(protocol=sha(ROOT/'research/joint_feedback_1127/slice_replay_protocol.json'),
                     results=results,physical_fits=0,api_requests=0,development_evaluation_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=lambda s:Path(s).resolve())
    main(p.parse_args().root)
