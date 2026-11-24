"""Budget-matched method search using only the original primary out-of-training predictions."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.run_b_varied_preferences import ARMS, ROOT, SLOTS, bucket, read, require, save
from superagent.ab_method_search import rank, signature
from superagent.ab_run import sha
from superagent.data import load_development_task
from superagent.prediction_evidence import fingerprint
from superagent.scheduler_bridge import fingerprint as decision_digest
from superagent.splits import sample_groups, take
from superagent.task_profile import training_facts


PLAN=ROOT/'docs/batches/B_autonomous_feedback_replay_20261124_zh.md'
ARMS_BY_BANK={'titles':('reference','expanded'),'movie':('transfer',)}
POLICIES=('with_feedback','distance_only')


def load_bank(out,task_name):
    task=load_development_task(out/'inputs'/SLOTS[task_name]/'train.zip')
    protocol=read(out/'models_frozen.json')
    predictions,caps,origins,duplicates={},{},{},[]
    shared=None
    for arm in ARMS_BY_BANK[task_name]:
        folder=out/arm
        require(ARMS[arm][0]==task_name and sha(folder/'frozen.json')==protocol['arms'][arm],'Bank source differs')
        snapshot=read(folder/'frozen.json')
        split=read(folder/'agent/primary_split.json')
        require(shared in (None,split),'Different fit or primary rows in pooled bank')
        shared=split
        events=read(folder/'agent/prediction_evidence.json')
        for row in snapshot['trials']:
            if row['split']!='primary':
                continue
            name=row['id']
            entry,=[e for e in events if e['candidate_id']==name and e['split_id']=='primary']
            values=np.asarray(entry['predictions'],dtype=int)
            require(entry['row_ids']==split['search_indices'] and fingerprint(values.tolist())==entry['prediction_sha256'] and
                    row['payload']==entry['evaluation_payload_sha256']==split['evaluation_payload_sha256'] and
                    fingerprint(task.y_train[entry['row_ids']].tolist())==entry['label_sha256'],'Invalid bank prediction')
            if name in predictions:
                require(caps[name]==snapshot['capabilities'][name],'Unequal duplicate configuration')
                duplicates.append(dict(candidate=name,kept=origins[name],also=arm,changed_rows=int(np.sum(predictions[name]!=values))))
            else:
                predictions[name]=values
                caps[name]=snapshot['capabilities'][name]
                origins[name]=arm
    fit=np.asarray(shared['train_indices'])
    held=np.asarray(shared['search_indices'])
    groups=sample_groups(task,'shared_text')
    require(set(groups[fit]).isdisjoint(groups[held]),'Primary overlaps model training')
    profile=training_facts(replace(task,X_train=take(task.X_train,fit),y_train=task.y_train[fit]))
    return caps,predictions,task.y_train[held],groups[held],profile,dict(primary=shared['row_ids_sha256'],origins=origins,duplicates=duplicates)


def choose(caps,predictions,feedback,feedback_y,profile,budget,policy):
    # The caller does not pass evaluation answers or other candidates' performance to the selector.
    state=dict(training_profile=profile,prediction_evidence=[])
    measured=[]
    trajectory=[]
    scores={}
    for trial in range(1,budget+1):
        available=[n for n in caps if n not in scores]
        result=rank(state,caps,available,measured)
        name=result['chosen']
        local=predictions[name][feedback]
        score=float(np.mean(local==feedback_y))
        scores[name]=score
        trajectory.append(dict(candidate=name,feedback_correct=int(np.sum(local==feedback_y)),
                               choice_sha256=decision_digest(result)))
        real=policy=='with_feedback'
        measured.append(dict(trial=trial,candidate=name,search_score=score if real else .5))
        state['prediction_evidence'].append(dict(trial=trial,predictions=local.tolist() if real else [0]*len(local)))
    chosen=min(scores,key=lambda n:(-scores[n],signature(caps[n]),n))
    return dict(trajectory=trajectory,selected=chosen,observations=len(scores))


def run(out):
    snapshots={}
    banks={}
    for name in ARMS_BY_BANK:
        caps,predictions,labels,groups,profile,meta=load_bank(out,name)
        banks[name]=(predictions,labels,groups)
        fold=np.asarray([bucket(g,'b-varied-feedback-fold-v1/',4) for g in groups])
        budget=min(3,len(caps)-1)
        entry=dict(**meta,candidate_count=len(caps),budget=budget,rows=len(labels),groups=len(set(groups)),
                   fold_sha256=fingerprint(fold.tolist()),folds=[])
        snapshots[name]=entry
        if len(caps)<3:
            entry['status']='insufficient_observed_methods'
            continue
        for i in range(4):
            feedback=np.flatnonzero(fold!=i)
            held=np.flatnonzero(fold==i)
            require(len(feedback)>0 and len(held)>0 and set(groups[feedback]).isdisjoint(groups[held]),'Feedback overlaps evaluation')
            entry['folds'].append(dict(fold=i,feedback_rows=len(feedback),held_rows=len(held),
                choices={p:choose(caps,predictions,feedback,labels[feedback],profile,budget,p) for p in POLICIES}))
    decision_file=out/'feedback_paths.json'
    save(decision_file,dict(plan=sha(PLAN),runner=sha(Path(__file__)),banks=snapshots))
    locked=read(decision_file)
    results={}
    for name,bank in locked['banks'].items():
        if not bank['folds']:
            results[name]={'status':bank['status']}
            continue
        predictions,labels,groups=banks[name]
        fold=np.asarray([bucket(g,'b-varied-feedback-fold-v1/',4) for g in groups])
        pairs=[]
        choices_changed=0
        winners_changed=0
        for part in bank['folds']:
            held=fold==part['fold']
            active,control=(part['choices'][p] for p in POLICIES)
            choices_changed+=([e['candidate'] for e in active['trajectory']]!=[e['candidate'] for e in control['trajectory']])
            winners_changed+=active['selected']!=control['selected']
            before=predictions[control['selected']][held]==labels[held]
            after=predictions[active['selected']][held]==labels[held]
            pairs.append(dict(fold=part['fold'],rows=int(sum(held)),feedback_correct=int(after.sum()),
                              distance_correct=int(before.sum()),gained=int((~before&after).sum()),lost=int((before&~after).sum())))
        results[name]={key:sum(p[key] for p in pairs) for key in ('rows','feedback_correct','distance_correct','gained','lost')}
        results[name].update(fold_metrics=pairs,changed_search_folds=choices_changed,changed_winner_folds=winners_changed)
    save(out/'feedback_results.json',dict(paths_sha256=sha(decision_file),banks=results,new_supervised_fits=0,new_api_calls=0))
    for name,r in results.items():
        print(name,{k:v for k,v in r.items() if k!='fold_metrics'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
