"""Cross-fit selection over truly observed primary predictions, with no training or external held-out inputs."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.run_b_open_methods import ROOT,bin_id,read,require,save
from superagent.ab_method_search import rank,signature
from superagent.ab_run import sha
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.open_methods import capability
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups,take
from superagent.task_profile import training_facts


PLAN=ROOT/'docs/batches/B_open_allocation_replay_20261124_zh.md'


def bank(archive,paths):
    task=load_development_task(archive)
    groups=sample_groups(task,'shared_text')
    caps,preds,duplicates={},{},[]
    shared=None
    for path in paths:
        master=read(path/'run_manifest.json')
        require(master['task_archive_sha256']==sha(archive),'Incompatible source packet')
        split=read(path/'primary_split.json')
        require(shared in (None,split),'Incompatible primary row/split identities')
        shared=split
        base=master['capabilities']
        registry=path/'candidate_registry.json'
        specs=read(registry)['entries'] if registry.exists() else {}
        for event in read(path/'prediction_evidence.json'):
            if event['split_id']!='primary':
                continue
            name=event['candidate_id']
            values=np.asarray(event['predictions'])
            require(values.ndim==1 and fingerprint(event['predictions'])==event['prediction_sha256'] and
                    event['row_ids']==split['search_indices'] and event['evaluation_payload_sha256']==split['evaluation_payload_sha256'],
                    'Prediction record changed')
            cap=base.get(name) or capability(specs[name])
            if name in preds:
                require(cap==caps[name],'Same ID different configuration')
                duplicates.append(dict(candidate=name,changed_rows=int(np.sum(preds[name]!=values))))
                continue
            caps[name],preds[name]=cap,values
    inner=np.array(shared['train_indices'])
    outer=np.array(shared['search_indices'])
    require(not set(groups[inner]).intersection(groups[outer]),'Classifier fit and model-choice groups overlap')
    require(fingerprint(task.y_train[outer].tolist())==shared['label_sha256'],'Primary labels differ')
    profile=training_facts(replace(task,X_train=take(task.X_train,inner),y_train=task.y_train[inner]))
    return dict(caps=caps,preds=preds,labels=task.y_train[outer],fold=np.array([
        bin_id(str(g).encode(),'b-open-allocation-fold-20261124/',4) for g in groups[outer]]),profile=profile,
        metric=TaskSpec.from_task(task).metric,duplicates=duplicates,primary=shared['row_ids_sha256'])


def choose(pool,visible,feedback,balanced):
    caps=pool['caps']
    source={n:c['origin']=='runtime_composed' for n,c in caps.items()}
    state=dict(training_profile=pool['profile'],prediction_evidence=[])
    measured=[]
    scores={}
    for i in range(min(4,len(caps)-1)):
        eligible=[n for n in caps if n not in scores]
        result=rank(state,caps,eligible,measured)
        winner=result['chosen']
        if balanced and scores:
            average=sum(scores.values())/len(scores)
            adjustment={}
            for origin in set(source.values()):
                samples=[v for n,v in scores.items() if source[n]==origin]
                posterior=(average+sum(samples))/(1+len(samples))
                adjustment[origin]=.5*(posterior-average)+result['exploration_weight']/np.sqrt(1+len(samples))
            winner=min(result['candidates'],key=lambda x:(-x['acquisition']-adjustment[source[x['candidate']]],
                                                          x['config_sha256'],x['candidate']))['candidate']
        local=pool['preds'][winner][visible]
        scores[winner]=pool['metric'].evaluate(feedback,local)['score']
        measured.append(dict(trial=i+1,candidate=winner,search_score=scores[winner]))
        state['prediction_evidence'].append(dict(trial=i+1,predictions=local.tolist()))
    best=min(scores,key=lambda n:(-scores[n],signature(caps[n]),n))
    return dict(order=list(scores),feedback_scores=scores,winner=best)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    inputs={slot:(a.matrix/'inputs'/slot/'train.zip',[a.matrix/(slot+'_'+m)/'agent' for m in ('evidence','open')]) for slot in ('q1','q2','q3')}
    inputs.update(r1=(a.released/'inputs/r1/train.zip',[a.old_table/'path/agent',a.released/'r1_open/agent']),
                  r2=(a.released/'inputs/r2/train.zip',[a.old_titles/'expanded/agent',a.released/'r2_open/agent']))
    pools={s:bank(*v) for s,v in inputs.items()}
    paths={}
    for slot,pool in pools.items():
        entry=dict(candidates=len(pool['caps']),primary=pool['primary'],duplicates=pool['duplicates'],
                   classes=list(pool['metric'].labels),metric=pool['metric'].name,folds=[])
        for fold in range(4):
            visible=pool['fold']!=fold
            require(visible.any() and (~visible).any() and len(pool['caps'])>=3,'Insufficient observed bank')
            y=pool['labels'][visible]
            entry['folds'].append({name:choose(pool,visible,y,flag) for name,flag in (('original',False),('balanced',True))})
        paths[slot]=entry
    save(a.out/'paths.json',dict(plan=sha(PLAN),runner=sha(Path(__file__)),tasks=paths))
    locked=read(a.out/'paths.json')
    results={}
    for slot,task in locked['tasks'].items():
        pool=pools[slot]
        y=pool['labels']
        predicted={p:np.empty_like(y) for p in ('original','balanced')}
        changed_orders=changed_winners=0
        for i,choice in enumerate(task['folds']):
            held=pool['fold']==i
            changed_orders+=choice['original']['order']!=choice['balanced']['order']
            changed_winners+=choice['original']['winner']!=choice['balanced']['winner']
            for policy in predicted:
                predicted[policy][held]=pool['preds'][choice[policy]['winner']][held]
        scores={name:pool['metric'].evaluate(y,p)['score'] for name,p in predicted.items()}
        results[slot]=dict(**scores,delta=scores['balanced']-scores['original'],rows=len(y),changed_orders=changed_orders,
                           changed_winners=changed_winners,new_fits=0)
    deltas=[r['delta'] for r in results.values()]
    save(a.out/'results.json',dict(paths_sha256=sha(a.out/'paths.json'),tasks=results,
        advance_to_training=sum(d>0 for d in deltas)>=3 and min(deltas)>=-.02,api_calls=0,supervised_fits=0))
    print(results,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('matrix','released','old-titles','old-table','out'):
        parser.add_argument('--'+k,type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args())
