"""One predeclared feature intervention on a corroborated, observable group, with frozen models."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.compare_joint_feedback import CASES, PLAN, raw_records
from scripts.replay_joint_error_groups import observations
from scripts.run_planner_self_review import distinct
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.conditional_review import group_observation
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.models import SoftVoteModel, predict_with_model, probability_with_model
from superagent.opening_evidence import row_signals


def adjust(images, threshold, target, lower, upper):
    result=images.copy()
    contrast=row_signals('image',images)['image_contrast']
    use=(contrast>0) & (contrast<=threshold)
    if np.any(use):
        values=images[use].astype(float)
        axes=tuple(range(1,values.ndim))
        means=values.mean(axis=axes,keepdims=True)
        scale=(target/contrast[use]).reshape((-1,)+(1,)*len(axes))
        changed=np.clip(means+(values-means)*scale,lower,upper)
        result[use]=np.rint(changed) if np.issubdtype(images.dtype,np.integer) else changed
    return result,use


def score_x(model, x, y, metric):
    prediction=predict_with_model('image',model,x)
    matrix,classes=None,None
    if metric.requires_scores:
        classes,matrix=probability_with_model('image',model,x)
    return prediction,metric.evaluate(y,prediction,matrix,classes)['score']


def measure(model,x,truth,metric,params,expected=None):
    before,before_score=score_x(model,x,truth,metric)
    if expected is not None:
        require(np.array_equal(before,expected), 'Stored incumbent and independent inference disagree')
    altered,mask=adjust(x,**params)
    after,after_score=score_x(model,altered,truth,metric)
    require(np.array_equal(altered[~mask],x[~mask]) and np.array_equal(after[~mask],before[~mask]),
            'Changed predictions or features outside the diagnosed group')
    return dict(total=len(x),eligible=int(sum(mask)),changed=int(sum(after!=before)),baseline=before_score,
                proposed=after_score,delta=after_score-before_score,gained=int(sum((before!=truth)&(after==truth))),
                lost=int(sum((before==truth)&(after!=truth))))


def fresh_panel(root,case,model,metric,params):
    sampling=read(PLAN)['sampling']
    for code,x,y in raw_records(root/'sources'):
        if code!=case:continue
        ids=distinct(x,y)
        chosen,remainder=train_test_split(ids,train_size=sampling['training']+sampling['evaluation'],
                                         stratify=y[ids],random_state=sampling['seed'])
        rng=np.random.default_rng(102711)
        held=rng.choice(remainder,300,replace=False)
        require(set(chosen).isdisjoint(held),'Fresh panel overlaps development rows')
        return measure(model,x[held],y[held],metric,params)
    raise ValueError('Unknown registered source')


def examine(root,case):
    folder=root/'development'/case
    directory=folder/'feedback-local/run/agent'
    opening=read(directory/'task_opening.json')
    if opening['check_status']!='supported':return dict(result='no_corroborated_group')
    brief=read(directory/'opening_brief.json')
    definition=brief['checks'].get(opening['check'],{})
    task=load_training_task(folder/'task.zip',read_declarations=True)
    if task.kind!='image' or (definition.get('kind'),definition.get('signal'),definition.get('side'))!=('slice','image_contrast','low'):
        return dict(result='inapplicable_observed_feature')
    metric=MetricSpec.for_task(task)
    def partition(fold):return read(directory/(fold+'_split.json'))
    fit=task.X_train[partition('primary')['fit']]
    params=dict(threshold=definition['cut'],target=float(np.median(row_signals('image',fit)['image_contrast'])),
                lower=float(np.min(fit)),upper=float(np.max(fit)))
    base=opening['incumbent_members']
    history=read(directory/'history.json')
    seen=set()
    results={}
    def check(fold):
        ids,bank=observations(directory,fold,task,metric,opening['tested_candidate'])
        require(seen.isdisjoint(map(int,ids)), 'Overlapping evaluation folds')
        seen.update(map(int,ids))
        models=[]
        for n in base:
            trial=next(r['trial'] for r in history if r['status']=='completed' and r['split_id']==fold and r['candidate']==n)
            models.append(joblib.load(directory/f'trial_{trial}.joblib')['model'])
        require(all(m is not None for m in models),'A frozen estimator is unavailable')
        model=models[0] if len(models)==1 else SoftVoteModel(task.kind,models,np.asarray(metric.labels))
        incumbent=group_observation(bank,base,metric)
        results[fold]=measure(model,task.X_train[ids],task.y_train[ids],metric,params,incumbent['predictions'])
        require(abs(results[fold]['baseline']-incumbent['score'])<1e-12,'Stored metric was not reproduced')
    for f in ('primary','confirmation_1'):check(f)
    a,b=results['primary'],results['confirmation_1']
    accepted=min(a['delta'],b['delta'])>0 and (a['delta']+b['delta'])/2>=.003 and b['gained']>b['lost']
    if accepted and all(any(r['status']=='completed' and r['candidate']==n and r['split_id']=='confirmation_2' for r in history) for n in base):
        check('confirmation_2')
        accepted=results['confirmation_2']['delta']>=0
    item=dict(result='retained' if not accepted else 'qualified_for_new_panel',feature=opening['check'],
              parameters_from_primary_fit=params,folds=results,fits=0,model_requests=0)
    if accepted:
        selected=read(directory/'final_selection.json')
        model_path=directory/selected['model_file']
        require(sha(model_path)==selected['model_sha256'],'Final incumbent changed')
        item['fresh_panel']=fresh_panel(root,case,joblib.load(model_path)['model'],metric,params)
    return item


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=lambda s:Path(s).resolve())
    root=p.parse_args().root
    save(root/'development/intervention.json',dict(protocol=sha(ROOT/'research/joint_feedback_1127/intervention_protocol.json'),
         results={c:examine(root,c) for c in CASES},physical_fits=0,api_requests=0))
