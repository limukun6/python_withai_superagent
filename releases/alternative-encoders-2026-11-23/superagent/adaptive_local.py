"""Optional training-only entry point. The scheduler never receives outer data."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from .benchmark import train_candidate
from .splits import grouped_partition,overlap_audit,take
from .metrics import MetricSpec
from .evidence_schedule import catalog_for,schedule

def save(path,value):path.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def train(task,directory,seed=707,time_limit=300,fit_limit=90):
    directory=Path(directory)
    metric=MetricSpec.for_task(task)
    if metric.name!='accuracy':raise ValueError('Adaptive prototype currently supports accuracy only')
    directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+time_limit
    def event(row):
        with (directory/'events.jsonl').open('a') as file:file.write(json.dumps(row,allow_nan=False)+'\n')
    payloads=[];truth=[]
    for index in range(3):
        fit,held,_=grouped_partition(task,seed+10000*index,'shared_text')
        audit=overlap_audit(task,fit,held)
        if audit.get('search_duplicate_rows',0) or audit.get('search_rows_sharing_text',0):
            raise ValueError('Groups overlap')
        save(directory/f'split_{index}.json',dict(train_indices=fit.tolist(),search_indices=held.tolist(),**audit))
        payload=dict(kind=task.kind,X_train=take(task.X_train,fit),y_train=task.y_train[fit],
            X_search=take(task.X_train,held),y_search=task.y_train[held],metric_spec=metric.record(),
            retain_model=False,feature_cache_dir=str(directory/'feature_cache'))
        path=directory/f'payload_{index}.joblib';joblib.dump(payload,path)
        payloads.append(path);truth.append(payload['y_search'])
    def execute(payload,candidate,artifact):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Run time limit exhausted')
        model,metrics=train_candidate(payload,candidate,artifact,min(fit_limit,remaining))
        if metrics['status']!='completed' or metrics['classifier_fits']!=1:
            event(dict(event='failure',candidate=candidate,metrics=metrics))
            raise RuntimeError('Worker failed; no silent retry')
        return model
    def measure(candidate,index):
        artifact=directory/f'measured_{index}_{candidate}.joblib'
        execute(payloads[index],candidate,artifact)
        output=joblib.load(artifact);mask=np.asarray(output['search_predictions'])==truth[index]
        if not np.isclose(mask.mean(),output['metrics']['search_accuracy']):raise ValueError('Score mismatch')
        return mask
    selected=schedule(catalog_for(task.kind),measure,event,seed)
    event(dict(event='selection_frozen',candidate=selected))
    full=directory/'refit_data.joblib'
    joblib.dump(dict(kind=task.kind,X_train=task.X_train,y_train=task.y_train,X_search=task.X_train[:1],
        mode='refit',metric_spec=metric.record(),feature_cache_dir=str(directory/'feature_cache')),full)
    model=execute(full,selected,directory/'refit_component_0.joblib')
    path=directory/'selected_model.joblib'
    joblib.dump(dict(kind=task.kind,candidate=selected,model=model),path)
    spec=directory/'task_spec.json'
    save(spec,{**task.spec,'metric':metric.name,'metric_spec':metric.record(),'model_kind':task.kind,
               'task_kind':task.kind+'_classification','class_count':len(metric.labels)})
    save(directory/'final_selection.json',dict(family=selected,model_file=path.name,model_format='candidate_bundle_v1',
        model_sha256=sha(path),task_spec_sha256=sha(spec),test_accessed=False,refit=True,training_rows=len(task.y_train)))
    summary=dict(selected=selected,seconds=time.monotonic()-started,fits=1+len(list(directory.glob('measured_*.joblib'))),
                 model_sha256=sha(path),api_calls=0)
    save(directory/'summary.json',summary)
    return summary

def command(args):
    from .data import load_training_task
    result=train(load_training_task(args.task_package),Path(args.run_dir).resolve(),args.seed)
    print(json.dumps(result));return 0
