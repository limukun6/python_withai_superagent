"""Training-only fact projection prototype. Not imported by the live AB control loop."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from superagent.contracts import TaskSpec
from superagent.profile import profile


def training_facts(task):
    count=len(task.y_train)
    if count<1:raise ValueError('A training profile needs labelled training rows')
    # A TrainingTask deliberately has no validation; never inspect a DevelopmentTask's y_val here.
    training_only=SimpleNamespace(kind=task.kind,spec=task.spec,X_train=task.X_train,y_train=task.y_train,y_val=None)
    existing=profile(training_only)
    existing.pop('validation_label_counts',None)
    declared=TaskSpec.from_task(task)
    counts=existing['train_label_counts']
    if sum(counts.values())!=count:raise ValueError('Label-count total differs from training rows')
    if set(counts)!={str(label) for label in declared.metric.labels}:raise ValueError('Label domain differs from the task contract')
    existing.update(train_rows=count,primary_metric=declared.metric.name)
    facts=dict(profile_version='training-facts-prototype-v1',computed_from='training data only',
        validation_accessed=False,formal_test_accessed=False,statistics=existing,
        majority_class_fraction=max(counts.values())/count,minority_class_fraction=min(counts.values())/count)
    return facts


def append_common_profile(raw_context,checked_context,facts):
    if {key:value for key,value in checked_context.items() if key not in ('checks','memory')}!=raw_context:
        raise ValueError('Raw/checked common context already differs')
    raw,checked=deepcopy(raw_context),deepcopy(checked_context)
    raw['training_profile']=deepcopy(facts);checked['training_profile']=deepcopy(facts)
    if {key:value for key,value in checked.items() if key not in ('checks','memory')}!=raw:
        raise ValueError('Profile insertion changed the experimental common fields')
    return raw,checked


def facts_sha256(facts):
    return hashlib.sha256(json.dumps(facts,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()
