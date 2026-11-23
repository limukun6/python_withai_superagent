"""Bounded, training-only facts for the formal A/B decision boundary."""
from copy import deepcopy
from types import SimpleNamespace

from .contracts import TaskSpec
from .profile import profile


def training_facts(task):
    if len(task.y_train)<1:
        raise ValueError('A training profile needs labelled training rows')
    clean=SimpleNamespace(kind=task.kind,spec=task.spec,X_train=task.X_train,y_train=task.y_train,y_val=None)
    statistics=profile(clean)
    statistics.pop('validation_label_counts',None)
    statistics.pop('test_accessed',None)
    declared=TaskSpec.from_task(task)
    counts=statistics['train_label_counts']
    if sum(counts.values())!=len(task.y_train) or set(counts)!={str(label) for label in declared.metric.labels}:
        raise ValueError('Training profile label counts differ from the task contract')
    statistics.update(train_rows=len(task.y_train),primary_metric=declared.metric.name)
    truncated={}
    for key,value in list(statistics.items()):
        if isinstance(value,list) and len(value)>32:
            truncated[key]=dict(total_items=len(value),visible_items=32,rule='first 32 in schema order')
            statistics[key]=value[:32]
    return dict(profile_version='training-facts-v1',computed_from='all supplied training rows; no external validation or test',
        validation_accessed=False,formal_test_accessed=False,statistics=deepcopy(statistics),truncated_fields=truncated,
        majority_class_fraction=max(counts.values())/len(task.y_train),minority_class_fraction=min(counts.values())/len(task.y_train))
