"""Resolve effective task goals without presenting defaults as declared intent."""
import numpy as np


def resolve_goal(task, metric):
    spec=task.spec or {};schema=spec.get('input_schema') or {}
    labels=np.asarray(task.y_train)
    if labels.ndim!=1 or not len(labels):raise ValueError('Task requires nonempty scalar training labels')
    if labels.dtype.kind not in 'biufUS':raise ValueError('Task labels require finite scalar values')
    if labels.dtype.kind in 'f' and not np.isfinite(labels).all():raise ValueError('Task labels must be finite')
    if labels.dtype.kind in 'US' and any(not s.strip() for s in labels.tolist()):raise ValueError('Task labels must not be empty')
    shape_status='unknown'
    if hasattr(task,'X_train'):
        x=task.X_train
        if len(x)!=len(labels):raise ValueError('Task feature and label row counts disagree')
        if task.kind=='single_text' and any(not isinstance(v,str) for v in x):
            raise ValueError('Single text task requires one string per row')
        if task.kind=='pair_text' and any(not isinstance(v,(tuple,list)) or len(v)!=2 or
                                        any(not isinstance(t,str) for t in v) for v in x):
            raise ValueError('Pair text task requires two strings per row')
        if task.kind in ('numeric_tabular','categorical_tabular','image'):
            a=np.asarray(x)
            if a.ndim not in ((3,4) if task.kind=='image' else (2,)):
                raise ValueError('Task input shape contradicts its kind')
            if task.kind in ('numeric_tabular','image') and a.dtype.kind not in 'biuf':
                raise ValueError('Numeric/image task requires numeric inputs')
        shape_status='confirmed'
    sources=spec.get('goal_sources',{})
    metric_source=sources.get('primary_metric','declared_spec' if 'metric' in spec else 'default_accuracy')
    metric_declared=metric_source in ('declared_manifest','declared_spec')
    positive=spec.get('positive_label')
    applies=len(metric.labels)==2 and metric.name in ('binary_f1','roc_auc')
    pair=schema.get('pair_semantics')
    return {'resolution_version':1,'fields':{
        'input_kind':{'value':task.kind,'source':spec.get('schema_origin','task_adapter'),'status':shape_status},
        'primary_metric':{'value':metric.name,'source':metric_source,
                          'status':'confirmed' if metric_declared else 'unknown',
                          'effective_policy':'maximize'},
        'label_domain':{'value':list(metric.labels),'source':'training_labels','status':'confirmed'},
        'positive_label':{'value':positive if positive is not None else (metric.labels[-1] if applies else None),
            'source':'declared_spec' if positive is not None else ('default_sorted_last' if applies else 'not_applicable'),
            'status':'confirmed' if positive is not None else ('unknown' if applies else 'not_applicable')},
        'pair_semantics':{'value':pair,'source':'input_schema' if pair is not None else 'undeclared',
                          'status':'confirmed' if pair is not None else 'unknown'}},
        'scope':'Effective defaults remain executable; unknown intent is not a verified task declaration'}
