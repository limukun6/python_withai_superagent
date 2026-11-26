"""Build an anonymous, training-only task brief and executable diagnostics before the first fit."""
from collections import Counter
from datetime import datetime, timezone
import re

import numpy as np

from .data import TrainingTask
from .native_schema import description
from .splits import _declared_column, take
from .task_files import open_package


GOAL = re.compile(r'predict|classif|identify|recogniz|distinguish|determine|target|label|score|report|'
                  r'hold out|held.out|unseen|future|group|recording|sample|example|each row|each image|each comment|'
                  r'\b(?:differ\w*|vary|repeated|overlap|independent|correlated|shift|unavailable)\b|'
                  r'不同|变化|重复|关联|缺失|独立|重叠|来自|'
                  r'预测|分类|识别|目标|评分|标签|划分|分组|未来|每一|每个|样本',re.I)
OMIT = re.compile(r'https?://|@|[/\\]|\.(zip|csv|np[yz]|json|parquet)\b|\b(dataset|corpus|benchmark)\b|'
                  r'benchmark|leaderboard|baseline|api.?key|password|credential|ignore (previous|all)|'
                  r'(test|validation|held.out) (labels? (are|is)|score (was|is))|\d+(\.\d+)?\s*[%％]|'
                  r'数据集名|排行榜|密钥|忽略指令',re.I)
ENTITY = re.compile(r'(?:unseen|held.out|new) (?:values? of|[\w-]+ identified by|groups? (?:of|in))\b|'
                    r'\b(?:hold out (?:entire|whole) groups|group (?:the )?(?:internal )?(?:split|folds?) by)\b|按.{0,16}分组|此前未见的实体',re.I)
TIME = re.compile(r'\b(?:earlier to later|chronological(?:ly)? (?:by|using)|future values? of|ordered by time)\b|按.{0,12}时间顺序',re.I)
NEGATED = re.compile(r"\b(?:if|maybe|might|could|not|never|except|example only|instead|unless)\b|不是|不要|假设|例如",re.I)


def _mentions(text, name):
    return re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text,re.I)


def instructions(package, task):
    columns=[col['name'] for col in task.spec.get('input_schema',{}).get('columns',[]) if isinstance(col,dict) and isinstance(col.get('name'),str)]
    if task.spec.get('declaration_evidence',{}).get('fields',{}).get('columns',{}).get('status')=='proposed':
        columns=[]  # Evidence-only names cannot establish a grouping or chronological split.
    try:
        with open_package(package,'train',shared=True) as archive:
            prose=description(archive)
    except (ValueError,OSError,UnicodeError):
        prose=''
    lines=[]
    declarations=set()
    for part in re.split(r'[\r\n;。；!?]+|\.(?:\s+|$)',prose):
        line=part.strip(' #\t*-')
        if not (0<len(line)<=300 and GOAL.search(line)) or OMIT.search(line):
            continue
        present=[i for i,n in enumerate(columns) if _mentions(line,n)]
        if len(present)==1 and not NEGATED.search(line):
            for kind,pattern in [('entity',ENTITY),('temporal',TIME)]:
                if pattern.search(line):
                    declarations.add((kind,columns[present[0]]))
        for i in sorted(present,key=lambda i:-len(columns[i])):
            line=re.sub(r'(?<!\w)'+re.escape(columns[i])+r'(?!\w)',f'field_{i+1}',line,flags=re.I)
        if line not in lines and len(lines)<8:
            lines.append(line)
    existing=task.spec.get('evaluation_protocol')
    origin='structured' if existing else 'ordinary'
    if not existing and len(declarations)>1:
        raise ValueError('Conflicting documented split columns or protocols')
    if not existing and declarations:
        kind,column=next(iter(declarations))
        task=TrainingTask(task.kind,task.X_train,task.y_train,
                          dict(task.spec,evaluation_protocol={'kind':kind,'column':column}))
        origin='documented_column'
    return task,lines,origin


def row_signals(kind, x):
    """Only geometry is exported. Literal text and identifier values stay inside the worker."""
    if kind=='image':
        array=np.asarray(x)
        axes=tuple(range(1,array.ndim))
        return {'image_contrast':np.std(array,axis=axes),'mean_pixel_value':np.mean(array,axis=axes)}
    if kind in ('single_text','pair_text'):
        pairs=kind=='pair_text'
        lengths=np.array([[len(str(s).split()) for s in pair] if pairs else [len(str(pair).split())] for pair in x])
        result={'word_count':np.sum(lengths,axis=1)}
        if pairs:
            result['length_difference']=np.abs(lengths[:,0]-lengths[:,1])
        return result
    array=np.asarray(getattr(x,'values',x))
    if kind=='numeric_tabular':
        return {'missing_fields':np.sum(~np.isfinite(array),axis=1),
                'mean_absolute_value':np.mean(np.nan_to_num(np.abs(array),nan=0.),axis=1)}
    empty={'','nan','na','null','none','?'}
    return {'missing_fields':np.array([sum(str(v).strip().casefold() in empty for v in row) for row in array]),
            'row_text_length':np.array([sum(len(str(v)) for v in row) for row in array])}


def make_brief(task, metric, folds, lines, origin):
    fit=folds['primary'][0]
    y=task.y_train[fit]
    signals=row_signals(task.kind,take(task.X_train,fit))
    declared=task.spec.get('evaluation_protocol',{})
    names=[c.get('name') for c in task.spec.get('input_schema',{}).get('columns',[])]
    column=declared.get('column') or declared.get('entity_column') or declared.get('time_column')
    field=f'field_{names.index(column)+1}' if column in names and column else None
    counts=Counter(y.tolist())
    frequencies=[counts[c] for c in metric.labels]
    facts={f'doc_{i+1}':s for i,s in enumerate(lines)}
    shape=task.spec.get('feature_shape') or ([len(names)] if names else [task.kind])
    facts.update(contract=dict(objective='classification',modality=task.kind,metric=metric.name,
                               classes=len(metric.labels),split=declared.get('kind','grouped_stratified'),
                               field=field,split_source=origin),
                 balance=dict(fitting_rows=len(fit),class_counts=frequencies),
                 shape=shape)
    probes={}
    if field and declared.get('kind') in ('temporal','entity'):
        probes['split_integrity']=dict(kind=declared['kind'],source='contract',field=field)
    if min(frequencies)<max(frequencies):
        rare=int(np.argmin(frequencies))
        probes['minority_errors']=dict(kind='class',class_index=rare,source='balance')
    for name,values in signals.items():
        low,high=np.quantile(values,[.25,.75])
        facts[name]=dict(q25=round(float(low),4),q75=round(float(high),4))
        for end,cut in [('low',low),('high',high)]:
            mask=values<=cut if end=='low' else values>=cut
            if min(int(mask.sum()),int((~mask).sum()))>=5:
                probes[name+'_'+end]=dict(kind='slice',signal=name,side=end,cut=float(cut),source=name)
    indexes=np.linspace(0,len(fit)-1,min(3,len(fit)),dtype=int)
    examples=[dict(class_index=metric.labels.index(y[i]),**{key:round(float(values[i]),3) for key,values in signals.items()}) for i in indexes]
    return dict(facts=facts,anonymized_fitting_examples=examples,checks=probes,
                rule='slice/class checks test whether core error exceeds remaining-row error by at least 0.05; split checks verify separation')


def subset(task,metric,held,check):
    if check['kind']=='class':
        return task.y_train[held]==metric.labels[check['class_index']]
    signal=row_signals(task.kind,take(task.X_train,held))[check['signal']]
    return signal<=check['cut'] if check['side']=='low' else signal>=check['cut']


def ordered_time(values):
    try:
        numbers=np.asarray(values,dtype=float)
    except (ValueError,TypeError):
        times=[]
        for value in values:
            date=datetime.fromisoformat(str(value).strip().replace('Z','+00:00'))
            times.append((date if date.tzinfo is not None else date.replace(tzinfo=timezone.utc)).timestamp())
        numbers=np.asarray(times)
    if not np.isfinite(numbers).all():
        raise ValueError('Incomplete or nonfinite temporal key')
    return numbers


def split_audit(task,folds):
    protocol=task.spec['evaluation_protocol']
    column=protocol.get('column') or protocol.get('entity_column') or protocol.get('time_column')
    values=_declared_column(task,column)
    if protocol['kind']=='temporal':
        values=ordered_time(values)
    rows={}
    for fold,(fit,held,_) in folds.items():
        if protocol['kind']=='entity':
            overlap=len(set(values[fit].tolist()) & set(values[held].tolist()))
            rows[fold]=dict(overlapping_entities=overlap,passes=overlap==0)
        else:
            rows[fold]=dict(strict_earlier=bool(max(values[fit])<min(values[held])),
                            passes=bool(max(values[fit])<min(values[held])))
    return rows


def verified_windows(task,seed,ordinary_partitions):
    protocol=task.spec.get('evaluation_protocol',{})
    if protocol.get('kind')=='temporal':
        from .splits import TEMPORAL_WINDOWS, _require_both_sides
        values=ordered_time(_declared_column(task,protocol.get('column') or protocol.get('time_column')))
        order=np.argsort(values,kind='stable')
        folds={}
        for name,window in zip(('primary','confirmation_1','confirmation_2'),TEMPORAL_WINDOWS):
            low,middle,high=(round(len(order)*p) for p in window)
            fit,held=order[low:middle],order[middle:high]
            if not len(held):
                raise ValueError('Temporal confirmation has no rows')
            earlier=fit[values[fit]<min(values[held])]
            _require_both_sides(task.y_train,earlier,held,'strict temporal')
            folds[name]=(earlier,held,dict(protocol='temporal',window=list(window),train_rows=len(earlier),
                                          heldout_rows=len(held),tied_training_rows_removed=len(fit)-len(earlier)))
    else:
        folds=ordinary_partitions(task,seed)
    if protocol.get('kind') in ('entity','temporal') and not all(r['passes'] for r in split_audit(task,folds).values()):
        raise ValueError('Training windows violate the documented separation')
    return folds
