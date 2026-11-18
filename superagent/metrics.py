"""Task-owned classification scores. Accuracy remains an honest diagnostic."""
from dataclasses import dataclass
import math
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,roc_auc_score

SUPPORTED=('accuracy','macro_f1','weighted_f1','micro_f1','binary_f1','balanced_accuracy','roc_auc')


def validate_metric(name):
    if not isinstance(name,str) or name not in SUPPORTED:
        raise ValueError('Supported primary metrics are '+', '.join(SUPPORTED))
    return name


@dataclass(frozen=True)
class MetricSpec:
    name: str
    labels: tuple

    def __post_init__(self):
        validate_metric(self.name)
        if len(self.labels)<2 or len(set(self.labels))!=len(self.labels):
            raise ValueError('Metric requires at least two distinct training classes')
        for value in self.labels:
            if type(value) not in (str,int,float,bool) or isinstance(value,float) and not math.isfinite(value):
                raise ValueError('Class domain must contain finite scalar labels')
        if self.name=='binary_f1' and len(self.labels)!=2:
            raise ValueError('binary_f1 requires two classes; declare an averaging rule for multiclass')

    @classmethod
    def for_task(cls,task):
        labels=np.unique(task.y_train).tolist()
        positive=task.spec.get('positive_label')
        if positive is not None:
            if len(labels)!=2 or positive not in labels:raise ValueError('positive_label must identify one of two training classes')
            labels=[v for v in labels if v!=positive]+[positive]
        return cls(validate_metric(task.spec.get('metric','accuracy')),tuple(labels))

    @classmethod
    def from_payload(cls,payload):
        record=payload.get('metric_spec')
        if record is None:return cls('accuracy',tuple(np.unique(payload['y_train']).tolist()))
        if not isinstance(record,dict) or set(record)!={'name','labels','direction'} or record['direction']!='maximize':
            raise ValueError('Malformed metric specification')
        return cls(record['name'],tuple(record['labels']))

    def record(self):return {'name':self.name,'labels':list(self.labels),'direction':'maximize'}

    @property
    def requires_scores(self):return self.name=='roc_auc'

    def evaluate_model(self,kind,model,inputs,labels,predictions=None):
        from .models import predict_with_model,probability_with_model
        predicted=predict_with_model(kind,model,inputs) if predictions is None else predictions
        if self.requires_scores:
            classes,scores=probability_with_model(kind,model,inputs)
            return self.evaluate(labels,predicted,scores,classes)
        return self.evaluate(labels,predicted)

    def evaluate(self,labels,predictions,scores=None,score_labels=None):
        labels,predictions=np.asarray(labels),np.asarray(predictions)
        if labels.ndim!=1 or predictions.ndim!=1 or len(labels)!=len(predictions) or not len(labels):
            raise ValueError('Metrics require equally sized nonempty scalar vectors')
        domain=set(self.labels)
        if not set(labels.tolist())<=domain or not set(predictions.tolist())<=domain:
            raise ValueError('Evaluation contains a class outside the training domain')
        accuracy=float(accuracy_score(labels,predictions))
        macro=float(f1_score(labels,predictions,labels=list(self.labels),average='macro',zero_division=0))
        score=accuracy
        if self.name.endswith('_f1'):
            average=self.name.removesuffix('_f1')
            options={'pos_label':self.labels[-1]} if average=='binary' else {}
            score=float(f1_score(labels,predictions,labels=list(self.labels),average=average,zero_division=0,**options))
        elif self.name=='balanced_accuracy':score=float(balanced_accuracy_score(labels,predictions))
        elif self.name=='roc_auc':
            if scores is None or score_labels is None:raise ValueError('ROC-AUC requires model scores and their class-column order')
            matrix=np.asarray(scores,dtype=float);order=list(np.asarray(score_labels).tolist())
            if (matrix.shape!=(len(labels),len(self.labels)) or len(set(order))!=len(order)
                or set(order)!=domain or not np.isfinite(matrix).all()):
                raise ValueError('Invalid model probability columns for ROC-AUC')
            if np.any(matrix<0) or np.any(matrix>1) or not np.allclose(matrix.sum(axis=1),1.,atol=1e-6):
                raise ValueError('ROC-AUC interface requires normalized probabilities')
            if len(set(labels.tolist()))!=len(self.labels):raise ValueError('ROC-AUC is undefined when an evaluation class is absent')
            if len(self.labels)==2:
                score=float(roc_auc_score(labels==self.labels[-1],matrix[:,order.index(self.labels[-1])]))
            else:
                sorted_labels=sorted(self.labels)
                aligned=matrix[:,[order.index(v) for v in sorted_labels]]
                score=float(roc_auc_score(labels,aligned,labels=sorted_labels,multi_class='ovr',average='macro'))
        return {'accuracy':accuracy,'macro_f1':macro,'primary_metric':self.name,
            'score':score,'rows':len(labels)}


def score_of(row):
    """Legacy accuracy records are allowed; a missing F1 is never substituted."""
    name=validate_metric(row.get('primary_metric','accuracy'))
    value=row.get('search_score')
    if value is None and name=='accuracy':value=row.get('search_accuracy')
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        return None
    return float(value)


def completed(row):return row.get('status')=='completed' and score_of(row) is not None


def metric_of(history,default='accuracy'):
    names={validate_metric(row.get('primary_metric','accuracy')) for row in history}
    if len(names)>1:raise ValueError('Cannot compare experiments with different primary metrics')
    return next(iter(names),validate_metric(default))
