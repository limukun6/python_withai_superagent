"""Evaluate one three-finalist majority vote on the same three confirmation splits."""
from dataclasses import dataclass
from pathlib import Path
import math

import joblib
import numpy as np


def majority(predictions):
    anchor,second,third=(np.asarray(v) for v in predictions)
    if anchor.ndim!=1 or second.shape!=anchor.shape or third.shape!=anchor.shape:
        raise ValueError('Expected three aligned vectors')
    return np.where(second==third,second,anchor)


@dataclass
class ConfirmedVote:
    kind: str
    models: list
    classes_: np.ndarray

    def predict(self,inputs):
        from .models import predict_with_model
        return majority([predict_with_model(self.kind,m,inputs) for m in self.models])

    def predict_proba(self,inputs):
        from .models import predict_with_model
        predictions=[predict_with_model(self.kind,m,inputs) for m in self.models]
        shares=sum(weight*(p[:,None]==self.classes_) for weight,p in zip((1.000001,1.,1.),predictions,strict=True))
        return shares/3.000001


def decision(folder,history,confirmation):
    folder=Path(folder)
    eligible=(confirmation.get('valid_confirmation') and confirmation['primary_metric']=='accuracy'
              and len(confirmation['shortlist'])==3 and len(confirmation['seeds'])==3)
    if not eligible:
        return dict(adopt=False,reason='Three completed confirmation splits and three accuracy finalists are required.')
    leader=confirmation['selected_candidate']
    names=[leader,*[name for name in confirmation['shortlist'] if name!=leader]]
    trials={r['candidate']:r['trial'] for r in history}
    folds=[]
    for index in range(3):
        inputs=(folder/'inner_data.joblib' if index==0 else folder/f'confirmation_data_{index}.joblib')
        labels=joblib.load(inputs)['y_search']
        predictions=[]
        for name in names:
            num=trials[name]
            path=folder/(f'trial_{num}.joblib' if index==0 else f'confirmation_{index}_{num}.joblib')
            row=joblib.load(path)
            predictions.append(row['search_predictions'])
            if abs(np.mean(row['search_predictions']==labels)-row['metrics']['search_accuracy'])>1e-12:
                raise ValueError('Mismatched confirmation rows')
        vote=majority(predictions)
        anchor=predictions[0]
        gain=int(np.sum((anchor!=labels)&(vote==labels)))
        loss=int(np.sum((anchor==labels)&(vote!=labels)))
        folds.append(dict(rows=len(labels),corrected=gain,lost=loss,delta=(gain-loss)/len(labels)))
    gains=sum(r['corrected'] for r in folds)
    losses=sum(r['lost'] for r in folds)
    adopt=bool(np.mean([r['delta'] for r in folds])>=.002 and sum(r['delta']>0 for r in folds)>=2
               and gains-losses>=math.sqrt(gains+losses))
    return dict(adopt=adopt,components=names,splits=folds,corrected=gains,lost=losses,
                reason='Require a 0.2-point mean gain, gains in at least two splits and a positive paired margin of one square root of the discordant count.')
