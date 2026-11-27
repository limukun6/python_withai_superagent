"""A supervised preference vote from independent training groups in similar semantic contexts."""
from functools import partial
import hashlib
import json

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from .ab_sentence import (ALTERNATIVES, AlternativeSentenceFeatures, DeepPreferenceFeatures,
                          SentenceFeatures, encode, encode_alternative, verified_directory, verified_files)
from .data import TrainingTask
from .local_sentence import LocalSentenceFeatures, installed, vectors
from .models import Candidate, SoftVoteModel, probability_with_model
from .paired_encoder import AVAILABLE_ENCODERS, PairedEncoderFeatures, vectors_for
from .splits import sample_groups


def unit(values):
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True),1e-12)


def views(rows, provider):
    # Embed each individual text without labels, or features from other rows.
    unique = list(dict.fromkeys(x for pair in rows for x in pair))
    lookup = {text:i for i,text in enumerate(unique)}
    encoded = provider(unique)
    left, right = (encoded[[lookup[pair[t]] for pair in rows]] for t in (0,1))
    return unit(left-right), unit(left+right)


class AnalogyIndex:
    def fit(self, rows, labels, groups, provider):
        self.change, self.context = views(rows, provider)
        self.signed = 2*np.asarray(labels,dtype=float)-1
        if set(self.signed)!={-1.,1.}:
            raise ValueError('Expected a two-sided preference target')
        self.groups = np.asarray(groups)
        self.k = int(np.clip(round(np.sqrt(len(set(groups)))),3,40))
        # Resolve exact similarities by input content, consistently for swapped sides.
        self.order = np.asarray(sorted(range(len(rows)),key=lambda i:(
            tuple(sorted(rows[i])), int(self.signed[i]*np.sign((rows[i][0]>rows[i][1])-(rows[i][0]<rows[i][1]))))))
        return self

    def predict_probability(self, rows, provider):
        diff, middle = views(rows, provider)
        output = np.full(len(rows), .5, dtype=float)
        for start in range(0, len(rows), 96):
            directional = diff[start:start+96] @ self.change.T
            context = np.maximum(0.,middle[start:start+96] @ self.context.T)
            similarity = np.abs(directional)*context
            for offset, scores in enumerate(similarity):
                # Try the closest rows first and expand only when many share the same training group.
                stable = scores[self.order]
                count = min(len(stable),max(4*self.k,self.k))
                while True:
                    minimum = np.partition(stable,len(stable)-count)[len(stable)-count]
                    eligible = np.flatnonzero(stable>=minimum)
                    closest = self.order[eligible[np.argsort(-stable[eligible],kind='stable')]]
                    selected, used = [],set()
                    for j in closest:
                        if scores[j]<=0:
                            break
                        if self.groups[j] not in used:
                            used.add(self.groups[j])
                            selected.append(j)
                            if len(selected)==self.k:
                                break
                    if len(selected)==self.k or count==len(stable) or minimum<=0:
                        break
                    count = min(2*count,len(stable))
                if selected:
                    weights = np.square(scores[selected]).astype(float)
                    votes = self.signed[selected]*np.sign(directional[offset,selected])
                    margin = np.dot(weights,votes)/(weights.sum()+2*weights.mean())
                    output[start+offset] = (1.+margin)/2.
        return output



def embedding_function(feature):
    if isinstance(feature, LocalSentenceFeatures):
        return partial(vectors,feature.spec)
    if isinstance(feature,PairedEncoderFeatures):
        return partial(vectors_for,feature.variant)
    if isinstance(feature,AlternativeSentenceFeatures):
        return lambda rows:encode_alternative(rows,feature.variant)[0]
    if isinstance(feature,(DeepPreferenceFeatures,SentenceFeatures)):
        return lambda rows:encode(rows,deep=isinstance(feature,DeepPreferenceFeatures))[0]
    raise ValueError('No standalone frozen representation was declared by this model')


class AnalogyClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,features):
        self.features=features

    def fit(self,rows,labels):
        self.classes_=np.array([0,1])
        groups=sample_groups(TrainingTask('pair_text',rows,np.asarray(labels)),'shared_text')
        self.index_=AnalogyIndex().fit(rows,labels,groups,embedding_function(self.features))
        return self

    def predict_proba(self,rows):
        positive=self.index_.predict_probability(rows,embedding_function(self.features))
        return np.column_stack((1-positive,positive))

    def predict(self,rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]


def fit_analogy(rows,labels,query,*,feature):
    model=Pipeline([('classifier',AnalogyClassifier(feature))])
    model.fit(rows,labels)
    return model,model.predict(query)


def candidate_for(feature):
    config=dict(encoder=feature.encoder_model,revision=feature.encoder_revision,classifier='preference_analogy',
                representation='direction and midpoint of individual frozen embeddings',
                independent_training_groups=True,orientation='complement',neighbors='clipped_sqrt_groups_3_40')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    return Candidate('preference_analogy_'+digest,config,partial(fit_analogy,feature=feature))


def supported():
    features=[PairedEncoderFeatures(v,'linear') if v in AVAILABLE_ENCODERS else AlternativeSentenceFeatures('pair_text',v)
              for v,spec in ALTERNATIVES.items() if verified_files(spec['env'],spec['files']) is not None]
    if verified_directory() is not None:
        features.append(SentenceFeatures('pair_text'))
    if verified_directory(deep=True) is not None:
        features.append(DeepPreferenceFeatures('pair_text'))
    features.extend(LocalSentenceFeatures('pair_text',spec) for spec in installed())
    return [candidate_for(feature) for feature in features]


def derived_candidate(incumbent):
    feature=getattr(incumbent,'named_steps',{}).get('features')
    if not isinstance(feature,SentenceFeatures):
        return None
    expected=candidate_for(feature)
    return next((c for c in supported() if c.family==expected.family),None)


class WeightedProbabilityVote(SoftVoteModel):
    def __init__(self,kind,models,classes,weights):
        super().__init__(kind,models,np.asarray(classes))
        self.weights=np.asarray(weights,dtype=float)
        if (len(models)!=len(self.weights) or not models or not np.isfinite(self.weights).all()
                or (self.weights<0).any() or not np.isclose(self.weights.sum(),1.)):
            raise ValueError('Invalid frozen probability weights')

    def predict_proba(self,inputs):
        values=[]
        for model in self.models:
            classes,scores=probability_with_model(self.kind,model,inputs)
            if set(classes)!=set(self.classes_):
                raise ValueError('Fitted probability classes disagree')
            values.append(scores[:,[list(classes).index(k) for k in self.classes_]])
        return np.average(values,axis=0,weights=self.weights)
