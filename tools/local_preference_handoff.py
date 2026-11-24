"""Local competence of a third expert using training disagreements on similar text edits."""
import math

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


def edit_signatures(pairs):
    vectorizer=HashingVectorizer(analyzer="char_wb",ngram_range=(3,5),n_features=2**15,
                                alternate_sign=False,norm=None)
    left,right=zip(*pairs,strict=True)
    differences=abs(vectorizer.transform(left)-vectorizer.transform(right))
    return normalize(differences,norm="l2",copy=False)


def build_memory(data,pairs,groups,labels):
    ids=np.flatnonzero(data["conflict"])
    y=np.asarray(labels)
    if not len(ids):
        return {"size":0}
    return {"edits":edit_signatures([pairs[i] for i in ids]),"features":data["features"][ids,:3],
            "unanimous":data["unanimous"][ids],"success":(data["third"][ids]==y[ids]).astype(float),
            "groups":np.asarray(groups)[ids],"size":len(ids)}


def competence(memory,data,pairs,groups):
    predicted=np.full(len(pairs),.5)
    ids=np.flatnonzero(data["conflict"])
    if not len(ids) or not memory["size"]:
        return predicted
    edits=edit_signatures([pairs[i] for i in ids])
    similarities=(edits @ memory["edits"].T).toarray()
    ref=memory["features"]
    for j,i in enumerate(ids):
        # Also exclude the query group in the empirical prior, so learning feedback is group-out.
        candidates=np.flatnonzero((memory["groups"]!=groups[i]) & (memory["unanimous"]==data["unanimous"][i]))
        if len(candidates)<8:
            continue
        prior=(1+memory["success"][candidates].sum())/(2+len(candidates))
        distance=np.mean((data["features"][i,:3]-ref[candidates])**2,axis=1)
        similarity=.5*similarities[j,candidates]+.5*np.exp(-distance/.05)
        k=min(len(candidates),max(8,min(32,math.ceil(math.sqrt(len(candidates))))))
        chosen=np.argsort(-similarity,kind="stable")[:k]
        weights=similarity[chosen]
        predicted[i]=(np.dot(weights,memory["success"][candidates[chosen]])+8*prior)/(weights.sum()+8)
    return predicted


def route(data, estimated, threshold):
    if threshold is None:
        return data["base"].copy()
    return np.where(data["conflict"] & (estimated>threshold), data["third"], data["base"])


def learn_threshold(data,labels,rates):
    from tools.preference_handoff import changes
    reference=data["base"]
    best_key,choice=(0,0),None
    candidates=[]
    for threshold in (.5,.55,.60):
        score=changes(reference,route(data,rates,threshold),labels)
        candidates.append({"threshold":threshold,**score})
        key=(score["net"],-score["switches"])
        if key>best_key:
            best_key,choice=key,threshold
    return choice,candidates
