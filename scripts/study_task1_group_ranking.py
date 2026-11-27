"""Test conditional group ranking with out-of-group row calibration for a binary task."""
from collections import defaultdict
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.study_task1_candidates import ROLES,bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.joint_context import blocks,tokens

PLAN=ROOT/"research/task1_feedback_extension_1127/within_context_rank.json"


def keyed(x,columns):
    return np.asarray(tokens(x[:,columns]))


class WithinGroupRank:
    def __init__(self,profile,context):
        self.profile=profile
        self.context=context

    def frame(self,x):
        return np.column_stack((x,keyed(x,self.context)))

    def fit(self,x,y):
        from catboost import CatBoostRanker,Pool
        profile=keyed(x,self.profile)
        context=keyed(x,self.context)
        self.global_mean=(float(np.sum(y))+1)/(len(y)+2)
        self.profiles=defaultdict(lambda:np.zeros(3))
        self.contexts=defaultdict(lambda:np.zeros(2))
        for p,c,outcome in zip(profile,context,y,strict=True):
            self.profiles[p][0]+=1
            self.profiles[p][1]+=outcome
            self.contexts[c]+=[1,outcome]
        usable=np.array([0<self.profiles[p][1]<self.profiles[p][0] for p in profile])
        require(usable.sum()>=100,"Too few matched groups with conflicting choices")
        indices=np.flatnonzero(usable)
        order=indices[np.argsort(profile[indices],kind="stable")]
        values=self.frame(x)
        ids=np.unique(profile,return_inverse=True)[1][order]
        pool=Pool(values[order],y[order],group_id=ids,cat_features=list(range(values.shape[1])))
        self.model=CatBoostRanker(loss_function="PairLogit",iterations=450,depth=6,learning_rate=.05,l2_leaf_reg=3.,
                                  random_seed=4291,verbose=False,allow_writing_files=False,thread_count=3)
        self.model.fit(pool)
        scores=np.asarray(self.model.predict(values),dtype=float)
        self.center=float(scores.mean())
        for p,value in zip(profile,scores,strict=True):
            self.profiles[p][2]+=value
        self.fit_rows=int(usable.sum())
        return self

    def project(self,x):
        rank=np.asarray(self.model.predict(self.frame(x)),dtype=float)
        matrix=[]
        for r,p,c in zip(rank,keyed(x,self.profile),keyed(x,self.context),strict=True):
            pn,positive,total=self.profiles.get(p,(0.,0.,0.))
            cn,other=self.contexts.get(c,(0.,0.))
            pm=(positive+8*self.global_mean)/(pn+8)
            cm=(other+8*self.global_mean)/(cn+8)
            centered=r-(total+4*self.center)/(pn+4)
            matrix.append([centered,r,logit(pm),logit(cm),np.log1p(pn),np.log1p(cn)])
        return np.asarray(matrix)


def train(out):
    folder=out/"rank_pairwise"
    folder.mkdir(exist_ok=False)
    source=read(out/"inputs.json")
    data=bound(out,"fitting.joblib",source)
    x,labels=data["x"],data["y"]
    classes,y=np.unique(labels,return_inverse=True)
    require(len(classes)==2,"Binary ranking calibration requires two classes")
    profile,context=blocks(x)
    calibration=np.empty((len(y),6))
    counts=[]
    start=time.monotonic()
    for i,(fit,held) in enumerate(GroupKFold(3).split(x,y,data["groups"])):
        save(folder/("ranking_fit_"+str(i)+".json"),dict(protocol=sha(PLAN),rows=len(fit)))
        model=WithinGroupRank(profile,context).fit(x[fit],y[fit])
        calibration[held]=model.project(x[held])
        counts.append(model.fit_rows)
    save(folder/"calibration_fit.json",dict(rows=len(y),out_of_fit=True))
    link=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000)).fit(calibration,y)
    save(folder/"ranking_fit_full.json",dict(rows=len(y)))
    model=WithinGroupRank(profile,context).fit(x,y)
    bank={}
    for role in ROLES:
        q=bound(out,role+"_features.joblib",source)
        p=link.predict_proba(model.project(q))
        require(p.shape==(len(q),2) and np.isfinite(p).all() and np.allclose(p.sum(1),1),"Invalid calibrated probabilities")
        path=folder/(role+".npy")
        np.save(path,p,allow_pickle=False)
        bank[role]=sha(path)
    result=dict(protocol=sha(PLAN),classes=classes.tolist(),predictions=bank,rank_fits=4,calibrator_fits=1,
                rank_training_rows=counts+[model.fit_rows],profile_fields=len(profile),context_fields=len(context),
                seconds=round(time.monotonic()-start,2),api_requests=0)
    save(folder/"sealed.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("train","elect"))
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    if a.action=="train":
        train(a.out)
    else:
        elect(a.out,("pairwise",),"rank",PLAN)
