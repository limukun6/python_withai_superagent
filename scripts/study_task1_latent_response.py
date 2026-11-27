"""Fit hidden respondent classes with context-specific offsets to strictly cross-fitted tree logits."""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import blocks, tokens

PLAN=ROOT/"research/task1_feedback_extension_1127/latent_response_classes.json"
METHODS=("two","four")


def context_features(x,columns):
    return np.column_stack((x[:,columns], tokens(x[:,columns])))


class LatentResponse:
    def __init__(self,k,centered=False):
        self.k=k
        self.centered=centered

    def coefficients(self, beta):
        if self.centered:
            return beta-(beta@self.prior)[:,None]
        return beta

    def fit(self,rows,labels,base):
        self.profile,self.context=blocks(rows)
        ids=np.asarray(tokens(rows[:,self.profile]))
        groups,g=np.unique(ids,return_inverse=True)
        self.group_key={v:i for i,v in enumerate(groups)}
        self.encoder=OneHotEncoder(handle_unknown="ignore",min_frequency=12)
        design=self.encoder.fit_transform(context_features(rows,self.context)).tocsr()
        count=np.bincount(g,minlength=len(groups))
        packed=sparse.coo_matrix((labels-base,(g,np.arange(len(g)))),shape=(len(groups),len(g))).tocsr()
        residuals=(packed@design).toarray()/np.sqrt(count[:,None]+4.)
        cluster=KMeans(n_clusters=self.k,n_init=4,random_state=9281).fit_predict(residuals)
        q=np.eye(self.k)[cluster]*.8+.2/self.k
        beta=np.zeros((design.shape[1],self.k))
        offset=logit(np.clip(base,1e-6,1-1e-6))
        y=labels[:,None]
        for _ in range(14):
            sample=q[g]
            self.prior=(q.sum(0)+4)/(len(groups)+4*self.k)
            def objective(flat):
                coefficients=flat.reshape(beta.shape)
                logits=offset[:,None]+design@self.coefficients(coefficients)
                loss=(sample*(np.logaddexp(0.,logits)-y*logits)).sum()+12.*(coefficients**2).sum()
                gradient=design.T@(sample*(expit(logits)-y))
                if self.centered:
                    gradient-=gradient.sum(axis=1)[:,None]*self.prior[None,:]
                gradient+=24.*coefficients
                return loss,gradient.ravel()
            solution=minimize(objective,beta.ravel(),jac=True,method="L-BFGS-B",
                              options=dict(maxiter=120,ftol=1e-9))
            require(solution.success,"Latent-class joint correction did not converge")
            beta=solution.x.reshape(beta.shape)
            z=offset[:,None]+design@self.coefficients(beta)
            ll=y*z-np.logaddexp(0.,z)
            by_group=np.zeros((len(groups),self.k))
            np.add.at(by_group,g,ll)
            evidence=np.log(self.prior)[None,:]+.75*by_group
            q=np.exp(evidence-logsumexp(evidence,axis=1,keepdims=True))
        self.beta,self.posterior=self.coefficients(beta),q
        if self.centered:
            require(np.max(np.abs(self.beta@self.prior))<1e-8,"Uncentered latent offsets")
        return self

    def predict(self,rows,base):
        codes=np.asarray(tokens(rows[:,self.profile]))
        q=np.stack([self.posterior[self.group_key[c]] if c in self.group_key else self.prior for c in codes])
        x=self.encoder.transform(context_features(rows,self.context)).tocsr()
        offset=logit(np.clip(base,1e-6,1-1e-6))
        values=(q*expit(offset[:,None]+x@self.beta)).sum(1)
        return np.column_stack((1-values,values))


def run(out):
    info=read(out/"inputs.json")
    fit=bound(out,"fitting.joblib",info)
    x,raw=fit["x"],fit["y"]
    classes,y=np.unique(raw,return_inverse=True)
    teachers=out/"distill_teachers"
    d=read(teachers/"sealed.json")
    file=teachers/"out_of_fold.npy"
    require(d["inputs"]==sha(out/"inputs.json") and sha(file)==d["predictions"] and d["overlap"]==0,"Unbound teachers")
    base=np.load(file,allow_pickle=False)[:,0]
    anchor=out/"trial_anchor"
    rec=read(anchor/"sealed.json")
    require(rec["classes"]==classes.tolist(),"Reference classes differ")
    for name,k in zip(METHODS,(2,4)):
        folder=out/("latent_"+name)
        folder.mkdir(exist_ok=False)
        start=time.monotonic()
        model=LatentResponse(k).fit(x,y,base)
        hashes={}
        for role in ROLES:
            rows=bound(out,role+"_features.joblib",info)
            f=anchor/(role+".npy")
            require(rec["predictions"][role]==sha(f),"Changed tree scores")
            p=model.predict(rows,np.load(f,allow_pickle=False)[:,1])
            require(np.isfinite(p).all() and np.allclose(p.sum(1),1.),"Invalid latent-class prediction")
            f=folder/(role+".npy")
            np.save(f,p,allow_pickle=False)
            hashes[role]=sha(f)
        save(folder/"sealed.json",dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),classes=classes.tolist(),
          predictions=hashes,mixture_fits=1,kmeans_initializations=1,kmeans_restarts=4,joint_m_steps=14,
          groups=len(model.group_key),latent_classes=k,mixture_prior=model.prior.tolist(),
          seconds=round(time.monotonic()-start,2),api_requests=0))


def grade(out):
    elect(out,METHODS,"latent",PLAN)
    info=read(out/"inputs.json")
    result={}
    for role in ROLES:
        y=bound(out,role+"_answers.joblib",info)
        hits={}
        for name in ("trial_anchor",*("latent_"+s for s in METHODS)):
            root=out/name
            rec=read(root/"sealed.json")
            file=root/(role+".npy")
            require(sha(file)==rec["predictions"][role],"Changed frozen scores")
            hits[name]=np.asarray(rec["classes"])[np.load(file,allow_pickle=False).argmax(1)]==y
        result[role]={s:paired(hits["latent_"+s],hits["trial_anchor"]) for s in METHODS}
    save(out/"latent_election/vs_original.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    with threadpool_limits(limits=2):
        (run if a.action=="run" else grade)(a.out)
