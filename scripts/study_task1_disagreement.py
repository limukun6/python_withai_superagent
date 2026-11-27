"""Test local competence on model disagreements with grouped out-of-fit feedback."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix,hstack
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

from scripts.study_task1_candidates import ROLES,bound,paired
from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.splits import sample_groups

BANK=("trial_anchor","trial_attribute_alignment","trial_pretrained_icl","icl_global_structure",
      "icl_conditional_context","tabm_raw","tabm_structure")
PLAN=ROOT/"research/task1_feedback_extension_1127/disagreement_gate.json"


def assemble(x,prob):
    ref=prob[:,1:3].mean(axis=1)
    base=ref>.5
    side=np.where(base,1.,-1.)
    margin=(logit(np.clip(prob,1e-4,1-1e-4))-logit(np.clip(ref,1e-4,1-1e-4))[:,None])*side[:,None]
    disagree=(prob>.5)!=base[:,None]
    offset=abs(logit(np.clip(ref,1e-4,1-1e-4)))
    numeric=np.column_stack((np.clip(margin,-5,5),disagree.mean(1)))
    categorical=np.char.add(np.where(base,"1|","0|")[:,None],x.astype(str))
    return numeric,categorical,offset,base,disagree.any(1)


class Gate:
    def features(self,numeric,categorical):
        return hstack((csr_matrix(numeric),self.encoder.transform(categorical)),format="csr")

    def fit(self,x,p,y):
        numeric,categorical,offset,base,mask=assemble(x,p)
        self.encoder=OneHotEncoder(handle_unknown="ignore",min_frequency=12)
        self.encoder.fit(categorical[mask])
        matrix=self.features(numeric[mask],categorical[mask])
        response=(base[mask]==y[mask]).astype(float)
        penalty=np.full(matrix.shape[1],16.)
        penalty[:numeric.shape[1]]=8.
        def objective(w):
            z=offset[mask]+matrix@w
            return (np.logaddexp(0,z).sum()-response@z+.5*np.sum(penalty*w**2),
                    matrix.T@(expit(z)-response)+penalty*w)
        answer=minimize(objective,np.zeros(matrix.shape[1]),method="L-BFGS-B",jac=True)
        require(answer.success,"Local competence gate did not converge")
        self.coef=answer.x
        return self

    def predict(self,x,p):
        numeric,categorical,offset,base,mask=assemble(x,p)
        output=base.copy()
        if np.any(mask):
            predicted=offset[mask]+self.features(numeric[mask],categorical[mask])@self.coef
            output[mask]=np.where(predicted<0,~base[mask],base[mask])
        return output


def run(out):
    folder=out/"disagreement"
    folder.mkdir(exist_ok=False)
    info=read(out/"inputs.json")
    seals={m:read(out/m/"sealed.json") for m in BANK}
    def load(role):
        scores=[]
        for m in BANK:
            file=out/m/(role+".npy")
            require(sha(file)==seals[m]["predictions"][role],"Changed predictor bank")
            scores.append(np.load(file,allow_pickle=False)[:,1])
        return (bound(out,role+"_features.joblib",info),np.column_stack(scores),
                bound(out,role+"_answers.joblib",info)==seals["trial_anchor"]["classes"][1])
    x,p,y=load("nomination")
    group=sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    folds=[]
    for ix,(fit,held) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=91729).split(x,y,group)):
        gate=Gate().fit(x[fit],p[fit],y[fit])
        selection=gate.predict(x[held],p[held])
        baseline=p[held,1:3].mean(1)>.5
        folds.append(dict(fold=ix,rows=len(held),**paired(selection==y[held],baseline==y[held])))
    nominate=(sum(r["net"] for r in folds)>=3 and sum(r["net"]>0 for r in folds)>=2)
    save(folder/"decision.json",dict(protocol=sha(PLAN),groups=len(set(group)),folds=folds,nominate=nominate))
    hold=[]
    if nominate:
        gate=Gate().fit(x,p,y)
        for f in ROLES[1:]:
            hx,hp,hy=load(f)
            base=hp[:,1:3].mean(1)>.5
            hold.append(dict(role=f,rows=len(hy),**paired(gate.predict(hx,hp)==hy,base==hy)))
    result=dict(decision=sha(folder/"decision.json"),net_crossfit=sum(r["net"] for r in folds),checks=hold,
                fits=4+int(nominate),eligible=bool(hold and min(r["net"] for r in hold)>=0 and
                sum(r["net"] for r in hold)/sum(r["rows"] for r in hold)>=.003),api_requests=0)
    save(folder/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    run(parser.parse_args().out)
