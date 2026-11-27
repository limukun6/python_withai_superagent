"""Learn a regularized log-odds correction on independent training feedback."""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from scripts.study_task1_candidates import METHODS, ROLES, bound, paired
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN=ROOT/"research/task1_feedback_extension_1127/feedback_fusion.json"


def features(probabilities, reference):
    odds = logit(np.clip(probabilities, 1e-4, 1-1e-4))
    offset = logit(np.clip(reference, 1e-4, 1-1e-4))
    delta = np.clip(odds-offset[:,None], -5,5)
    uncertainty = 1 - abs(2*reference-1)
    disagreement = probabilities.max(axis=1)-probabilities.min(axis=1)
    return np.column_stack((delta,delta*uncertainty[:,None],delta*disagreement[:,None])),offset


def fit_correction(x,offset,y):
    def objective(w):
        z=offset+x@w
        loss=np.logaddexp(0,z).sum()-y@z+8*(w@w)
        return loss,x.T@(expit(z)-y)+16*w
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B")
    require(fit.success,"Correction did not converge")
    return fit.x


def run(out):
    folder=out/"feedback_fusion"
    folder.mkdir(exist_ok=False)
    info=read(out/"inputs.json")
    elected=read(out/"election/results.json")
    seals={m:read(out/("trial_"+m)/"sealed.json") for m in METHODS}
    classes=np.array(seals[METHODS[0]]["classes"])
    require(len(classes)==2 and all(s["classes"]==classes.tolist() for s in seals.values()),"Class domain mismatch")
    receipts=[]
    for role in ROLES:
        bank={}
        for name in METHODS:
            path=out/("trial_"+name)/(role+".npy")
            require(sha(path)==seals[name]["predictions"][role],"Unbound probabilities")
            bank[name]=np.load(path,allow_pickle=False)[:,1]
        base=bank["anchor"]
        share=elected["share"]
        reference=(1-share)*base+share*np.mean([bank[m] for m in elected["selected"]],axis=0)
        x,offset=features(np.column_stack([bank[m] for m in METHODS]),reference)
        truth=(bound(out,role+"_answers.joblib",info)==classes[1]).astype(float)
        if role=="nomination":
            w=fit_correction(x,offset,truth)
            save(folder/"locked.json",dict(protocol=sha(PLAN),methods=list(METHODS),weights=w.tolist(),
                 initial_recipe=sha(out/"election/results.json"),training_rows=len(truth),classifier_fits=1))
        correct=(expit(offset+x@w)>.5)==truth
        record=dict(role=role,rows=len(truth),vs_equal=paired(correct,(reference>.5)==truth),
                    vs_original=paired(correct,(base>.5)==truth))
        receipts.append(record)
    net=sum(r["vs_equal"]["net"] for r in receipts[1:])
    pct=net/sum(r["rows"] for r in receipts[1:])
    result=dict(lock=sha(folder/"locked.json"),roles=receipts,confirmation_gain=pct,
                eligible=pct>=.003 and min(r["vs_equal"]["net"] for r in receipts[1:])>=0,
                api_requests=0,development_only=True)
    save(folder/"results.json",result)
    print(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    run(parser.parse_args().out)
