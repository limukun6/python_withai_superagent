"""Cross-fit small corrections using the dispersion of independently fitted members."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.preprocessing import StandardScaler

from scripts.study_task1_conditional_opportunity import ROLES,STUDY,load_matrix,metrics,ordered,panels,sources
from scripts.study_task1_cross_context import fit_offset
from scripts.study_task1_outer import compare,prediction_csv
from scripts.study_task1_seed_completion import origin
from scripts.unified_released_study import ROOT,read,require,save,sha

PLAN=ROOT/"research/task1_feedback_extension_1127/seed_disagreement_reading.json"
STAGE="disagreement_reading_1128"
HEADS=("calibration","disagreement")


def features(parts):
    member=np.stack([np.mean(p,axis=0) for p in parts])
    prior=member.mean(0)
    odds=logit(np.clip(prior,.001,.999))
    within=np.sqrt(np.mean([np.var(p,axis=0) for p in parts],axis=0))
    between=member.std(0)
    median=np.mean([np.median(p,axis=0) for p in parts],axis=0)-prior
    support=np.mean([np.mean(np.asarray(p)>.5,axis=0) for p in parts],axis=0)-expit(odds/.25)
    return prior,np.column_stack((odds,within,between,median,support,odds*within,odds*between))


class Offset:
    def __init__(self,name,bank,target):
        self.columns=1 if name=="calibration" else 7
        p,mat=bank
        self.scale=StandardScaler().fit(mat[:,:self.columns])
        # Flat coordinates on the fitting panel carry no transferable evidence.
        self.live=self.scale.var_>1e-12
        self.coef,_=fit_offset(self.design(mat),np.asarray(target,dtype=int),p,positive=False)

    def design(self,matrix):
        z=self.scale.transform(matrix[:,:self.columns])[:,self.live]
        return np.column_stack((np.ones(len(matrix)),np.clip(z,-5,5)))

    def predict(self,bank):
        p,m=bank
        return expit(logit(np.clip(p,.001,.999))+np.clip(self.design(m)@self.coef,-.6,.6))


def build(out,i):
    fold,src,x,y,q,rc=sources(out,i)
    cache=out/STUDY/str(i)/"control"
    original=read(cache/"sealed.json")
    root=out/STAGE/str(i)
    root.mkdir(parents=True,exist_ok=False)
    classes=np.unique(y)
    seed=read(src/"seed_replication.json")
    bank={}
    for role in (*ROLES,"external"):
        parts=[]
        for j,m in enumerate(rc["members"]):
            seeds=[None]+(seed["seeds"] if seed["action"]=="adopt" and m in seed["eligible"] else [])
            values=[]
            for n,s in enumerate(seeds):
                if role=="external":
                    path=src/(f"refit_component_{j}.joblib" if s is None else f"seed_refit_{j}_{s}.joblib")
                    values.append(ordered(joblib.load(path)["model"],q,classes))
                else:
                    values.append(load_matrix(cache,f"{role}_{j}_{n}"))
            parts.append(values)
        bank[role]=features(parts)
        np.testing.assert_allclose(bank[role][0],load_matrix(cache,role),rtol=2e-6,atol=3e-7)
    partition=panels(src,x,y)
    truth={r:y[held]==classes[1] for r,(_,held) in partition.items()}
    checks={}
    eligible=[]
    for head in HEADS:
        runs=[]
        for a,b in (ROLES,ROLES[::-1]):
            model=Offset(head,bank[a],truth[a])
            runs.append(metrics(model.predict(bank[b]),truth[b],bank[b][0]))
        n=sum(v["rows"] for v in runs)
        net=sum(v["net"] for v in runs)
        brier=sum(v["brier_change"]*v["rows"] for v in runs)/n
        checks[head]=dict(folds=runs,net=net,brier_change=brier)
        if net/n>=.002 and min(v["net"] for v in runs)>=0 and brier<=0:
            eligible.append(head)
    chosen=min(eligible,key=lambda k:(-checks[k]["net"],checks[k]["brier_change"],HEADS.index(k))) if eligible else "retain"
    save(root/"decision.json",dict(protocol=sha(PLAN),source=sha(cache/"sealed.json"),cross_checks=checks,chosen=chosen))
    merged=tuple(np.concatenate([bank[r][j] for r in ROLES]) for j in (0,1))
    joined_y=np.concatenate([truth[r] for r in ROLES])
    predictions={"retain":bank["external"][0]}
    evidence=root/"external_features.npz"
    np.savez(evidence,prior=bank["external"][0],features=bank["external"][1])
    models={}
    for head in HEADS:
        model=Offset(head,merged,joined_y)
        path=root/(head+".joblib")
        joblib.dump(model,path)
        models[head]=sha(path)
        predictions[head]=model.predict(bank["external"])
    predictions["chosen"]=predictions[chosen]
    prior=classes[(predictions["retain"]>.5).astype(int)].astype(str)
    require(np.array_equal(prior,prediction_csv(fold/"replicated/solve.csv",len(q))),"Prior does not reproduce")
    seals={}
    for n,p in predictions.items():
        path=root/(n+".npy")
        np.save(path,classes[(p>.5).astype(int)].astype(str),allow_pickle=False)
        seals[n]=sha(path)
    op=rc["fits"]+original["additional_control_fits"]+4+int(chosen!="retain")
    require(op<=64,"Calibration exceeds declared fit ceiling")
    save(root/"sealed.json",dict(decision=sha(root/"decision.json"),predictions=seals,optimizer_fits=6,
         classes=classes.tolist(),models=models,external_features=sha(evidence),
         extra_task_classifier_fits=0,simulated_path_fits=op,api_requests=0))
    print("disagreement",i,chosen,{k:v["net"] for k,v in checks.items()},flush=True)


def grade(out):
    root=out/STAGE
    for i in range(5):
        area=root/str(i)
        seal=read(area/"sealed.json")
        evidence=area/"external_features.npz"
        require(sha(evidence)==seal["external_features"],"Changed deployment features")
        with np.load(evidence,allow_pickle=False) as f:
            features=(f["prior"],f["features"])
        for head,digest in seal["models"].items():
            path=area/(head+".joblib")
            require(sha(path)==digest,"Changed research head")
            p=joblib.load(path).predict(features)
            actual=np.asarray(seal["classes"])[(p>.5).astype(int)].astype(str)
            require(np.array_equal(actual,np.load(area/(head+".npy"),allow_pickle=False)),"Replayed head differs")
    save(root/"score_lock.json",dict(protocol=sha(PLAN),folds={str(i):sha(root/str(i)/"sealed.json") for i in range(5)}))
    hits,groups,records={},[],[]
    for i in range(5):
        area=root/str(i)
        fold=origin(out,i)
        seal=read(area/"sealed.json")
        p=fold/"outer_answers.joblib"
        require(sha(p)==read(fold/"inputs.json")["outer_answers"] and seal["decision"]==sha(area/"decision.json"),
                "Wrong score packet")
        packet=joblib.load(p)
        y=packet["y"].astype(str)
        groups.extend(packet["groups"])
        scores={}
        for k,s in seal["predictions"].items():
            path=area/(k+".npy")
            require(sha(path)==s,"Changed prediction")
            v=np.load(path,allow_pickle=False)==y
            scores[k]=int(v.sum())
            hits.setdefault(k,[]).extend(v)
        records.append(dict(fold=i,rows=len(y),correct=scores,decision=read(area/"decision.json"),path_fits=seal["simulated_path_fits"]))
    total={k:int(np.sum(v)) for k,v in hits.items()}
    require(total["retain"]==read(out/"seed_completion/result.json")["correct"]["replicated"],"Wrong frozen five-fold score")
    paired={k:compare(np.asarray(h),np.asarray(hits["retain"]),groups) for k,h in hits.items() if k!="retain"}
    gains=[(r["correct"]["chosen"]-r["correct"]["retain"])/r["rows"] for r in records]
    value=dict(lock=sha(root/"score_lock.json"),rows=len(groups),folds=records,correct=total,paired=paired,
               advance=bool(paired["chosen"]["gain"]>=.003 and sum(d>0 for d in gains)>=3 and min(gains)>=-.003),
               new_optimizer_fits=30,new_task_classifier_fits=0,model_api_requests=0,designated_validation_opened=False,
               official_test_opened=False,h2_opened=False)
    save(root/"result.json",value)
    print(total,paired["chosen"],value["advance"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    parser.add_argument("--folds",type=int,choices=range(5),nargs="+",default=list(range(5)))
    args=parser.parse_args()
    if args.action=="build":
        from scripts.study_task1_disagreement_reading import build as build_portable
        for i in args.folds:
            build_portable(args.out,i)
    else:
        grade(args.out)
