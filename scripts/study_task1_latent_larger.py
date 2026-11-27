"""Verify the centering variant and compare grouped-response corrections on larger packets."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_latent_response import LatentResponse
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_remaining_folds import FOLDS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.autonomous_run import partitions
from superagent.joint_context import ContextTokenTree
from superagent.metrics import MetricSpec
from superagent.remaining_fold_review import NAMES
from superagent.score_evidence import checked_score_record

PLAN=ROOT/"research/task1_feedback_extension_1127/latent_larger.json"
MODEL="catboost_recurring_context"
REGIMES=("uncentered","centered")


def cached(root,name,x,y,fit,held):
    for arm in ("partner","regular","extended"):
        side=root/arm
        if not (side/"sealed.json").is_file():
            continue
        native=side/"run/agent"
        if name in NAMES[:3]:
            path=native/(name+".joblib")
            candidates=[native/("trial_"+str(r["trial"])+".joblib") for r in read(native/"history.json")
                        if r["status"]=="completed" and r["split_id"]==name and r["candidate"]==MODEL]
        else:
            path=native/(name+"_review.joblib")
            candidates=[]
            decision_file=native/"remaining_fold_nomination.json"
            if decision_file.is_file():
                opts=read(decision_file)["choices"]
                choices=sorted({m for a in opts for m in a})
                if MODEL in choices:
                    dest=native/(name+"_review_"+str(choices.index(MODEL))+".joblib")
                    if dest.is_file():
                        candidates=[dest]
        if not candidates or not path.exists():
            continue
        payload=joblib.load(path)
        if not all(np.array_equal(payload[k],v) for k,v in (("X_train",x[fit]),("y_train",y[fit]),
                                      ("X_search",x[held]),("y_search",y[held]))):
            continue
        for f in candidates:
            b=joblib.load(f)
            evidence=b["prediction_record"]
            metric=MetricSpec.from_payload(payload)
            require(evidence["evaluation_payload_sha256"]==sha(path),"Unbound worker payload")
            p,classes=checked_score_record(evidence["score_record"],evidence,metric,MODEL,y[held])
            require(classes==np.unique(y).tolist(),"Unaligned cross-fitted score")
            return p[:,1],dict(source="existing_worker",arm=arm,payload_sha256=sha(path),worker_sha256=sha(f))
    return None


def solve(out):
    base=out/"remaining_folds"
    require(not (base/"scoring_lock.json").exists(),"Outer scores already opened")
    source=out/"development"
    require(read(source/"latent_election/results.json")["eligible"] is False,"Wrong source study")
    for i in FOLDS:
        root=base/str(i)
        info=read(root/"inputs.json")
        train=root/"anonymous/training.zip"
        query=root/"anonymous/queries.zip"
        require(sha(train)==info["entry_inputs"] and sha(query)==info["entry_queries"],"Changed packets")
        x,y=arrays(train,["C_train.npy","y_train.npy"])
        q,=arrays(query,["C_test.npy"])
        classes,encoded=np.unique(y,return_inverse=True)
        task=SimpleNamespace(kind="categorical_tabular",X_train=x,y_train=y,spec={})
        split=partitions(task,18643,names=NAMES)
        require(len(split)==5,"Need a complete group-held teacher")
        dest=root/"latent_diagnostic"
        dest.mkdir(exist_ok=False)
        oof=np.full(len(y),np.nan)
        covered=np.zeros(len(y),dtype=int)
        sources=[]
        for name,(fit,held,_) in split.items():
            observed=cached(root,name,x,y,fit,held)
            if observed is None:
                tree=ContextTokenTree(dict(pair_limit=3)).fit(x[fit],y[fit])
                require(np.array_equal(tree.classes_,classes),"Unaligned teacher")
                observed=tree.predict_proba(x[held])[:,1],dict(source="new_fit")
            oof[held]=observed[0]
            covered[held]+=1
            sources.append(dict(fold=name,fit_rows=len(fit),held_rows=len(held),**observed[1]))
        require(np.isfinite(oof).all() and (covered==1).all(),"Incomplete teacher")
        original=root/"measurement_nominee"
        ref=read(original/"sealed.json")
        prior=original/"original_single.npy"
        require(sha(prior)==ref["predictions"]["original_single"] and ref["classes"]==classes.tolist(),"Unfrozen full tree")
        before=np.load(prior,allow_pickle=False)[:,1]
        hashes={}
        for regime in REGIMES:
            model=LatentResponse(4,centered=regime=="centered").fit(x,encoded,oof)
            prob=model.predict(q,before)
            require(np.isfinite(prob).all() and np.allclose(prob.sum(1),1.),"Invalid mixture")
            target=dest/(regime+".npy")
            np.save(target,prob,allow_pickle=False)
            hashes[regime]=sha(target)
        new=sum(rec["source"]=="new_fit" for rec in sources)
        save(dest/"sealed.json",dict(protocol=sha(PLAN),inputs=sha(root/"inputs.json"),original=sha(original/"sealed.json"),
            classes=classes.tolist(),predictions=hashes,teacher_sources=sources,new_teacher_fits=new,reused_teacher_fits=5-new,
            latent_model_fits=2,initializations=2,joint_m_steps=28,api_requests=0))
        print(i,"new teacher fits",new,"reused",5-new,flush=True)
    save(out/"latent_all_sealed.json",dict(protocol=sha(PLAN),folds={str(i):sha(base/str(i)/"latent_diagnostic/sealed.json") for i in FOLDS}))


def development(out):
    dest=out/"latent_centering_check"
    dest.mkdir(exist_ok=False)
    dev=out/"development"
    info=read(dev/"inputs.json")
    fit=bound(dev,"fitting.joblib",info)
    classes,y=np.unique(fit["y"],return_inverse=True)
    t=dev/"distill_teachers"
    require(sha(t/"out_of_fold.npy")==read(t/"sealed.json")["predictions"],"Unbound development teacher")
    m=LatentResponse(4,centered=True).fit(fit["x"],y,np.load(t/"out_of_fold.npy",allow_pickle=False)[:,0])
    reference=dev/"trial_anchor"
    before=read(reference/"sealed.json")
    results={}
    for role in ROLES:
        f=reference/(role+".npy")
        require(sha(f)==before["predictions"][role],"Changed original tree")
        q=bound(dev,role+"_features.joblib",info)
        probs=m.predict(q,np.load(f,allow_pickle=False)[:,1])
        output=dest/(role+".npy")
        np.save(output,probs,allow_pickle=False)
        results[role]=sha(output)
    save(dest/"locked.json",dict(protocol=sha(PLAN),predictions=results))
    rows=[]
    for role in ROLES:
        truth=bound(dev,role+"_answers.joblib",info)
        result={}
        for name, p, receipt in (("original",reference/(role+".npy"),before),
                                 ("uncentered",dev/"latent_four"/(role+".npy"),read(dev/"latent_four/sealed.json")),
                                 ("centered",dest/(role+".npy"),read(dest/"locked.json"))):
            require(sha(p)==receipt["predictions"][role],"Changed development prediction")
            result[name]=int((classes[np.load(p,allow_pickle=False).argmax(1)]==truth).sum())
        rows.append(dict(role=role,rows=len(truth),correct=result))
    save(dest/"result.json",dict(lock=sha(dest/"locked.json"),rows=rows,model_fits=1,initializations=1,joint_m_steps=14))
    print(rows,flush=True)


def grade(out):
    base=out/"remaining_folds"
    sealed=read(out/"latent_all_sealed.json")
    ordinary=read(base/"result.json")
    hits={n:[] for n in ("original_single","partner","regular",*REGIMES)}
    rows=[]
    groups=[]
    for i in FOLDS:
        root=base/str(i)
        dest=root/"latent_diagnostic"
        d=read(dest/"sealed.json")
        require(sealed["folds"][str(i)]==sha(dest/"sealed.json"),"Changed latent solution")
        labels=root/"outer_answers.joblib"
        require(sha(labels)==read(root/"inputs.json")["outer_answers"],"Changed outer assessment")
        key=joblib.load(labels)
        y=key["y"].astype(str)
        current={}
        for name in hits:
            if name in ("partner","regular"):
                f=root/name/"solve.csv"
                require(sha(f)==read(root/name/"sealed.json")["prediction"],"Changed ordinary solution")
                result=prediction_csv(f,len(y))
            else:
                origin=root/"measurement_nominee" if name=="original_single" else dest
                f=origin/(name+".npy")
                record=read(origin/"sealed.json")
                require(sha(f)==record["predictions"][name],"Changed model predictions")
                result=np.asarray(record["classes"]).astype(str)[np.load(f,allow_pickle=False).argmax(1)]
            current[name]=result==y
            hits[name].extend(current[name])
        groups.extend(key["groups"])
        rows.append(dict(fold=i,rows=len(y),correct={n:int(v.sum()) for n,v in current.items()}))
    references=("original_single","partner","regular")
    comparisons={reg:{n:compare(np.asarray(hits[reg]),np.asarray(hits[n]),groups) for n in references} for reg in REGIMES}
    primary="uncentered"
    qualify=True
    for n in references:
        gain=[(r["correct"][primary]-r["correct"][n])/r["rows"] for r in rows]
        qualify=qualify and comparisons[primary][n]["gain"]>=.003 and sum(d>0 for d in gain)>=2 and min(gain)>=-.003
    result=dict(protocol=sha(PLAN),lock=sha(out/"latent_all_sealed.json"),ordinary_lock=ordinary["lock"],folds=rows,
         correct={n:int(np.sum(v)) for n,v in hits.items()},comparisons=comparisons,advance_to_ordinary=bool(qualify),api_requests=0)
    save(out/"latent_diagnostic_result.json",result)
    print(result["correct"],result["advance_to_ordinary"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("development","solve","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    with threadpool_limits(limits=2):
        {"development":development,"solve":solve,"grade":grade}[a.action](a.out)
