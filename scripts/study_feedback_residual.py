"""Fit an antisymmetric correction to actual out-of-fold errors of a frozen vote."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.study_disagreement_routing import correction
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT,read,require,sha,predictions
from superagent.analogy_review import incumbent_parts
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask,load_prediction_features
from superagent.experiment_policy import identity
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model,text_statistics
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/feedback_residual_protocol.json"


def coordinates(rows,vectorizer,svd):
    texts=list(dict.fromkeys(t for pair in rows for t in pair))
    lookup={s:i for i,s in enumerate(texts)}
    a,b=[np.array([lookup[row[j]] for row in rows]) for j in (0,1)]
    style=np.stack([text_statistics(t) for t in texts])
    topics=svd.transform(vectorizer.transform(texts))
    return style[a]-style[b],.5*(style[a]+style[b]),.5*(topics[a]+topics[b])


def design(parts,scales):
    delta,mid,topic=parts
    ds,mm,ms,tm,ts=scales
    d=np.clip(delta/ds,-4,4)
    m=np.clip((mid-mm)/ms,-4,4)
    t=np.clip((topic-tm)/ts,-4,4)
    products=(d[:,:,None]*t[:,None,:]).reshape(len(d),-1)/np.sqrt(t.shape[1])
    return np.column_stack([d,d*m,products])


def offset(bank,weights):
    p=np.average(bank,axis=0,weights=weights)
    return logit(np.clip(p[:,1],1e-6,1-1e-6)),p.argmax(axis=1)


def run(root,case):
    folder=root/("portfolio-recruitment" if case=="first" else "recruitment-replication")/"comparison"
    arm=folder/"challenger"
    agent=arm/"run/agent"
    out=root/"feedback-residual"/case
    out.mkdir(parents=True,exist_ok=True)
    seal=read(arm/"sealed.json")
    final=read(agent/"final_selection.json")
    file=agent/final["model_file"]
    members=seal["members"]
    require(sha(file)==seal["model"] and seal["supervised_fits"]<read(folder.parent/"inputs.json")["fit_cap"],
            "Unverified model or insufficient residual budget")
    models,weights=incumbent_parts(members,joblib.load(file)["model"])
    training=joblib.load(agent/"refit_data.joblib")
    metric=MetricSpec.from_payload(training)
    require(metric.name=="accuracy" and list(metric.labels)==[0,1] and training["pair_semantics"]=="preference",
            "Residual is defined for binary preferences")
    group=sample_groups(TrainingTask(training["kind"],training["X_train"],training["y_train"]),"shared_text")
    lookup={t:g for row,g in zip(training["X_train"],group) for t in row}
    history={(r["split_id"],r["candidate"]):agent/f"trial_{r['trial']}.joblib" for r in read(agent/"history.json")
             if r["status"]=="completed"}
    seen=set()
    def panel(name):
        path=agent/(name+".joblib")
        payload=joblib.load(path)
        digest=sha(path)
        rows=payload["X_search"]
        y=np.asarray(payload["y_search"])
        gs={lookup[t] for row in rows for t in row}
        require(not seen.intersection(gs),"Overlapping feedback roles")
        bank=[]
        for member in members:
            trial=(agent/("portfolio_"+name+"_"+identity(member)[:20]+".joblib") if name.startswith("review_")
                   else history.get((name,member),agent/f"analogy_{name}_{member}.joblib"))
            if not trial.exists():
                return None
            item=joblib.load(trial)
            facts=item["metrics"]
            require(facts["resolved_configuration"]["id"]==member,"Unexpected classifier")
            obs=planner_observation(item,facts,dict(sha256=digest,labels=y),metric)
            bank.append(np.asarray(obs["probabilities"]))
        seen.update(gs)
        off,base=offset(bank,weights)
        return rows,y,off,base
    earlier=[p for name in ("primary","confirmation_1","confirmation_2") if (p:=panel(name)) is not None]
    lock(out/"inputs.json",dict(protocol=sha(PLAN),run=sha(arm/"sealed.json"),members=members,weights=weights,
                               training_folds=len(earlier),fitting_groups=len(seen)))
    if len(earlier)<2:
        lock(out/"results.json",dict(eligible=False,new_fits=0,model_api_requests=0))
        return
    rows=[r for batch,_,_,_ in earlier for r in batch]
    labels=np.concatenate([y for _,y,_,_ in earlier])
    offsets=np.concatenate([o for _,_,o,_ in earlier])
    fitted=out/"adjustment.joblib"
    closed=out/"fitted.json"
    if not closed.exists():
        unique=list(dict.fromkeys(t for row in rows for t in row))
        vocab=TfidfVectorizer(min_df=2,max_features=12000,sublinear_tf=True)
        vocab_matrix=vocab.fit_transform(unique)
        dims=min(6,min(vocab_matrix.shape)-1)
        require(dims>=1,"Insufficient variation for topics")
        axis=TruncatedSVD(dims,random_state=6139).fit(vocab_matrix)
        parts=coordinates(rows,vocab,axis)
        d,m,t=parts
        scales=(np.maximum(np.sqrt(np.mean(d**2,axis=0)),1e-4),m.mean(0),np.maximum(m.std(0),1e-4),
                t.mean(0),np.maximum(t.std(0),1e-4))
        x=design(parts,scales)
        with (out/"fit_started.json").open("x") as f:
            f.write('{"attempted":1}\n')
        params=correction(x,offsets,labels)
        joblib.dump((vocab,axis,scales,params),fitted)
        lock(closed,dict(inputs=sha(out/"inputs.json"),parameters=sha(fitted),rows=len(rows),columns=x.shape[1],
                         classifier_fits=1,unsupervised_topics=1))
    require(read(closed)["parameters"]==sha(fitted),"Correction changed before validation")
    vocab,axis,scales,params=joblib.load(fitted)
    def predict(rows,off,base):
        value=off+design(coordinates(rows,vocab,axis),scales)@params
        return np.where(value==0,base,(value>0).astype(int))
    checks=[]
    for name in ("review_1","review_2"):
        test=panel(name)
        require(test is not None,"Incomplete confirmation for a selected component")
        check,y,off,base=test
        checks.append(dict(fold=name,rows=len(y),**paired(predict(check,off,base)==y,base==y)))
    allow=min(r["net"] for r in checks)>=0 and sum(r["net"] for r in checks)/sum(r["rows"] for r in checks)>=.002
    lock(out/"decision.json",dict(parameters=sha(closed),checks=checks,adopt=allow))
    activate_resource_directory(root/"arctic-library")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/"feature_bank"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    query=load_prediction_features(folder/"q.zip",training["kind"],read(agent/"task_spec.json"))
    bank=[]
    for model in models:
        order,p=probability_with_model(training["kind"],model,query)
        bank.append(p[:,[list(order).index(c) for c in metric.labels]])
    off,base=offset(bank,weights)
    require(sha(arm/"reload.csv")==seal["predictions"] and np.array_equal(base,predictions(arm/"reload.csv",len(query))),
            "Protected outer predictions changed")
    shadow=predict(query,off,base)
    deployed=shadow if allow else base
    digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    lock(out/"outer_lock.json",dict(decision=sha(out/"decision.json"),base=digest(base),shadow=digest(shadow),deployed=digest(deployed)))
    truth=folder/"truth.joblib"
    require(sha(truth)==read(folder.parent/"inputs.json")["cases"]["comparison"]["truth"],"Outer truth changed")
    y=joblib.load(truth)["labels"]
    unanimous=~np.any(np.stack([p.argmax(axis=1)==y for p in bank]),axis=0)
    result=dict(eligible=True,seal=sha(out/"outer_lock.json"),rows=len(y),fixed_correct=int((base==y).sum()),checks=checks,
                adopt=allow,shadow=paired(shadow==y,base==y),deployed=paired(deployed==y,base==y),
                unanimous_errors=int(unanimous.sum()),shadow_fixed_unanimous=int(((shadow==y)&unanimous).sum()),
                deployed_fixed_unanimous=int(((deployed==y)&unanimous).sum()),extra_fits=1,base_fits=0,model_api_requests=0,
                designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",required=True,choices=("first","second"))
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    with threadpool_limits(limits=2):
        run(args.root,args.case)
