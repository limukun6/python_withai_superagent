"""Learn joint nonlinear interactions on two representations elected from existing evidence."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.models import text_statistics
from superagent.prediction_evidence import fingerprint
from superagent.preference_analogy import embedding_function
from superagent.splits import sample_groups,take


PLAN=ROOT/"research/paired_encoder_search_1127/semantic_tree_protocol.json"


def fit_projections(providers,texts):
    result=[]
    for encode in providers:
        raw=encode(texts)
        mu=raw.mean(axis=0)
        sd=np.maximum(raw.std(axis=0),1e-4)
        pca=PCA(n_components=min(16,*raw.shape),svd_solver="randomized",random_state=3701)
        pca.fit(np.clip((raw-mu)/sd,-6,6))
        result.append((mu,sd,pca))
    return result


def matrices(rows,providers,projections):
    texts=list(dict.fromkeys(s for pair in rows for s in pair))
    ix={s:i for i,s in enumerate(texts)}
    features=[np.array([text_statistics(s) for s in texts])]
    for encoder,(mu,sd,pca) in zip(providers,projections,strict=True):
        features.append(pca.transform(np.clip((encoder(texts)-mu)/sd,-6,6)))
    matrix=np.concatenate(features,axis=1)
    a,b=[matrix[[ix[row[side]] for row in rows]] for side in (0,1)]
    delta=a-b
    midpoint=(a+b)*.5
    return np.column_stack([delta,midpoint]),np.column_stack([-delta,midpoint])


def run(root):
    out=root/"semantic-tree"
    out.mkdir(exist_ok=True)
    old=root/"resource-audition"
    upstream=read(old/"inputs.json")
    fitting=read(old/"fitting.json")
    digests=read(old/"predictions.json")["components"]
    source=root/"arctic-menu/original/run/agent/refit_data.joblib"
    require(sha(source)==upstream["source"] and sha(old/"worker_input.joblib")==fitting["payload"],
            "Changed audition training source")
    data=joblib.load(source)
    rows,y=data["X_train"],np.asarray(data["y_train"])
    metric=MetricSpec.from_payload(data)
    require(list(metric.labels)==[0,1] and metric.name=="accuracy","Expected side preference task")
    groups=sample_groups(TrainingTask(data["kind"],rows,y),"shared_text")
    folds=[h for _,h in StratifiedGroupKFold(5,shuffle=True,random_state=upstream["seed"]).split(groups,y,groups)]
    nomination,check,audit=folds[:3]
    fit=np.sort(np.concatenate(folds[3:]))
    parts=dict(fitting=fit,nomination=nomination,confirmation=check,audit=audit)
    require(all(upstream["partitions"][k]["key"]==fingerprint(v) for k,v in parts.items()) and
            sum(len(set(groups[v])) for v in parts.values())==len(set(groups)),"Changed or overlapping groups")
    activate_resource_directory(root/"arctic-library")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/"arctic-menu/fit_features"),
                      HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    fitted={}
    candidates=[]
    for name,record in digests.items():
        file=old/name/"trial.joblib"
        receipt=read(file.parent/"receipt.json")
        require(receipt["payload_sha256"]==fitting["payload"] and sha(file)==receipt["trial_sha256"],"Different teacher")
        obj=joblib.load(file)
        p=np.asarray(planner_observation(obj,obj["metrics"],dict(sha256=fitting["payload"],labels=y[nomination]),metric)["probabilities"])
        require(hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()==record["nomination"],"Unbound nomination score")
        model=obj["model"]
        fitted[name]=model
        feature=getattr(model,"named_steps",{}).get("features")
        if feature is not None:
            provider=embedding_function(feature)
            key=(feature.encoder_model,feature.encoder_revision)
            candidates.append((-int((p.argmax(axis=1)==y[nomination]).sum()),record["nomination"],key,name,provider))
    selected={}
    for _,_,key,name,provider in sorted(candidates,key=lambda c:c[:4]):
        selected.setdefault(key,(name,provider))
        if len(selected)==2:
            break
    require(len(selected)==2,"Too few suitable distinct backbones")
    members,providers=zip(*selected.values())
    ref=read(old/"confirmation.json")["choices"]["expanded"]
    weights={m:w*(1-ref["share"]) for m,w in upstream["protected"].items()}
    weights[ref["member"]]=ref["share"]
    lock(out/"inputs.json",dict(protocol=sha(PLAN),source=sha(source),selection=sha(old/"predictions.json"),
         reference=sha(old/"confirmation.json"),elected=list(members),weights=weights,fit_rows=len(fit),fit_groups=len(set(groups[fit]))))
    trained=out/"trained.joblib"
    seal=out/"trained.json"
    if not seal.exists():
        training=take(rows,fit)
        unique=list(dict.fromkeys(s for pair in training for s in pair))
        projections=fit_projections(providers,unique)
        x,reverse=matrices(training,providers,projections)
        tree=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,learning_rate=.06,
             min_samples_leaf=max(12,int(np.ceil(np.sqrt(len(fit))/2))),l2_regularization=1,
             early_stopping=False,random_state=3701)
        with (out/"fit_started.json").open("x") as f:
            f.write('{"attempted":1}\n')
        tree.fit(np.vstack([x,reverse]),np.r_[y[fit],1-y[fit]])
        joblib.dump((projections,tree),trained)
        lock(seal,dict(inputs=sha(out/"inputs.json"),model=sha(trained),new_classifier_fits=1,unsupervised_pcas=2))
    require(sha(trained)==read(seal)["model"],"Different tree")
    projections,tree=joblib.load(trained)
    def evaluate(role,idx,shares):
        batch=take(rows,idx)
        x,r=matrices(batch,providers,projections)
        positive=.5*(tree.predict_proba(x)[:,1]+1-tree.predict_proba(r)[:,1])
        new=np.c_[1-positive,positive]
        original={}
        for name in weights:
            model=fitted[name]
            p=model.predict_proba(batch)[:,[list(model.classes_).index(c) for c in metric.labels]]
            require(hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()==digests[name][role],
                    "Source no longer reproduces its archived reference")
            original[name]=p
        old_vote=sum(w*original[m] for m,w in weights.items())
        truth=y[idx]
        right=old_vote.argmax(axis=1)==truth
        unanimous=~np.any(np.vstack([a.argmax(axis=1)==truth for a in original.values()]),axis=0)
        corrected=new.argmax(axis=1)==truth
        scores={str(w):paired(((1-w)*old_vote+w*new).argmax(axis=1)==truth,right) for w in shares}
        return dict(role=role,rows=len(idx),protected_correct=int(right.sum()),head_correct=int(corrected.sum()),
                    all_prior_members_wrong=int(unanimous.sum()),head_fixed_unanimous=int((corrected&unanimous).sum()),
                    shares=scores)
    first=evaluate("confirmation",check,(0.,.25,.5,1.))
    share=min((0.,.25,.5,1.),key=lambda w:(-first["shares"][str(w)]["correct"],w))
    lock(out/"choice.json",dict(seal=sha(seal),first=first,share=share))
    second=evaluate("audit",audit,(share,)) if share>0 else None
    advance=bool(second and second["shares"][str(share)]["net"]>0 and
                 (first["shares"][str(share)]["net"]+second["shares"][str(share)]["net"])/(len(check)+len(audit))>=.002)
    result=dict(choice=sha(out/"choice.json"),elected=list(members),share=share,first=first,second=second,advance=advance,
                classifier_fits=1,unsupervised_projections=2,model_api_requests=0,designated_validation_opened=False,
                official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    with threadpool_limits(limits=2):
        run(p.parse_args().root)
