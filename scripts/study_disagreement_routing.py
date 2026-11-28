"""Learn when to deviate from a confirmed equal vote on groups all base models held out."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.study_portfolio_stability import group_interval
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/disagreement_router_protocol.json"


def probabilities(folder,member,payload,truth,metric):
    receipt=read(folder/"receipt.json")
    file=folder/"trial.joblib"
    require(receipt==dict(member=member,payload_sha256=payload,trial_sha256=sha(file),fits=1),"Inconsistent fitted probability source")
    bundle=joblib.load(file)
    return np.asarray(planner_observation(bundle,bundle["metrics"],dict(sha256=payload,labels=truth),metric)["probabilities"])


def correction(features,offset,targets):
    penalty=np.sqrt(len(targets))
    def loss(params):
        logits=offset+features@params
        value=(np.logaddexp(0.,logits)-targets*logits).sum()+penalty*np.dot(params,params)/2
        return value,features.T@(expit(logits)-targets)+penalty*params
    result=minimize(loss,np.zeros(features.shape[1]),jac=True,method="L-BFGS-B")
    require(result.success and np.isfinite(result.x).all(),"Correction did not converge")
    return result.x


def run(root):
    out=root/"disagreement-routing"
    out.mkdir(exist_ok=False)
    scale=root/"audition-scale"
    info=read(scale/"inputs.json")
    completed=read(scale/"results.json")
    require(completed["advance"] and info["share"]==.5,"Requires the elected equal-weight comparison at scale")
    member,protected=info["member"],info["protected"]
    source=root/"analogy-portfolio-raw/portfolio_revised/run/agent/refit_data.joblib"
    require(sha(source)==read(root/"portfolio-stability/inputs.json")["training"],"Different underlying training data")
    d=joblib.load(source)
    truth=np.asarray(d["y_train"])
    groups=sample_groups(TrainingTask(d["kind"],d["X_train"],truth),"shared_text")
    metric=MetricSpec.from_payload(d)
    require(list(metric.labels)==[0,1] and metric.name=="accuracy","Expected binary comparisons")
    lock(out/"inputs.json",dict(plan=sha(PLAN),audition=sha(scale/"inputs.json"),upstream=sha(scale/"results.json"),
         evaluation="training held groups only",source=sha(source),seeds=info["seeds"]))
    ledger=out/"meta_fits.jsonl"
    reports=[]
    for seed in info["seeds"]:
        full={k:np.zeros(len(truth),dtype=bool) for k in ("base","raw","decided")}
        seen=np.zeros(len(truth),dtype=bool)
        rows=[]
        for fold,(_,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(groups,truth,groups),1):
            sub=Path(f"seed_{seed}")/f"fold_{fold}"
            reference=root/"portfolio-stability"/sub
            partition=read(reference/"partition.json")
            require(partition["held"]==fingerprint(held) and not seen[held].any(),"Row order differs")
            seen[held]=True
            y=truth[held]
            probs=np.stack([probabilities(reference/m,m,partition["payload"],y,metric) for m in protected],axis=1)
            weights=np.asarray(list(protected.values()))
            aggregate=np.average(probs,axis=1,weights=weights)
            outsider=probabilities(scale/sub/member,member,partition["payload"],y,metric)
            old=aggregate.argmax(axis=1); new=outsider.argmax(axis=1)
            fixed_vote=((.5*aggregate+.5*outsider).argmax(axis=1))
            base=fixed_vote==y
            require(int(base.sum())==read(scale/sub/"outcome.json")["correct"],"Locked reference did not reproduce")
            disagree=old!=new
            confidence=lambda p:np.abs(logit(np.clip(p[:,1],1e-6,1-1e-6)))
            offset=confidence(outsider)-confidence(aggregate)
            support=np.average(probs.argmax(axis=2)==new[:,None],axis=1,weights=weights)
            variance=np.average((probs[:,:,1]-aggregate[:,1,None])**2,axis=1,weights=weights)
            x=np.column_stack([np.ones(len(y)),2*support-1,2*np.sqrt(variance)])
            roles=np.asarray([int(hashlib.sha256(("B trust disagreements v1/"+g).encode()).hexdigest()[:8],16)%3 for g in groups[held]])
            targets=(new==y).astype(float)
            full["base"][held]=base
            rotations=[]
            for role in range(3):
                fit=(roles==role)&disagree
                check=roles==(role+1)%3
                audit=roles==(role+2)%3
                count=len(set(groups[held][fit]))
                params=None
                raw=fixed_vote.copy()
                if count>=24 and len(set(targets[fit]))==2:
                    with ledger.open("a") as f:
                        f.write(json.dumps(dict(seed=seed,fold=fold,role=role,examples=int(fit.sum())))+"\n")
                    params=correction(x[fit],offset[fit],targets[fit])
                    margin=offset+x@params
                    raw[disagree&(margin>0)]=new[disagree&(margin>0)]
                    raw[disagree&(margin<0)]=old[disagree&(margin<0)]
                verified=paired(raw[check]==y[check],base[check])
                accepted=params is not None and verified["net"]>0
                final=raw if accepted else fixed_vote
                full["raw"][held[audit]]=raw[audit]==y[audit]
                full["decided"][held[audit]]=final[audit]==y[audit]
                rotations.append(dict(role=role,fit_disagreements=int(fit.sum()),fit_groups=count,accepted=accepted,
                    correction=None if params is None else params.tolist(),confirmation=verified,
                    audit=paired(final[audit]==y[audit],base[audit])))
            rows.append(dict(fold=fold,held=len(y),disagreements=int(disagree.sum()),rotations=rotations,
                             audit=paired(full["decided"][held],base)))
        require(seen.all(),"Incomplete assignment")
        outcome=dict(seed=seed,rows=len(truth),elected_equal_correct=int(full["base"].sum()),
                     deployed=paired(full["decided"],full["base"]),unrestricted=paired(full["raw"],full["base"]),
                     cluster_interval=group_interval(full["decided"],full["base"],groups,seed),
                     accepted_rotations=sum(x["accepted"] for f in rows for x in f["rotations"]),
                     positive_folds=sum(f["audit"]["net"]>0 for f in rows),folds=rows)
        lock(out/f"seed_{seed}.json",outcome)
        reports.append({k:v for k,v in outcome.items() if k!="folds"})
        if outcome["deployed"]["net"]<=0 or outcome["positive_folds"]<3:
            break
    eligible=len(reports)==2 and all(r["deployed"]["net"]/r["rows"]>=.002 for r in reports) and sum(r["positive_folds"] for r in reports)>=7
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),assignments=reports,advance=eligible,
         meta_fits=len(ledger.read_text().splitlines()) if ledger.exists() else 0,base_fits=0,model_api_requests=0,
         designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(reports,eligible,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    run(p.parse_args().root)
