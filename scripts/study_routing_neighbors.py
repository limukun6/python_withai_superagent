"""Learn when to listen to a new member from similar, genuinely held disagreements."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.study_disagreement_routing import probabilities
from scripts.study_portfolio_stability import group_interval
from scripts.study_resource_audition import paired
from scripts.study_resource_menu import inventory
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.data import TrainingTask
from superagent.local_sentence import bank_path
from superagent.metrics import MetricSpec
from superagent.paired_encoder import text_key
from superagent.prediction_evidence import fingerprint
from superagent.preference_analogy import views
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/routing_neighbors_protocol.json"


def adjusted(probability,sim,fit,apply,residual,groups,geometry):
    result=probability.copy()
    count=len(set(groups[fit]))
    if count<24:
        return result
    limit=min(24,max(3,int(np.ceil(np.sqrt(count)))))
    for i in apply:
        scores=sim[i,geometry]
        order=np.argsort(-scores,kind="stable")
        _,first_positions=np.unique(groups[fit[order]],return_index=True)
        which=order[np.sort(first_positions)[:limit]]
        w=np.maximum(0,scores[which])**2
        if w.sum()>0:
            result[i]+=float(w@residual[fit[which]])/(w.sum()+3*w.mean())
    return np.clip(result,0.,1.)


def run(root):
    out=root/"case-routing"
    out.mkdir(exist_ok=False)
    archive=root/"audition-scale"
    prior=read(archive/"inputs.json")
    member=prior["member"]
    source=root/"analogy-portfolio-raw/portfolio_revised/run/agent/refit_data.joblib"
    require(sha(source)==read(root/"portfolio-stability/inputs.json")["training"]
        and read(archive/"results.json")["advance"] and not read(root/"disagreement-routing/results.json")["advance"],
            "Prior registered studies changed")
    data=joblib.load(source)
    rows,y=data["X_train"],np.asarray(data["y_train"])
    groups=sample_groups(TrainingTask(data["kind"],rows,y),"shared_text")
    metric=MetricSpec.from_payload(data)
    spec,=[s for h,s in inventory(root/"arctic-library").items() if member=="local_sentence_"+h[:20]]
    vector_file=bank_path(root/"feature_bank",spec)
    lock(out/"inputs.json",dict(protocol=sha(PLAN),audition=sha(archive/"inputs.json"),
        previous_router=sha(root/"disagreement-routing/results.json"),training=sha(source),vector_file=sha(vector_file)))
    with np.load(vector_file,allow_pickle=False) as saved:
        idx={k:i for i,k in enumerate(saved["keys"].tolist())}
        raw=saved["vectors"]
        direction,midpoint=views(rows,lambda texts:raw[[idx[text_key(s)] for s in texts]])
    del idx,raw
    reports=[]
    for seed in prior["seeds"]:
        labels={arm:np.zeros(len(y),dtype=bool) for arm in ("base","matched","shuffled")}
        seen=np.zeros(len(y),dtype=bool)
        folds=[]
        setups=0
        for fold,(_,held) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(groups,y,groups),1):
            sub=Path(f"seed_{seed}")/f"fold_{fold}"
            old=root/"portfolio-stability"/sub
            part=read(old/"partition.json")
            require(not seen[held].any() and part["held"]==fingerprint(held),"Unexpected fold order")
            seen[held]=True
            truth=y[held]
            ensemble=sum(w*probabilities(old/m,m,part["payload"],truth,metric) for m,w in prior["protected"].items())
            addition=probabilities(archive/sub/member,member,part["payload"],truth,metric)
            baseline=(ensemble+addition).argmax(axis=1)
            labels["base"][held]=baseline==truth
            require(int(labels["base"][held].sum())==read(archive/sub/"outcome.json")["correct"],"Frozen vote differs")
            original,extra=ensemble.argmax(axis=1),addition.argmax(axis=1)
            disputed=np.flatnonzero(original!=extra)
            def confidence(p):
                return np.abs(logit(np.clip(p[disputed,1],1e-6,1-1e-6)))
            chance=expit(confidence(addition)-confidence(ensemble))
            residual=(extra[disputed]==truth[disputed]).astype(float)-chance
            mid=midpoint[held[disputed]]
            diff=direction[held[disputed]]
            similarity=.5*((mid@mid.T)+np.abs(diff@diff.T))
            full_roles=np.array([int(hashlib.sha256(("B trust disagreements v1/"+g).encode()).hexdigest()[:8],16)%3
                                 for g in groups[held]])
            local=full_roles[disputed]
            stats=[]
            for role in range(3):
                fit=np.flatnonzero(local==role)
                other=np.flatnonzero(local!=role)
                check=full_roles==(role+1)%3
                audit=full_roles==(role+2)%3
                previous=groups[held[disputed]]
                permutation=np.random.default_rng(seed+10*fold+role).permutation(fit)
                for arm,geometry in (("matched",fit),("shuffled",permutation)):
                    expectation=adjusted(chance,similarity,fit,other,residual,previous,geometry)
                    pred=baseline.copy()
                    high,low=expectation[other]>.5,expectation[other]<.5
                    pred[disputed[other[high]]]=extra[disputed[other[high]]]
                    pred[disputed[other[low]]]=original[disputed[other[low]]]
                    formed=len(set(previous[fit]))>=24
                    setups+=int(formed)
                    confirmation=paired(pred[check]==truth[check],baseline[check]==truth[check])
                    accepted=formed and confirmation["net"]>0
                    chosen=pred if accepted else baseline
                    labels[arm][held[audit]]=chosen[audit]==truth[audit]
                    stats.append(dict(arm=arm,role=role,fit_disagreements=len(fit),fit_groups=len(set(previous[fit])),
                                      confirmation=confirmation["net"],accepted=accepted))
            folds.append(dict(fold=fold,rows=len(held),disputed=len(disputed),rotations=stats,
                **{arm:paired(labels[arm][held],labels["base"][held]) for arm in ("matched","shuffled")}))
        require(seen.all(),"Incomplete out-of-fold audit")
        summary={arm:paired(labels[arm],labels["base"]) for arm in ("matched","shuffled")}
        report=dict(seed=seed,rows=len(y),equal_correct=int(labels["base"].sum()),**summary,
                    wins=sum(f["matched"]["net"]>0 for f in folds),
                    matched_vs_shuffled=paired(labels["matched"],labels["shuffled"]),
                    cluster_interval=group_interval(labels["matched"],labels["base"],groups,seed),reference_sets=setups)
        lock(out/f"seed_{seed}.json",dict(report,folds=folds))
        reports.append(report)
        if report["matched"]["net"]<=0 or report["wins"]<3 or report["matched_vs_shuffled"]["net"]<=0:
            break
    advance=len(reports)==2 and sum(r["wins"] for r in reports)>=7 and all(
        r["matched"]["net"]/r["rows"]>=.002 and r["matched_vs_shuffled"]["net"]>0 for r in reports)
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),assignments=reports,advance=advance,
        reference_sets=sum(r["reference_sets"] for r in reports),base_classifier_fits=0,model_api_requests=0,
        designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(reports,advance,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    with threadpool_limits(limits=2):
        run(p.parse_args().root)
