"""Reconsider an observed nonmember against the protected model on untouched scoring folds."""
import json
import os

import joblib
import numpy as np

from .ab_run import sha
from .analogy_review import incumbent_parts
from .experiment_policy import identity, option_keys
from .models import predict_with_model, probability_with_model
from .prediction_evidence import fingerprint
from .preference_analogy import WeightedProbabilityVote
from .splits import take
from .task_constraints import permitted
from .task_semantics import declared_semantics


EXTRA_FOLDS=("review_1","review_2")
SHARES=(.25,.5,1.)


def reserve_fits(total, normal_folds, *, other_reserve=0):
    # Cover two fresh comparisons and a final candidate fit while retaining the normal finalist search.
    reserve=min(total//3,len(EXTRA_FOLDS)*(normal_folds+1)+2)
    return reserve if total-reserve-other_reserve>=4*(normal_folds+1) else 0


def evidence(records, members, weights, metric):
    batch=[records[m] for m in members]
    first=batch[0]
    truth=np.asarray(first["labels"])
    matrices=[]
    for item in batch:
        matrix=np.asarray(item["probabilities"],dtype=float)
        if (item["row_key"]!=first["row_key"] or not np.array_equal(truth,item["labels"])
                or matrix.shape!=(len(truth),len(metric.labels)) or not np.isfinite(matrix).all()
                or (matrix<0).any() or (matrix>1).any() or not np.allclose(matrix.sum(axis=1),1,atol=1e-6)):
            raise ValueError("Mismatched or invalid row-level probability evidence")
        matrices.append(matrix)
    scores=np.average(matrices,axis=0,weights=weights)
    predicted=first["predictions"] if len(members)==1 else np.asarray(metric.labels)[scores.argmax(axis=1)]
    value=metric.evaluate(truth,predicted,scores,metric.labels)["score"]
    return dict(score=value,probabilities=scores,labels=truth,row_key=first["row_key"])


def combined_score(base, other, weight, metric):
    if weight==1.:
        return other["score"]
    matrix=(1-weight)*base["probabilities"]+weight*other["probabilities"]
    pred=np.asarray(metric.labels)[matrix.argmax(axis=1)]
    return metric.evaluate(base["labels"],pred,matrix,metric.labels)["score"]


def nominate(primary, members, weights, caps, budget, metric, restrictions, blocked):
    base=evidence(primary,members,weights,metric)
    keys=option_keys(caps,primary)
    base_cost=sum(caps[m]["classifier_fits"] for m in members)
    choices=[]
    for name,obs in primary.items():
        if name in members or name in blocked or name not in caps or obs["probabilities"] is None:
            continue
        cap=caps[name]
        cost=len(EXTRA_FOLDS)*(base_cost+cap["classifier_fits"])+cap["refit_classifier_fits"]
        if cost>budget or not permitted(cap,restrictions):
            continue
        other=evidence({name:obs},[name],[1.],metric)
        if other["row_key"]!=base["row_key"] or not np.array_equal(other["labels"],base["labels"]):
            raise ValueError("Candidate did not use the incumbent's ordered evaluation rows")
        for w in SHARES:
            choices.append(dict(member=name,share=w,primary=combined_score(base,other,w,metric),fits_required=cost))
    choices.sort(key=lambda c:(-c["primary"],c["share"],keys[c["member"]]))
    winner=choices[0] if choices and choices[0]["primary"]>base["score"]+1e-12 else None
    return base["score"],choices,winner


def review_portfolio(task,metric,incumbent,stop,executor,directory,log,timeout,primary,folds,blocked):
    from .autonomous_run import planner_observation
    old=(incumbent,stop)
    spent=executor.accounting()["attempted_classifier_fits"]
    members=stop["members"]
    report=dict(incumbent_model=incumbent["model_sha256"],incumbent_members=members,metric=metric.record(),
                permitted_shares=[0.,*SHARES],confirmations=[],before_fits=spent)
    def record(status,**values):
        report.update(status=status,**values)
        report["extra_fits"]=executor.accounting()["attempted_classifier_fits"]-spent
        log.write_json("portfolio_review.json",report)
    try:
        file=directory/incumbent["model_file"]
        if sha(file)!=incumbent["model_sha256"]:
            raise ValueError("Protected model identity changed")
        models,weights=incumbent_parts(members,joblib.load(file)["model"])
        if any(m not in primary or primary[m]["probabilities"] is None for m in members):
            record("retained_without_probability_evidence")
            return old
        base,trials,winner=nominate(primary,members,weights,executor.caps,executor.limit-spent,
                                    metric,executor.restrictions,blocked)
        log.write_json("portfolio_incumbent.json",incumbent)
        record("nomination_locked",base_primary=base,incumbent_weights=weights,considered=trials,nomination=winner)
        if winner is None:
            record("retained_primary")
            return old
        name,share=winner["member"],winner["share"]
        pending=winner["fits_required"]
        scores={"primary":winner["primary"]}
        for split in EXTRA_FOLDS:
            fit,held,_=folds[split]
            truth=task.y_train[held]
            payload=directory/(split+".joblib")
            joblib.dump(dict(kind=task.kind,X_train=take(task.X_train,fit),y_train=task.y_train[fit],
                X_search=take(task.X_train,held),y_search=truth,metric_spec=metric.record(),
                pair_semantics=declared_semantics(task),capture_probabilities=True),payload)
            payload_hash=sha(payload)
            bindings=dict(sha256=payload_hash,labels=truth)
            bank={}
            for member in [*members,name]:
                target=directory/f"portfolio_{split}_{identity(member)[:20]}.joblib"
                pending-=executor.caps[member]["classifier_fits"]
                _,facts=executor.worker(payload,member,target,timeout,reserve=pending)
                bank[member]=planner_observation(joblib.load(target),facts,bindings,metric)
            ref=evidence(bank,members,weights,metric)
            other=evidence(bank,[name],[1.],metric)
            proposed=combined_score(ref,other,share,metric)
            scores[split]=proposed
            report["confirmations"].append(dict(fold=split,held=fingerprint(held),fit=fingerprint(fit),
                payload=payload_hash,rows=len(held),baseline=ref["score"],candidate=proposed,delta=proposed-ref["score"]))
            record("confirming")
            if proposed+1e-12<ref["score"]:
                record("retained_confirmation")
                return old
        checks=report["confirmations"]
        gain=float(np.average([r["delta"] for r in checks],weights=[r["rows"] for r in checks]))
        if gain+1e-12<.002:
            record("retained_confirmation",weighted_confirmation_gain=gain)
            return old
        model,_=executor.worker(directory/"refit_data.joblib",name,directory/"portfolio_refit.joblib",timeout)
        final_members=[name] if share==1. else [*members,name]
        final_weights=[1.] if share==1. else [*(w*(1-share) for w in weights),share]
        fitted=model if share==1. else WeightedProbabilityVote(task.kind,[*models,model],metric.labels,final_weights)
        family=name if share==1. else "feedback_weighted_vote"
        destination=directory/"portfolio_selected_model.joblib"
        temporary=directory/"portfolio_selected_model.tmp"
        joblib.dump(dict(kind=task.kind,candidate=family,model=fitted),temporary)
        recovered=joblib.load(temporary)["model"]
        probe=take(task.X_train,np.linspace(0,len(task.y_train)-1,min(12,len(task.y_train)),dtype=int))
        pred=predict_with_model(task.kind,fitted,probe)
        pred2=predict_with_model(task.kind,recovered,probe)
        order,prob=probability_with_model(task.kind,fitted,probe)
        order2,prob2=probability_with_model(task.kind,recovered,probe)
        if not (np.array_equal(pred,pred2) and np.array_equal(order,order2) and np.allclose(prob,prob2,atol=1e-8)):
            raise ValueError("New portfolio failed independent model reload")
        os.replace(temporary,destination)
        overall=float(np.average(list(scores.values()),weights=[len(next(iter(primary.values()))["labels"]),
                                                                 *(r["rows"] for r in checks)]))
        proposal=dict(action="stop",members=final_members,split=None,utility=overall,
            reason="An observed alternative improved the task metric on fresh training-group checks.",
            evidence=dict(scores=scores,weights=final_weights,selection_evidence="portfolio_review.json"))
        selection=dict(incumbent,selected_experiment="stop_"+identity([final_members,final_weights])[:12],
            family=family,ensemble_components=final_members if len(final_members)>1 else [],single_candidate=final_members[0],
            selection_metric="primary plus two previously unscored training folds",selection_score=overall,fold_scores=scores,
            weights=final_weights,incumbent_model_sha256=incumbent["model_sha256"],selection_evidence="portfolio_review.json",
            model_file=destination.name,model_sha256=sha(destination))
        old_choice=json.loads((directory/"selection.json").read_text(encoding="utf-8"))
        record("accepted",weighted_confirmation_gain=gain,final_members=final_members,weights=final_weights,
               model_sha256=selection["model_sha256"])
        log.write_json("selection.json",dict(choice=proposal,scope="frozen from internal training feedback"))
        try:
            log.write_json("final_selection.json",selection)
        except OSError:
            log.write_json("selection.json",old_choice)
            raise
        return selection,proposal
    except (ValueError,RuntimeError,OSError,KeyError,AssertionError) as exc:
        record("retained_after_failed_opportunity",error=f"{type(exc).__name__}: {exc}"[:400])
        if executor.accounting()["unknown_failed_attempts"]:
            raise
        return old
