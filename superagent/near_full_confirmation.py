"""Compare supported alternatives at near-final training size without consuming the final refit."""
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import TrainingTask
from .experiment_policy import identity
from .fusion_holdout import compare_scores
from .models import predict_with_model, probability_with_model
from .splits import sample_groups, take
from .task_semantics import declared_semantics


def partition(task,seed,regular):
    groups=sample_groups(task,"shared_text")
    domain=set(task.y_train.tolist())
    split=StratifiedGroupKFold(8,shuffle=True,random_state=seed+103)
    for pool,held in split.split(np.zeros(len(groups)),task.y_train,groups):
        if set(task.y_train[held].tolist())!=domain or set(task.y_train[pool].tolist())!=domain:
            continue
        inner=TrainingTask(task.kind,take(task.X_train,pool),task.y_train[pool],task.spec)
        try:
            folds={k:(pool[a],pool[b],r) for k,(a,b,r) in regular(inner,seed).items()}
        except ValueError:
            continue
        if set(groups[pool]) & set(groups[held]):
            raise ValueError("Near-final comparison includes search groups")
        return inner,folds,held,dict(status="sealed",pool=pool.tolist(),sealed=held.tolist(),folds=8)
    return task,regular(task,seed),np.array([],dtype=int),dict(status="unavailable")


def shortlist(choice,menu,caps,remaining):
    options=[choice]
    used=set(choice["members"])
    eligible=[a for a in menu if a["action"]=="stop" and len(a["evidence"]["scores"])>=2
              and a["utility"]>=choice["utility"]-.025]
    # When the agent proposes a blend, first give its strongest confirmed constituent a matched vote.
    order=sorted(eligible,key=lambda a:(not(len(choice["members"])>1 and len(a["members"])==1
                  and a["members"][0] in used),-a["utility"],identity(a["members"])))
    for option in order:
        keys=option["members"]
        if any(set(keys)==set(o["members"]) for o in options):
            continue
        union=used.union(keys)
        refit=max(sum(caps[m]["refit_classifier_fits"] for m in a["members"]) for a in [*options,option])
        if len(union)>3 or sum(caps[m]["classifier_fits"] for m in union)+refit>remaining:
            continue
        used=union
        options.append(option)
        if len(options)==3:
            break
    return options


def audit(task,inner,sealed,choice,menu,history,metric,executor,directory,log,timeout):
    caps=executor.caps
    remaining=executor.limit-executor.accounting()["attempted_classifier_fits"]
    options=shortlist(choice,menu,caps,remaining)
    members=sorted({m for a in options for m in a["members"]})
    reserve=max(sum(caps[m]["refit_classifier_fits"] for m in a["members"]) for a in options)
    acquired=[r["training_seconds"] for r in history if r["status"]=="completed"]
    available_seconds=executor.start+executor.wall-time.monotonic()
    if len(options)<2 or not len(sealed) or available_seconds<(len(members)+reserve)*max(acquired,default=1.)+10:
        log.write_json("near_full_result.json",dict(action="retain",members=choice["members"],reason="No affordable comparison"))
        return choice
    log.write_json("near_full_nomination.json",dict(ordinary=choice["members"],choices=[a["members"] for a in options],
                   near_full_rows=len(inner.y_train),held=len(sealed),remaining_fits=remaining,reserved=reserve))
    path=Path(directory)/"near_full_fit_data.joblib"
    joblib.dump(dict(kind=inner.kind,mode="refit",X_train=inner.X_train,y_train=inner.y_train,
                     X_search=take(inner.X_train,[0]),metric_spec=metric.record(),pair_semantics=declared_semantics(inner)),path)
    classes=np.asarray(metric.labels)
    x=take(task.X_train,sealed)
    bank={}
    try:
        for i,m in enumerate(members):
            still_needed=sum(caps[n]["classifier_fits"] for n in members[i+1:])
            model,_=executor.worker(path,m,Path(directory)/("near_full_component_"+str(i)+".joblib"),timeout,
                                    reserve=reserve+still_needed)
            if caps[m]["supports_probabilities"]:
                order,p=probability_with_model(task.kind,model,x)
                bank[m]=np.asarray(p,dtype=float)[:,[list(order).index(c) for c in classes]]
            else:
                prediction=predict_with_model(task.kind,model,x)
                bank[m]=(np.asarray(prediction)[:,None]==classes[None,:]).astype(float)
        groups=sample_groups(task,"shared_text")[sealed]
        def matrix(option):
            return np.mean([bank[m] for m in option["members"]],axis=0)
        base=matrix(choice)
        diagnostics=[]
        passing=[]
        for alternative in options[1:]:
            accept,checks=compare_scores(metric,task.y_train[sealed],classes,base,matrix(alternative),groups)
            diagnostics.append(dict(members=alternative["members"],passed=bool(accept),scores=checks))
            if accept:
                passing.append((alternative,checks[0]["gain"]))
        selected=min(passing,key=lambda v:(-v[1],len(v[0]["members"]),identity(v[0]["members"])))[0] if passing else choice
        log.write_json("near_full_result.json",dict(action="replace" if passing else "retain",members=selected["members"],
                       ordinary=choice["members"],checks=diagnostics,used_high_fidelity_fits=len(members)))
        return selected
    except (ValueError,RuntimeError,OSError,TypeError,KeyError) as error:
        if executor.accounting()["unknown_failed_attempts"]:
            raise
        log.write_json("near_full_result.json",dict(action="retain",members=choice["members"],reason=str(error)[:260]))
        return choice
