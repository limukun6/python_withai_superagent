"""Recheck a proposed probability blend on groups excluded from the entire search."""
import hashlib
from pathlib import Path

import joblib
import numpy as np

from .ab_run import sha
from .experiment_policy import identity
from .models import probability_with_model
from .splits import sample_groups, take


def compare_scores(metric, labels, classes, individual, vote, groups):
    # Hash whole input groups before scoring either member of the frozen comparison.
    sides=np.fromiter((hashlib.sha256(str(g).encode("utf8")).digest()[0] % 2 for g in groups),dtype=int)
    scores=[]
    for mask in (np.ones(len(labels),dtype=bool), sides==0, sides==1):
        if mask.sum()<25 or set(labels[mask].tolist())!=set(classes.tolist()):
            return False,scores
        row={}
        for name,p in (("single",individual),("vote",vote)):
            matrix=p[mask]
            row[name]=metric.evaluate(labels[mask],classes[matrix.argmax(axis=1)],matrix,classes)["score"]
        row.update(rows=int(mask.sum()),gain=row["vote"]-row["single"])
        scores.append(row)
    return scores[0]["gain"]>=.003 and min(row["gain"] for row in scores[1:])>=0,scores


def review(task,metric,choice,history,observed,sealed,directory,log):
    members=choice["members"]
    if len(members)!=2 or not len(sealed):
        log.write_json("fusion_check.json",dict(action="retain",members=members,reason="Single selection or no separate groups"))
        return choice
    folds=[f for f in observed if all(m in observed[f] for m in members)]
    if not folds:
        return choice
    weights=[len(observed[f][members[0]]["labels"]) for f in folds]
    means={m:float(np.average([observed[f][m]["score"] for f in folds],weights=weights)) for m in members}
    single=min(members,key=lambda m:(-means[m],identity(m)))
    lookup={(r["candidate"],r["split_id"]):Path(directory)/("trial_"+str(r["trial"])+".joblib")
            for r in history if r["status"]=="completed"}
    sources={m:{f:sha(lookup[m,f]) for f in folds} for m in members}
    log.write_json("fusion_nomination.json",dict(vote=members,single=single,matching_folds=folds,
                    scores=means,sources=sources,rows=len(sealed)))
    fallback=dict(action="stop",members=[single],split=None,utility=means[single],
                  evidence=dict(scores={f:observed[f][single]["score"] for f in folds}),
                  reason="The individual classifier retained its position on the separate training comparison.")
    try:
        x=take(task.X_train,sealed)
        classes=np.asarray(metric.labels)
        scores={}
        for m in members:
            parts=[]
            for f in folds:
                fitted=joblib.load(lookup[m,f])["model"]
                labels,p=probability_with_model(task.kind,fitted,x)
                order=[list(labels).index(c) for c in classes]
                parts.append(np.asarray(p,dtype=float)[:,order])
                del fitted
            scores[m]=np.mean(parts,axis=0)
            if not np.isfinite(scores[m]).all():
                raise ValueError("Invalid probability on the reserved comparison")
        vote=np.mean([scores[m] for m in members],axis=0)
        groups=sample_groups(task,"shared_text")[sealed]
        keep,checks=compare_scores(metric,task.y_train[sealed],classes,scores[single],vote,groups)
        log.write_json("fusion_check.json",dict(action="keep_vote" if keep else "choose_single",
                       members=members if keep else [single],scores=checks,inference_fits=0))
        return choice if keep else fallback
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        log.write_json("fusion_check.json",dict(action="choose_single",members=[single],reason=str(error)[:250]))
        return fallback
