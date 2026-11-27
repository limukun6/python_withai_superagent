"""Propose executable regularization changes for a competitive measured pipeline."""
from functools import partial
import hashlib
import inspect
import json
import math

from .models import Candidate,search_candidates


def identifier(spec):
    content=json.dumps(spec,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return "refine_strength_"+hashlib.sha256(content).hexdigest()[:20]


def tunable(candidate):
    params=inspect.signature(candidate.fit_predict).parameters
    return next((k for k in ("regularization","c") if k in params),None)


def materialize(spec, templates=None):
    if (set(spec)!={"kind","operator","template","C"} or spec["operator"]!="retune_strength" or
        type(spec["C"]) not in (float,int) or not math.isfinite(spec["C"]) or not .001<=spec["C"]<=64):
        raise ValueError("Invalid evidence-derived regularization")
    if templates is None:
        templates={c.family:c for c in search_candidates(spec["kind"],"all")}
    base=templates.get(spec["template"])
    key=tunable(base) if base is not None else None
    if key is None or type(base.params.get("C")) not in (float,int) or spec["C"]==base.params["C"]:
        raise ValueError("No distinct supported C value for the specified pipeline")
    params=dict(base.params,C=spec["C"],parent_template=base.family)
    return Candidate(identifier(spec),params,partial(base.fit_predict,**{key:spec["C"]}))


def propose(kind,caps,history,train_rows):
    primary=[r for r in history if r["status"]=="completed" and r["split_id"]=="primary"]
    if not primary:
        return {},None
    ranked=sorted(primary,key=lambda r:(-r["search_score"],r["trial"]))
    templates={c.family:c for c in search_candidates(kind,"all")}
    for row in ranked:
        name=row["candidate"]
        if row["search_score"]<ranked[0]["search_score"]-1/math.sqrt(max(train_rows,1)):
            break
        if name not in caps or name not in templates or type(templates[name].params.get("C")) not in (float,int):
            continue
        if tunable(templates[name]) is None:
            continue
        gap=max(0.,row["train_score"]-row["search_score"])
        multiplier=math.exp(min(math.log(8.),max(math.log(1.7),6*gap)))
        center=templates[name].params["C"]
        entries={}
        for value in (center/multiplier,center*multiplier):
            value=round(min(64.,max(.001,value)),6)
            if value!=center:
                spec=dict(kind=kind,operator="retune_strength",template=name,C=value)
                entries[identifier(spec)]=spec
        return entries,dict(reference_trial=row["trial"],template=name,training_score=row["train_score"],
                            search_score=row["search_score"],gap=gap,prior_C=center,factor=multiplier)
    return {},None


def capability(spec, baseline):
    template=baseline[spec["template"]]
    return dict(template,id=identifier(spec),origin="evidence_refinement",params=dict(template["params"],C=spec["C"],
                parent_template=spec["template"]))
