"""Give the one new asset a matched trial alongside the already elected portfolio."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import measure
from scripts.study_resource_audition import ROLES, paired
from scripts.study_static_menu import PLAN
from scripts.unified_released_study import env, read, require, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups,take


def run(root):
    out=root/"static-audition"
    out.mkdir(exist_ok=True)
    old=root/"resource-audition"
    inputs=read(old/"inputs.json")
    prior=read(old/"confirmation.json")["choices"]["expanded"]
    named=read(root/"static-menu-cold/inputs.json")["candidates_added"]
    require(len(named)==1,"Expected exactly one new eligible asset")
    added="local_sentence_"+named[0][:20]
    reference={m:w*(1-prior["share"]) for m,w in inputs["protected"].items()}
    require(prior["member"] and prior["member"] not in reference and added not in reference,"Prior audition changed")
    reference[prior["member"]]=prior["share"]
    source=root/"arctic-menu/original/run/agent/refit_data.joblib"
    require(sha(source)==inputs["source"],"Training collection changed")
    data=joblib.load(source)
    rows,y=data["X_train"],np.asarray(data["y_train"])
    metric=MetricSpec.from_payload(data)
    groups=sample_groups(TrainingTask(data["kind"],rows,y),"shared_text")
    splitter=StratifiedGroupKFold(5,random_state=inputs["seed"],shuffle=True)
    split=dict(zip(ROLES,(h for _,h in splitter.split(groups,y,groups))))
    require(all(fingerprint(ix)==inputs["partitions"][name]["key"] for name,ix in split.items()),"Changed scoring group")
    library=root/"potion-library"
    physical=out/"physical.jsonl"
    settings=env(library,physical)
    activate_resource_directory(library)
    activate_resource_directory(library,settings)
    cache=str(root/"arctic-menu/fit_features")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=cache,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":cache})
    payload=old/"worker_input.joblib"
    registry=root/"analogy-portfolio-raw/portfolio_revised/run/agent/runtime_registry_0.json"
    fitting=read(old/"fitting.json")
    require(sha(payload)==fitting["payload"] and sha(registry)==fitting["registry"],"Unmatched original audition")
    lock(out/"inputs.json",dict(plan=sha(PLAN),election=sha(old/"confirmation.json"),input=sha(payload),
         added=added,reference=reference,source=sha(source),parity=sha(root/"static-menu-cold/parity.json")))
    digests=read(old/"predictions.json")["components"]
    yfirst=y[split["nomination"]]
    first={m:np.asarray(measure(m,payload,registry,old/m,settings,metric,yfirst)) for m in reference}
    first[added]=np.asarray(measure(added,payload,registry,out/added,settings,metric,yfirst))

    def compare(role,shares):
        bank=first if role=="nomination" else {}
        query=take(rows,split[role])
        for m in [*reference,added]:
            if m not in bank:
                model=joblib.load((out if m==added else old)/m/"trial.joblib")["model"]
                order=[list(model.classes_).index(c) for c in metric.labels]
                bank[m]=model.predict_proba(query)[:,order]
            if m in reference:
                h=hashlib.sha256(np.ascontiguousarray(bank[m]).tobytes()).hexdigest()
                require(h==digests[m][role],"Stored reference probability differs")
        protected=sum(w*bank[m] for m,w in reference.items())
        truth=y[split[role]]
        measure_score=lambda p:np.asarray(metric.labels)[p.argmax(axis=1)]==truth
        base=measure_score(protected)
        scores={w:paired(measure_score((1-w)*protected+w*bank[added]),base) for w in shares}
        return int(base.sum()),scores

    expected=read(old/"results.json")["decisions"]["expanded"]
    require(prior==expected,"The old winner is unbound")
    base,first_scores=compare("nomination",(0.,.25,.5,1.))
    share=min(first_scores,key=lambda w:(-first_scores[w]["correct"],w))
    lock(out/"nomination.json",dict(inputs=sha(out/"inputs.json"),protected=base,
         rows=len(yfirst),scores={str(w):v for w,v in first_scores.items()},selected=share))
    checks={}
    accepted=share>0
    for role in ROLES[1:]:
        if not accepted:
            break
        ref,scores=compare(role,(share,))
        checks[role]=dict(rows=len(split[role]),protected=ref,**scores[share])
        lock(out/(role+".json"),checks[role])
        accepted=scores[share]["net"]>0
    count=len(physical.read_text().splitlines())
    require(count==1,"Expected one additional physical classifier fit")
    result=dict(nomination=sha(out/"nomination.json"),added=added,initial_rows=len(yfirst),existing_correct=base,
         added_standalone=first_scores[1.]["correct"],elected_share=share,checks=checks,advance=accepted,
         physical_fits=count,reused_fits=len(reference),model_api_requests=0,designated_validation_opened=False,
         official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    run(p.parse_args().root)
