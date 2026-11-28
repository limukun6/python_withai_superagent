"""Scale up the elected audition winner on two locked group-disjoint assignments."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import measure
from scripts.study_portfolio_stability import group_interval
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT, env, read, require, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/audition_scale_protocol.json"


def run(root):
    out=root/"audition-scale"
    out.mkdir(exist_ok=True)
    trial=root/"resource-audition"
    scale=root/"portfolio-stability"
    choice=read(trial/"confirmation.json")["choices"]
    receipt=read(trial/"results.json")
    require(choice["original"]==choice["expanded"] and receipt["confirmation"]==sha(trial/"confirmation.json")
            and receipt["audit_vs_protected"]["expanded"]["net"]>0,"No common improving nominee")
    member,share=choice["expanded"]["member"],choice["expanded"]["share"]
    require(member and share>0,"No elected additional model")
    protected=read(trial/"inputs.json")["protected"]
    archive=read(scale/"inputs.json")
    matched=[k for k,s in archive["recipes"].items() if dict(zip(s["members"],s["weights"]))==protected]
    require(len(matched)==1 and member not in protected,"Different protected election")
    original=root/"analogy-portfolio-raw/portfolio_revised/run/agent"
    source=original/"refit_data.joblib"
    require(sha(source)==archive["training"],"Changed training packet")
    d=joblib.load(source)
    labels=np.asarray(d["y_train"])
    metric=MetricSpec.from_payload(d)
    classes=np.asarray(metric.labels)
    groups=sample_groups(TrainingTask(d["kind"],d["X_train"],labels),"shared_text")
    lock(out/"inputs.json",dict(protocol=sha(PLAN),election=sha(trial/"confirmation.json"),trigger=sha(trial/"results.json"),
         archived_assignments=sha(scale/"inputs.json"),member=member,share=share,protected=protected,
         seeds=archive["seeds"],rows=len(labels),groups=len(set(groups))))
    fits=out/"physical.jsonl"
    library=root/"arctic-library"
    settings=env(library,fits)
    activate_resource_directory(library,settings)
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(root/"feature_bank")})
    folds,summaries=[],[]
    for seed in archive["seeds"]:
        splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        base_correct=np.zeros(len(labels),dtype=bool)
        new_correct=np.zeros(len(labels),dtype=bool)
        visited=np.zeros(len(labels),dtype=bool)
        for i,(fit,held) in enumerate(splitter.split(groups,labels,groups),1):
            old=scale/f"seed_{seed}"/f"fold_{i}"
            partition=read(old/"partition.json")
            payload=old/"training.joblib"
            require(partition==dict(fit=fingerprint(fit),held=fingerprint(held),payload=sha(payload)) and
                    not visited[held].any() and not set(groups[fit])&set(groups[held]),"Changed or overlapping partitions")
            visited[held]=True
            expected=read(old/"metrics.json")["correct"][matched[0]]
            registry=original/"runtime_registry_0.json"
            for baseline_member in protected:
                require((old/baseline_member/"receipt.json").exists(),"Cannot refit a missing reference")
            reference=sum(w*np.asarray(measure(m,payload,registry,old/m,settings,metric,labels[held]))
                          for m,w in protected.items())
            base_correct[held]=classes[reference.argmax(axis=1)]==labels[held]
            require(int(base_correct[held].sum())==expected,"Original arm did not reproduce")
            folder=out/f"seed_{seed}"/f"fold_{i}"
            challenger=np.asarray(measure(member,payload,registry,folder/member,settings,metric,labels[held]))
            new_correct[held]=classes[((1-share)*reference+share*challenger).argmax(axis=1)]==labels[held]
            evidence=dict(seed=seed,fold=i,held=len(held),fitting=len(fit),protected=expected,
                          **paired(new_correct[held],base_correct[held]))
            lock(folder/"outcome.json",evidence)
            folds.append(evidence)
            print(seed,i,evidence["net"],flush=True)
        require(visited.all(),"Incomplete out-of-fold coverage")
        result=dict(seed=seed,rows=len(labels),protected=int(base_correct.sum()),**paired(new_correct,base_correct),
                    positive_folds=sum(r["net"]>0 for r in folds if r["seed"]==seed),
                    cluster_interval=group_interval(new_correct,base_correct,groups,seed))
        lock(out/f"seed_{seed}"/"result.json",result)
        summaries.append(result)
        if result["net"]<=0 or result["positive_folds"]<3:
            break
    advanced=len(summaries)==2 and all(r["net"]/r["rows"]>=.002 for r in summaries) and sum(r["net"]>0 for r in folds)>=7
    actual=len(fits.read_text().splitlines())
    require(actual==len(folds)<=10,"Unexpected new fit count")
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),assignments=summaries,folds=folds,advance=advanced,
        actual_fits=actual,reused_fits=len(protected)*len(folds),model_api_requests=0,designated_validation_opened=False,
        official_test_opened=False,H2_opened=False))
    print(summaries,advanced,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=lambda v:Path(v).resolve(),required=True)
    run(p.parse_args().root)
