"""Use disjoint held groups to nominate, confirm and evaluate cheaper sub-portfolios."""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.study_disagreement_routing import probabilities
from scripts.study_portfolio_stability import group_interval
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.data import TrainingTask
from superagent.external_assets import model_resources
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN=ROOT/"research/paired_encoder_search_1127/member_removal_protocol.json"


def dependency_cost(root,members):
    assets={}
    for name in members:
        directory=root/"resource-audition"/name
        file=directory/"trial.joblib"
        require(sha(file)==read(directory/"receipt.json")["trial_sha256"],"Changed retained model")
        requirements,_=model_resources(joblib.load(file)["model"])
        records={}
        for res in requirements:
            files=res["files"].values() if res["path_kind"]=="directory" else [res["file"]]
            for f in files:
                records[f["sha256"]]=f["bytes"]
        assets[name]=records
    def cost(kept):
        blobs={}
        for name in kept:
            blobs.update(assets[name])
        return sum(blobs.values())
    return {drop:cost([m for m in members if m!=drop]) for drop in ["",*members]}


def run(root):
    out=root/"member-removal"
    out.mkdir(exist_ok=True)
    scale=root/"audition-scale"
    evidence=read(scale/"inputs.json")
    stable=root/"portfolio-stability"
    source=root/"analogy-portfolio-raw/portfolio_revised/run/agent/refit_data.joblib"
    require(sha(source)==read(stable/"inputs.json")["training"],"Different training roles")
    data=joblib.load(source)
    truth=np.asarray(data["y_train"])
    groups=sample_groups(TrainingTask(data["kind"],data["X_train"],truth),"shared_text")
    metric=MetricSpec.from_payload(data)
    require(metric.name=="accuracy","The recorded development gate uses accuracy")
    weights={m:(1-evidence["share"])*w for m,w in evidence["protected"].items()}
    weights[evidence["member"]]=evidence["share"]
    require(len(weights)>1 and np.isclose(sum(weights.values()),1),"Invalid protected allocation")
    costs=dependency_cost(root,weights)
    require(costs[""]>0,"No external resource available to measure")
    lock(out/"inputs.json",dict(protocol=sha(PLAN),election=sha(scale/"inputs.json"),
         seal=sha(scale/"results.json"),training=sha(source),weights=weights,external_bytes=costs))
    summaries=[]
    for seed in evidence["seeds"]:
        full=np.zeros(len(truth),dtype=bool)
        deployed=np.zeros(len(truth),dtype=bool)
        served=np.zeros(len(truth),dtype=bool)
        reduced=np.zeros(len(truth),dtype=bool)
        sizes=np.zeros(len(truth),dtype=np.int64)
        packets=[]
        for fold,(_,idx) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(groups,truth,groups),1):
            sub=Path(f"seed_{seed}")/f"fold_{fold}"
            a=stable/sub
            partition=read(a/"partition.json")
            require(partition["held"]==fingerprint(idx) and not served[idx].any(),"Overlapping held groups")
            y=truth[idx]
            banks={name:probabilities((scale/sub if name==evidence["member"] else a)/name,name,partition["payload"],y,metric)
                   for name in weights}
            probs={drop:np.average([p for n,p in banks.items() if n!=drop],axis=0,
                              weights=[weights[n] for n in banks if n!=drop]) for drop in costs}
            right={drop:np.asarray(metric.labels)[p.argmax(1)]==y for drop,p in probs.items()}
            require(int(right[""].sum())==read(scale/sub/"outcome.json")["correct"],"Full vote failed to replay")
            full[idx]=right[""]
            roles=np.asarray([int(hashlib.sha256(("B remove voter v1/"+str(g)).encode()).hexdigest(),16)%3 for g in groups[idx]])
            rotations=[]
            for role in range(3):
                nom,confirm,audit=(roles==(role+v)%3 for v in range(3))
                require(min(nom.sum(),confirm.sum(),audit.sum())>0,"Empty data role")
                def order(drop):
                    return (-int(right[drop][nom].sum()),costs[drop],len(weights)-bool(drop),
                            hashlib.sha256(probs[drop][nom].tobytes()).hexdigest())
                winner=min(probs,key=order)
                check=paired(right[winner][confirm],right[""][confirm])
                effective=winner if check["net"]>=0 else ""
                decision=dict(fold=fold,role=role,nomination=winner or None,
                    nomination_delta=int(right[winner][nom].sum())-int(right[""][nom].sum()),
                    confirmation=check,remove=effective or None)
                folder=out/sub
                folder.mkdir(parents=True,exist_ok=True)
                lock(folder/f"role_{role}.json",decision)
                rows=idx[audit]
                require(not served[rows].any(),"Repeated audit decision")
                served[rows]=True
                sizes[rows]=costs[effective]
                reduced[rows]=bool(effective)
                deployed[rows]=right[effective][audit]
                rotations.append(dict(**decision,audit=paired(deployed[rows],full[rows]),audit_rows=len(rows)))
            packets.append(dict(fold=fold,roles=rotations,net=paired(deployed[idx],full[idx])["net"]))
        require(served.all(),"Missing audit rows")
        record=dict(seed=seed,rows=len(truth),baseline=int(full.sum()),deployed=paired(deployed,full),
                    positive_folds=sum(p["net"]>0 for p in packets),removed_rows=int(reduced.sum()),
                    rows_with_fewer_resource_bytes=int((sizes<costs[""]).sum()),base_bytes=costs[""],
                    estimated_bytes_reduction=1-float(sizes.mean())/costs[""],
                    cluster_interval=group_interval(deployed,full,groups,seed))
        lock(out/f"seed_{seed}.json",dict(**record,folds=packets))
        summaries.append(record)
    deltas=[r["deployed"]["net"]/r["rows"] for r in summaries]
    accuracy=min(deltas)>=0 and max(deltas)>=.002
    efficiency=all(r["cluster_interval"][0]>-.003 and r["estimated_bytes_reduction"]>=.15 for r in summaries)
    result=dict(inputs=sha(out/"inputs.json"),assignments=summaries,accuracy_gate=accuracy,efficiency_gate=efficiency,
        model_fits=0,model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    run(parser.parse_args().root)
