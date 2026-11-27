"""Count which held-out mistakes could even be repaired by the four existing voters."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.study_disagreement_routing import probabilities
from scripts.unified_released_study import ROOT,read,require,sha
from superagent.data import TrainingTask
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PLAN=ROOT/"research/paired_encoder_search_1127/routing_headroom_protocol.json"


def run(root):
    out=root/"routing-headroom"
    out.mkdir(exist_ok=False)
    scale=root/"audition-scale"
    plan=read(scale/"inputs.json")
    source=root/"analogy-portfolio-raw/portfolio_revised/run/agent/refit_data.joblib"
    require(sha(source)==read(root/"portfolio-stability/inputs.json")["training"],"Training source mismatch")
    d=joblib.load(source)
    y=np.asarray(d["y_train"])
    groups=sample_groups(TrainingTask(d["kind"],d["X_train"],y),"shared_text")
    metric=MetricSpec.from_payload(d)
    weights=plan["protected"]
    names=[*weights,plan["member"]]
    lock(out/"inputs.json",dict(plan=sha(PLAN),assignment=sha(scale/"inputs.json"),results=sha(scale/"results.json"),
         training=sha(source),members=names,share=plan["share"]))
    states=[]
    reports=[]
    for seed in plan["seeds"]:
        right=np.zeros((len(y),len(names)),dtype=bool)
        combined=np.zeros(len(y),dtype=bool)
        incumbent=np.zeros(len(y),dtype=bool)
        seen=np.zeros(len(y),dtype=bool)
        grouping=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        for i,(_,held) in enumerate(grouping.split(groups,y,groups),1):
            sub=Path(f"seed_{seed}")/f"fold_{i}"
            old=root/"portfolio-stability"/sub
            partition=read(old/"partition.json")
            require(fingerprint(held)==partition["held"] and not seen[held].any(),"Overlapping or reordered rows")
            seen[held]=True
            bank={m:probabilities((old if m in weights else scale/sub)/m,m,partition["payload"],y[held],metric)
                  for m in names}
            protected=sum(w*bank[m] for m,w in weights.items())
            mixture=(1-plan["share"])*protected+plan["share"]*bank[names[-1]]
            combined[held]=np.asarray(metric.labels)[mixture.argmax(axis=1)]==y[held]
            incumbent[held]=np.asarray(metric.labels)[protected.argmax(axis=1)]==y[held]
            actual=read(scale/sub/"outcome.json")
            require(int(combined[held].sum())==actual["correct"] and int(incumbent[held].sum())==actual["protected"],
                    "Either deployed recipe diverged from its frozen evidence")
            right[held]=np.stack([np.asarray(metric.labels)[bank[m].argmax(axis=1)]==y[held] for m in names],axis=1)
        require(seen.all(),"Incomplete audit")
        total=right.sum(axis=1)
        none=total==0
        recoverable=~combined & (total>0)
        outsider=right[:,-1]
        require(int(none.sum()+recoverable.sum())==int((~combined).sum()),"Incomplete error decomposition")
        states.append(dict(wrong=~combined,none=none,recoverable=recoverable,right=right,
                           incumbent_only=incumbent&~outsider,outsider_only=outsider&~incumbent))
        reports.append(dict(seed=seed,rows=len(y),combined_correct=int(combined.sum()),combined_wrong=int((~combined).sum()),
            all_members_wrong=int(none.sum()),correct_member_overruled=int(recoverable.sum()),hindsight_any_member=int((total>0).sum()),
            both_blocks_right=int((incumbent&outsider).sum()),both_blocks_wrong=int((~incumbent&~outsider).sum()),
            incumbent_only=int((incumbent&~outsider).sum()),outsider_only=int((outsider&~incumbent).sum()),
            members={name:dict(correct=int(right[:,i].sum()),alone_right=int((right[:,i]&(total==1)).sum()))
                     for i,name in enumerate(names)}))
    first,second=states
    repeated={k:int((first[k]&second[k]).sum()) for k in ("wrong","none","recoverable","incumbent_only","outsider_only")}
    single_either=(first["right"].sum(axis=1)==1)&(second["right"].sum(axis=1)==1)
    repeated["same_member_alone_right"]=int((single_either&(first["right"]&second["right"]).any(axis=1)).sum())
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),assignments=reports,repeated=repeated,
        unique_training_rows=len(y),shared_title_groups=len(set(groups)),as_deployed_score=False,
        new_fits=0,model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(reports,repeated,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    run(p.parse_args().root)
