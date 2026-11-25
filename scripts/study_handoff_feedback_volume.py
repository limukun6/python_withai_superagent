"""Match two gate-learning feedback sizes while holding native evaluation predictions fixed."""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from scripts.prepare_conditional_handoff import ROOT, ROLES, partition
from scripts.run_conditional_handoff import comparison, learned, load_export, score_panel
from scripts.unified_released_study import EVIDENCE, read, require, save, sha
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.preference_handoff import binary, inputs, predict, select

PLAN=ROOT/"docs/batches/unified_feedback_volume_plan_20261124_zh.md"


def index_scores(out,n):
    folder=out/f"fold_{n}"
    expected=read(out/"protocol.json")["splits"][n]
    require(sha(folder/"probabilities.npz")==read(folder/"scores_frozen.json")["scores"] and
            sha(folder/"features.joblib")==expected["features"],"Unexpected source probabilities")
    panels=joblib.load(folder/"features.joblib")
    index={}
    with np.load(folder/"probabilities.npz",allow_pickle=False) as scores:
        for r in ROLES:
            ids=panels[r]["ids"]
            require(fingerprint(ids)==expected["roles"][r]["ids"] and scores[r].shape==(len(ids),3),"Row order differs")
            for row_id,probs in zip(ids,scores[r],strict=True):
                require(row_id not in index,"Duplicate source probability")
                index[row_id]=probs
    return index


def route_outputs(target_folder, target_probs, pairs, test_ids):
    outer=inputs(np.stack([target_probs[i] for i in test_ids]),[pairs[i] for i in test_ids])
    cols={"row_id":test_ids,"old_unanimous":outer["unanimous"],"third_conflict":outer["conflict"],
          "baseline":outer["base"],"equal_three":binary(outer["p"].mean(axis=1),outer["ties"])}
    for name in ("small","pooled"):
        saved=read(target_folder/(name+".json"))
        trial=predict(saved["policy"],outer)
        cols[name+"_shadow"]=trial
        cols[name]=trial if saved["selection"]["chosen"]=="soft" else outer["base"]
    cols["agent"]=cols["pooled"]
    return cols


def write(path,cols):
    with path.open("x",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(cols)
        for row in zip(*cols.values(),strict=True):
            writer.writerow(map(int,row))


def freeze(out,packet):
    protocol=read(out/"protocol.json")
    require((out/"local_five_fold_results.json").is_file() and sha(packet)==protocol["source_training_packet"],
            "Prior study or training packet missing")
    task=load_training_task(packet)
    groups=sample_groups(task,"shared_text")
    folds=list(partition(groups,np.asarray(read(EVIDENCE/"title_reserve_ids.json"))))
    destination=out/"feedback_volume"
    destination.mkdir(exist_ok=True)
    save(destination/"protocol.json",dict(plan=sha(PLAN),runner=sha(Path(__file__)),original=sha(out/"protocol.json"),
         first_results=sha(out/"local_five_fold_results.json"),new_base_fits=0,cap_gate_fits=10,model_api_allowed=0))
    for n,current in enumerate(folds):
        neighbor=(n-1)%5
        other=folds[neighbor]
        target_probs,more_probs=index_scores(out,n),index_scores(out,neighbor)
        test=current["outer"]
        confirm=other["confirm"]
        small=np.setdiff1d(np.r_[current["learn"],current["confirm"]],confirm)
        extra=other["outer"]
        total=np.r_[small,extra]
        base_groups=[set(groups[s["base"]]) for s in (current,other)]
        held=set(groups[np.r_[test,confirm]])
        require(all(held.isdisjoint(g) for g in base_groups) and
                base_groups[0].isdisjoint(groups[small]) and base_groups[1].isdisjoint(groups[extra]) and
                set(groups[small]).isdisjoint(groups[extra]) and held.isdisjoint(groups[total]) and
                set(groups[test]).isdisjoint(groups[confirm]),"A feedback source or group crossed an evaluation boundary")
        folder=destination/f"fold_{n}"
        folder.mkdir(exist_ok=True)
        # Project labels separately. No fitting or decision function receives outer labels.
        def observe(ids, scores):
            return inputs(np.stack([scores[i] for i in ids]),[task.X_train[i] for i in ids])
        conf=observe(confirm,target_probs)
        learning_probs={i:target_probs[i] for i in small}
        learning_probs.update({i:more_probs[i] for i in extra})
        for tag,ids in (("small",small),("pooled",total)):
            data=observe(ids,learning_probs)
            policy=learned(folder/tag,"soft",data,task.y_train[ids])
            decision=select({"soft":policy},data,task.y_train[ids],conf,task.y_train[confirm])
            save(folder/(tag+".json"),dict(policy=policy,selection=decision,learn_rows=len(ids),
                 learn_conflicts=int(data["conflict"].sum()),learn_ids=fingerprint(ids.tolist()),confirm_rows=len(confirm),
                 confirm_conflicts=int(conf["conflict"].sum()),confirm_ids=fingerprint(confirm.tolist()),
                 target_native_fold=n,additional_feedback_fold=neighbor if tag=="pooled" else None,
                 overlap_with_outer_groups=0))
        digests=[]
        for i in range(2):
            path=folder/f"predictions_{i}.csv"
            if not path.exists():
                write(path,route_outputs(folder,target_probs,task.X_train,test))
            digests.append(sha(path))
        require(digests[0]==digests[1],"Inference changed")
        save(folder/"sealed.json",dict(predictions=digests,small=sha(folder/"small.json"),pooled=sha(folder/"pooled.json")))
    save(destination/"all_sealed.json",[sha(destination/f"fold_{n}/sealed.json") for n in range(5)])


def score(out):
    folder=out/"feedback_volume"
    require(read(folder/"all_sealed.json")==[sha(folder/f"fold_{n}/sealed.json") for n in range(5)],"Unsealed comparison")
    assessed,all_cols,all_y,all_groups=[],{},[],[]
    protocol=read(out/"protocol.json")
    for n in range(5):
        base=folder/f"fold_{n}"
        seals=read(base/"sealed.json")
        target=out/"scoring_only"/f"fold_{n}.json"
        require(seals["predictions"]==[sha(base/f"predictions_{i}.csv") for i in range(2)] and
                sha(target)==protocol["splits"][n]["targets"],"Changed sealed row set")
        truth=read(target)
        cols=load_export(base/"predictions_0.csv")
        require(cols["row_id"].tolist()==truth["ids"] and not set(all_groups).intersection(truth["groups"]),
                "Training audit rows or group boundaries disagree")
        perf=score_panel(cols,truth["labels"],truth["groups"])
        choices={k:read(base/(k+".json"))["selection"]["chosen"] for k in ("small","pooled")}
        assessed.append({"fold":n,"decisions":choices,**perf})
        for k,v in cols.items():
            all_cols.setdefault(k,[]).append(v)
        all_groups.extend(truth["groups"]);all_y.extend(truth["labels"])
    cols={k:np.concatenate(v) for k,v in all_cols.items()}
    total=score_panel(cols,all_y,all_groups)
    positives=sum(r["methods"]["pooled"]["net"]>0 for r in assessed)
    effect=comparison(cols["small"],cols["pooled"],np.asarray(all_y),all_groups)
    used=total["methods"]["pooled"]
    result={"folds":assessed,"pooled":total,"positive_folds":positives,"extra_feedback_effect":effect,
            "passes":bool(positives>=4 and used["net"]>=5 and used["paired_95_fraction"][0]>0 and effect["net"]>0),
            "new_base_fits":0,"new_gate_fits":10,"model_api_calls":0,"course_validation_used":False,"course_test_used":False}
    save(folder/"result.json",result)
    print("Feedback volume:",[(r["decisions"],r["methods"]["pooled"]["net"]) for r in assessed],used,effect,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","score"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--packet",type=Path)
    a=p.parse_args()
    freeze(a.out,a.packet) if a.stage=="freeze" else score(a.out)
