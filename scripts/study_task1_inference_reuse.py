"""Make alternative sealed predictions from already fitted components and training evidence."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_outer import arrays, prediction_csv
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.score_evidence import checked_score_record

PLAN=ROOT/"research/task1_feedback_extension_1127/inference_reuse.json"


def combine(parts,w):
    return parts[0] if len(parts)==1 else w*parts[0]+(1-w)*parts[1]


def fit_weight(folds,metric,classes):
    if len(folds)<3 or len(folds[0][1])!=2:
        return .5,[]
    grid=np.arange(9)/8
    scores=[]
    for y,parts in folds:
        values=[]
        for w in grid:
            p=combine(parts,w)
            values.append(metric.evaluate(y,classes[p.argmax(1)],p,classes)["score"])
        scores.append(values)
    sizes=np.array([len(y) for y,_ in folds])
    chosen=[]
    gains=[]
    for i in range(len(folds)):
        mask=np.arange(len(folds))!=i
        train=np.average(np.array(scores)[mask],axis=0,weights=sizes[mask])
        ix=min(range(len(grid)),key=lambda j:(-train[j],abs(grid[j]-.5),grid[j]))
        chosen.append(float(grid[ix]))
        gains.append(scores[i][ix]-scores[i][4])
    valid=sum(d>0 for d in gains)>=2 and np.average(gains,weights=sizes)>=.003
    return (float(np.median(chosen)) if valid else .5),[dict(weight=w,gain=g) for w,g in zip(chosen,gains)]


def run(fold):
    src=fold/"regular"
    folder=fold/"reuse"
    folder.mkdir(exist_ok=False)
    native=src/"run/agent"
    seal=read(src/"sealed.json")
    members=seal["members"]
    require(1<=len(members)<=2 and seal["independent_reload"],"Expected a sealed plain selection")
    catalog=read(fold/"inputs.json")
    q=fold/"anonymous/queries.zip"
    require(sha(q)==catalog["entry_queries"],"Unbound query inputs")
    query,=arrays(q,["C_test.npy"])
    history=read(native/"history.json")
    files={(r["candidate"],r["split_id"]):native/("trial_"+str(r["trial"])+".joblib")
           for r in history if r["status"]=="completed"}
    folds=sorted(f for f in ("primary","confirmation_1","confirmation_2") if all((m,f) in files for m in members))
    metric=MetricSpec.from_payload(joblib.load(native/(folds[0]+".joblib")))
    classes=np.asarray(metric.labels)
    def predict(model):
        order,p=probability_with_model("categorical_tabular",model,query)
        return np.asarray(p,dtype=float)[:,[list(order).index(c) for c in classes]]
    fitting=[]
    replicas=[[] for _ in members]
    for f in folds:
        payload=native/(f+".joblib")
        y=joblib.load(payload)["y_search"]
        parts=[]
        for i,m in enumerate(members):
            bundle=joblib.load(files[m,f])
            record=bundle["prediction_record"]
            require(record["evaluation_payload_sha256"]==sha(payload),"Fold row identity mismatch")
            p,order=checked_score_record(record["score_record"],record,metric,m,y)
            parts.append(p[:,[list(order).index(c) for c in classes]])
            replicas[i].append(predict(bundle["model"]))
            del bundle
        fitting.append((y,parts))
    w,checks=fit_weight(fitting,metric,classes)
    save(folder/"decision.json",dict(protocol=sha(PLAN),source=sha(src/"sealed.json"),members=members,
             weight_first=w if len(members)==2 else 1.,leave_one_fold_out=checks,folds=folds))
    full=[predict(joblib.load(native/("refit_component_"+str(i)+".joblib"))["model"]) for i in range(len(members))]
    original=classes[combine(full,.5).argmax(1)].astype(str)
    require(np.array_equal(original,prediction_csv(src/"solve.csv",len(query))),"Reconstructed regular predictions differ")
    weighted=combine(full,w)
    mixed=.75*weighted+.25*combine([np.mean(p,axis=0) for p in replicas],w)
    receipts={}
    for name,probabilities in (("weighted",weighted),("replicas",mixed)):
        path=folder/(name+".npy")
        np.save(path,classes[probabilities.argmax(1)],allow_pickle=False)
        receipts[name]=sha(path)
    save(folder/"sealed.json",dict(decision=sha(folder/"decision.json"),rows=len(query),predictions=receipts,
         training_fits=0,model_api_requests=0,old_prediction_rebuilt=True))
    print(fold.name,members,w,checks,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold",required=True,type=lambda s:Path(s).resolve())
    run(parser.parse_args().fold)
