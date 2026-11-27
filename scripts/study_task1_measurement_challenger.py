"""Re-evaluate the program's locked direct-anchor nominee at larger fitting sizes."""
import argparse
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.study_task1_measurements import MeasurementTree
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_remaining_folds import FOLDS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.joint_context import ContextTokenTree

PLAN=ROOT/"research/task1_feedback_extension_1127/measurement_challenger.json"


def solve(out):
    election=out/"incumbent_comparison/result.json"
    result=read(election)
    name,share=result["all_development_nomination"]
    factory={"measure_relative":lambda:MeasurementTree(dict(relative=True,pair_limit=3)),
             "measure_measured":lambda:MeasurementTree(dict(relative=False,pair_limit=3))}
    require(name in factory and 0<share<=1,"The nominated measurement route is unavailable")
    base=out/"remaining_folds"
    require(not (base/"scoring_lock.json").exists(),"The comparison's answer scoring already started")
    for i in FOLDS:
        root=base/str(i)
        info=read(root/"inputs.json")
        training=root/"anonymous/training.zip"
        query=root/"anonymous/queries.zip"
        require(sha(training)==info["entry_inputs"] and sha(query)==info["entry_queries"],"Changed task packets")
        dest=root/"measurement_nominee"
        dest.mkdir(exist_ok=False)
        save(dest/"launch.json",dict(protocol=sha(PLAN),selection=sha(election),nominee=name,share=share,
                                    training_sha256=sha(training),query_sha256=sha(query)))
        x,y=arrays(training,["C_train.npy","y_train.npy"])
        q,=arrays(query,["C_test.npy"])
        start=time.monotonic()
        incumbent=ContextTokenTree(dict(pair_limit=3)).fit(x,y)
        challenger=factory[name]().fit(x,y)
        require(np.array_equal(incumbent.classes_,challenger.classes_),"Changed class order")
        p=np.asarray(incumbent.predict_proba(q),dtype=float)
        r=(1-share)*p+share*np.asarray(challenger.predict_proba(q),dtype=float)
        hashes={}
        for tag,prob in (("original_single",p),("nominee",r)):
            require(prob.shape==(len(q),len(incumbent.classes_)) and np.isfinite(prob).all()
                    and np.allclose(prob.sum(1),1.),"Invalid probabilities")
            path=dest/(tag+".npy")
            np.save(path,prob,allow_pickle=False)
            hashes[tag]=sha(path)
        save(dest/"sealed.json",dict(launch=sha(dest/"launch.json"),classes=incumbent.classes_.tolist(),
              predictions=hashes,training_rows=len(x),evaluation_rows=len(q),classifier_fits=2,
              seconds=round(time.monotonic()-start,2),api_requests=0))
        print(i,len(x),len(q),round(time.monotonic()-start,2),flush=True)
    save(out/"measurement_all_sealed.json",dict(protocol=sha(PLAN),folds={str(i):sha(base/str(i)/"measurement_nominee/sealed.json") for i in FOLDS}))


def grade(out):
    base=out/"remaining_folds"
    locked=read(out/"measurement_all_sealed.json")
    ordinary=read(base/"result.json")
    records=[]
    hits={n:[] for n in ("original_single","partner","regular","nominee")}
    groups=[]
    for i in FOLDS:
        root=base/str(i)
        side=root/"measurement_nominee"
        record=read(side/"sealed.json")
        require(sha(side/"sealed.json")==locked["folds"][str(i)],"Changed frozen proposal")
        classes=np.array(record["classes"]).astype(str)
        labels=root/"outer_answers.joblib"
        require(sha(labels)==read(root/"inputs.json")["outer_answers"],"Changed evaluation rows")
        key=joblib.load(labels)
        y=key["y"].astype(str)
        current={}
        for name in hits:
            if name in ("original_single","nominee"):
                f=side/(name+".npy")
                require(sha(f)==record["predictions"][name],"Unsealed probabilities")
                result=classes[np.load(f,allow_pickle=False).argmax(1)]
            else:
                f=root/name/"solve.csv"
                require(sha(f)==read(root/name/"sealed.json")["prediction"],"Changed ordinary prediction")
                result=prediction_csv(f,len(y))
            current[name]=result==y
            hits[name].extend(current[name])
        groups.extend(key["groups"])
        records.append(dict(fold=i,rows=len(y),correct={n:int(v.sum()) for n,v in current.items()},
                     delta={n:int(current["nominee"].sum()-current[n].sum()) for n in hits if n!="nominee"}))
    differences={n:compare(np.asarray(hits["nominee"]),np.asarray(v),groups) for n,v in hits.items() if n!="nominee"}
    passes=all(d["gain"]>=.003 and sum(r["delta"][n]>0 for r in records)>=2
               and min(r["delta"][n]/r["rows"] for r in records)>=-.003 for n,d in differences.items())
    result=dict(protocol=sha(PLAN),sealed=sha(out/"measurement_all_sealed.json"),ordinary_lock=ordinary["lock"],
                rows=len(groups),folds=records,correct={n:int(np.sum(v)) for n,v in hits.items()},
                comparisons=differences,advance_to_ordinary=bool(passes),classifier_fits=6,api_requests=0)
    save(out/"measurement_nominee_result.json",result)
    print(result["correct"], result["comparisons"], passes,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("solve","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    (solve if a.action=="solve" else grade)(a.out)
