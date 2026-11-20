"""Exercise the identical public agent on locally bundled numeric and image benchmarks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from superagent.pretrained_choice import digest
from tools.frontier_study import expect
from tools.pretrained_study import read, write
from tools.soft_vote_diagnostic import freeze_file
from tools.submission_smoke import archive_array, check_rows, command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--freeze",action="store_true")
    args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    wine=load_wine()
    digits=load_digits()
    cases={"a":("tabular","macro_f1",wine.data,wine.target),"b":("image","accuracy",digits.images,digits.target)}
    protocols={}
    for key,(modality,metric,x,y) in cases.items():
        flat=x.reshape(len(x),-1)
        _,groups=np.unique(flat,axis=0,return_inverse=True)
        splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=47417)
        splits=[]
        for train,held in splitter.split(flat,y,groups):
            expect(set(groups[train]).isdisjoint(groups[held]),"Overlapping duplicate groups")
            splits.append(dict(train=train.tolist(),held=held.tolist()))
        protocols[key]=dict(rows=len(y),shape=list(x.shape[1:]),modality=modality,metric=metric,indices=splits,
                            features_sha256=hashlib.sha256(x.tobytes()).hexdigest(),labels_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    src=Path(__file__).resolve()
    freeze_file(root/"protocol.json",dict(cases=protocols,seed=47417,policy="auto",source_sha256=digest(src),
                runtime={f.name:digest(f) for f in sorted((src.parents[1]/"superagent").glob("*.py"))},
                purpose="Functional transfer on real public datasets with one named policy. No tuning, labels in prediction packages, paid inference or course data."))
    if args.freeze:return
    for key,(modality,metric,x,y) in cases.items():
        for i,split in enumerate(protocols[key]["indices"]):
            folder=root/key/f"fold_{i}"
            folder.mkdir(parents=True,exist_ok=True)
            package=folder/"anonymous.zip"
            if not package.exists():
                manifest=dict(schema_version=1,modality=modality,metric=metric,
                              splits={k:dict(format="npz",path=k+".npz") for k in ("train","test")})
                with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED) as out:
                    out.writestr("manifest.json",json.dumps(manifest))
                    out.writestr("train.npz",archive_array(x[split["train"]],y[split["train"]]))
                    out.writestr("test.npz",archive_array(x[split["held"]]))
            run=folder/"run"
            if not (run/"run_outcome.json").exists():
                command(folder/"train.log","run",package,"--run-dir",run,"--planner","local","--method-choice","auto",
                        "--trials",7,"--portfolio","expanded","--feature-cache","run","--trial-timeout",480,"--time-limit",1600)
            outcome=read(run/"run_outcome.json")
            expect(outcome["requested_planner_completed"] and outcome["new_api_requests"]==0 and not outcome["formal_test_accessed"]
                   and not outcome["provided_validation_available"] and outcome["primary_metric"]==metric,"Invalid training outcome")
            print("Frozen",key,i+1,outcome["selected_candidate"],flush=True)
    report={}
    for key,(_,metric,x,y) in cases.items():
        pooled=y.copy()
        seen=set()
        folds=[]
        for i,split in enumerate(protocols[key]["indices"]):
            folder=root/key/f"fold_{i}";run=folder/"run"
            output=folder/"predictions.csv"
            command(folder/"predict.log","predict",folder/"anonymous.zip","--run-dir",run,"--output",output)
            manifest=read(folder/"prediction_manifest.json")
            expect(not manifest["test_labels_read"] and manifest["prediction_sha256"]==digest(output),"Invalid predictions")
            with output.open() as f:rows=list(csv.DictReader(f))
            ids=split["held"]
            check_rows(rows,set(map(str,np.unique(y))),len(ids))
            expect(seen.isdisjoint(ids),"Repeated evaluation row")
            seen.update(ids)
            pred=np.asarray([r["prediction"] for r in rows],dtype=y.dtype)
            pooled[ids]=pred
            summary=read(run/"summary.json")
            folds.append(dict(rows=len(ids),correct=int(np.sum(pred==y[ids])),winner=summary["selected_candidate"],
                              fits=summary["classifier_fit_accounting"]["completed_classifier_fits"],api_calls=summary["llm_calls"]))
        expect(len(seen)==len(y),"Missing evaluation rows")
        report[key]=dict(rows=len(y),correct=int(np.sum(pooled==y)),accuracy=float(np.mean(pooled==y)),
                         macro_f1=float(f1_score(y,pooled,average="macro")),metric=metric,folds=folds)
    write(root/"results.json",dict(cases=report,archive_mapping={"a":"scikit-learn wine","b":"scikit-learn digits"},
                                    protocol_sha256=digest(root/"protocol.json")))
    print(report,flush=True)


if __name__=="__main__":main()
