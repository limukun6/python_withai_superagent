"""Test whether paired confirmation evidence justifies replacing the first-five incumbent."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from superagent.pretrained_choice import digest
from superagent.splits import effective_grouping, sample_groups
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, write
from tools.soft_vote_diagnostic import freeze_file


def choose(folder):
    history=read(folder/"trials.json")
    confirmation=read(folder/"confirmation.json")
    rec=receipt(folder)
    incumbent,winner=rec["incumbent"],rec["winner"]
    expect(confirmation["primary_metric"]=="accuracy","This rule is specified for correctness feedback")
    if incumbent==winner:
        return dict(selected=winner,previous=incumbent,mean_winner=winner,reason="Both rules already select the same model")
    numbers={r["candidate"]:r["trial"] for r in history}
    full=joblib.load(folder/"refit_data.joblib")
    task=SimpleNamespace(kind=full["kind"],X_train=full["X_train"])
    groups=sample_groups(task,effective_grouping(task.kind,"shared_text"))
    summed=np.zeros(len(groups))
    visits=np.zeros(len(groups),dtype=int)
    wins=0
    for i in range(3):
        split=read(folder/("split.json" if i==0 else f"confirmation_split_{i}.json"))
        ids=np.asarray(split["search_indices"])
        payload=joblib.load(folder/("inner_data.joblib" if i==0 else f"confirmation_data_{i}.joblib"))
        y=payload["y_search"]
        np.testing.assert_array_equal(y,full["y_train"][ids])
        def correctness(name):
            trial=numbers[name]
            record=joblib.load(folder/(f"trial_{trial}.joblib" if i==0 else f"confirmation_{i}_{trial}.joblib"))
            return np.asarray(record["search_predictions"]==y,dtype=int)
        difference=correctness(winner)-correctness(incumbent)
        wins+=int(difference.sum()>0)
        summed[ids]+=difference
        visits[ids]+=1
    seen=visits>0
    d=summed[seen]/visits[seen]
    mean=float(d.mean())
    _,cluster=np.unique(groups[seen],return_inverse=True)
    sizes=np.bincount(cluster)
    deviations=np.bincount(cluster,weights=d)-mean*sizes
    se=float(np.sqrt(len(sizes)/(len(sizes)-1)*np.sum(deviations**2))/len(d))
    replace=wins>=2 and mean>=max(.002,se)
    return dict(selected=winner if replace else incumbent,previous=incumbent,mean_winner=winner,
                rows=len(d),groups=len(sizes),wins=wins,mean_row_gain=mean,group_standard_error=se)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("reference","finalists","output"):p.add_argument("--"+flag,required=True,type=Path)
    p.add_argument("--arm",required=True)
    p.add_argument("--freeze",action="store_true")
    args=p.parse_args()
    reference=read(args.reference/"protocol.json")
    parent=read(args.finalists/"protocol.json")
    src=Path(__file__).resolve()
    expect(parent["arm"]==args.arm and parent["reference_sha256"]==digest(args.reference/"protocol.json"),"Reference cohort changed")
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    records=[]
    for block in reference["locations"]:
        folder=Path(block[args.arm])
        paths=[folder/"confirmation.json",folder/"trials.json",folder/"refit_data.joblib",folder/"inner_data.joblib",folder/"split.json",
               *sorted(folder.glob("confirmation_data_*.joblib")),*sorted(folder.glob("confirmation_split_*.json")),
               *sorted(folder.glob("trial_*.joblib")),*sorted(folder.glob("confirmation_*_*.joblib"))]
        records.append(dict(folder=str(folder),files={f.name:digest(f) for f in paths}))
    freeze_file(root/"protocol.json",dict(reference_sha256=digest(args.reference/"protocol.json"),arm=args.arm,
                records=records,source_sha256=digest(src),finalist_protocol=digest(args.finalists/"protocol.json"),
                rule="Retain the best model from the first five searches unless the average-score winner beats it on two of three confirmation splits and gains at least 0.2 percentage points and one group-robust standard error. Average repeated correctness evidence within each row."))
    if args.freeze:return
    decisions=[]
    for i,row in enumerate(records):
        folder=root/f"fold_{i}";folder.mkdir(exist_ok=True)
        decision=choose(Path(row["folder"]))
        freeze_file(folder/"decision.json",decision)
        decisions.append(decision)
    folds=[]
    for i,decision in enumerate(decisions):
        frozen=read(args.finalists/f"fold_{i}"/"outer_predictions.json")
        held=reference["indices"][i]["outer"]
        expect(frozen["rows"]==held,"Outer rows changed")
        y=np.asarray(frozen["truth"])
        names=dict(mean=decision["mean_winner"],guarded=decision["selected"],first_five=decision["previous"])
        preds={k:frozen["full_finalists"][n] for k,n in names.items()}
        write(root/f"fold_{i}"/"outer_predictions.json",dict(rows=held,truth=y.tolist(),**preds))
        folds.append(dict(rows=len(y),correct={k:int(np.sum(v==y)) for k,v in preds.items()},
                          selected=decision["selected"],changed=decision["selected"]!=decision["mean_winner"]))
    result=dict(folds=folds,rows=sum(f["rows"] for f in folds),
                correct={k:sum(f["correct"][k] for f in folds) for k in folds[0]["correct"]},
                new_classifier_fits=0,new_api_calls=0)
    write(root/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":main()
