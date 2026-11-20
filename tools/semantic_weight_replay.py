"""Calibrate a frozen hybrid classifier's semantic contribution using only its inner search split."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import log_loss

from superagent.data import load_training_task
from superagent.method_choice import TEXT_MINILM
from superagent.minilm_features import MiniLMTextClassifier, feature_usage
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


WEIGHTS=(.5,1.,2.)


def probabilities(model,inputs):
    expect(isinstance(model,MiniLMTextClassifier) and not model.dense_only,"Expected a hybrid classifier")
    x=model.features(inputs)
    linear=model.classifier
    expect(x.shape[1]>384 and len(linear.classes_)>=2,"Missing fitted lexical or semantic coefficients")
    base=np.asarray(linear.decision_function(x)).reshape(len(inputs),-1)
    semantic=np.asarray(x[:,-384:]@linear.coef_[:,-384:].T)
    values={}
    for weight in WEIGHTS:
        logits=base+(weight-1)*semantic
        if len(linear.classes_)==2:
            pos=expit(logits[:,0])
            scores=np.column_stack((1-pos,pos))
        else:
            scores=softmax(logits,axis=1)
        values[weight]=scores
    np.testing.assert_allclose(values[1.],linear.predict_proba(x),atol=1e-10)
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("reference","archive","output"):p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--arm",required=True)
    p.add_argument("--freeze",action="store_true")
    args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    ref=read(args.reference/"protocol.json")
    expect(ref["archive_sha256"]==digest(args.archive),"Archive changed")
    records=[]
    for block in ref["locations"]:
        folder=Path(block[args.arm])
        r=receipt(folder)
        trials=read(folder/"trials.json")
        matching=[t for t in trials if t["candidate"]==TEXT_MINILM]
        primary=folder/f"trial_{matching[0]['trial']}.joblib" if matching else None
        records.append(dict(source=str(folder),final=r["winner"],final_hash=r["model_sha256"],
                            primary=str(primary) if primary else None,primary_hash=digest(primary) if primary else None))
    freeze_file(root/"protocol.json",dict(reference_sha256=digest(args.reference/"protocol.json"),records=records,
                arm=args.arm,weights=list(WEIGHTS),source_sha256=digest(Path(__file__)),
                rule="When the confirmed winner uses hybrid sentence features, compare half, original and double semantic logit contribution on its saved primary search split. Change only if log loss improves by 0.002 while accuracy falls by no more than 0.2 percentage points. Apply that scalar to the full refit. Also report both fixed scalars. Zero new fits."))
    if args.freeze:return
    opened=[]
    with train_only(args.archive,opened):task=load_training_task(args.archive)
    decisions=[]
    for i,rec in enumerate(records):
        source=Path(rec["source"])
        chosen=1.
        scores={}
        if rec["final"]==TEXT_MINILM:
            expect(rec["primary"] is not None,"Missing initial candidate")
            model=joblib.load(rec["primary"])["model"]
            data=joblib.load(source/"inner_data.joblib")
            before=feature_usage(source)
            probabilities_by_weight=probabilities(model,data["X_search"])
            expect(feature_usage(source)==before,"Additional training-side forward")
            y=data["y_search"]
            for w,proba in probabilities_by_weight.items():
                pred=model.classes_[proba.argmax(axis=1)]
                scores[w]=dict(loss=float(log_loss(y,proba,labels=model.classes_)),accuracy=float(np.mean(pred==y)))
            options=[w for w in WEIGHTS if scores[w]["accuracy"]>=scores[1.]["accuracy"]-.002]
            best=min(options,key=lambda w:(scores[w]["loss"],abs(w-1)))
            if scores[1.]["loss"]-scores[best]["loss"]>=.002:chosen=best
        dest=root/f"fold_{i}";dest.mkdir(exist_ok=True)
        freeze_file(dest/"decision.json",dict(chosen=chosen,search={str(w):s for w,s in scores.items()}))
        decisions.append(chosen)
    folds=[]
    for i,(rec,chosen) in enumerate(zip(records,decisions,strict=True)):
        source=Path(rec["source"])
        ids=ref["indices"][i]["outer"]
        x=take(task.X_train,ids)
        y=task.y_train[ids]
        model=joblib.load(source/"selected_model.joblib")["model"]
        if rec["final"]==TEXT_MINILM:
            cache=source.parent/(args.arm+"_prediction_features")
            before=feature_usage(cache)
            model.bind_prediction_weights(read(cache/"pretrained_runtime.json")["minilm_weights"],cache)
            scores=probabilities(model,x)
            pred={w:model.classes_[s.argmax(axis=1)] for w,s in scores.items()}
            expect(feature_usage(cache)==before,"Additional evaluation-side forward")
        else:
            pred=dict.fromkeys(WEIGHTS,predict_with_model(task.kind,model,x))
        recorded=read(args.reference/f"fold_{ref['fold_numbers'][i]-1}"/"outer_predictions.json")
        np.testing.assert_array_equal(pred[1.],recorded[args.arm])
        expect(recorded["rows"]==ids,"Prediction order changed")
        vectors=dict(incumbent=pred[1.],guarded_weight=pred[chosen],fixed_half=pred[.5],fixed_double=pred[2.])
        write(root/f"fold_{i}"/"outer_predictions.json",dict(rows=ids,truth=y.tolist(),**{k:v.tolist() for k,v in vectors.items()}))
        folds.append(dict(rows=len(ids),chosen=chosen,correct={k:int((v==y).sum()) for k,v in vectors.items()}))
    result=dict(folds=folds,rows=sum(f["rows"] for f in folds),correct={k:sum(f["correct"][k] for f in folds) for k in folds[0]["correct"]},
                new_classifier_fits=0,new_api_calls=0,additional_encoder_forwards=0,training_members=opened)
    write(root/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":main()
