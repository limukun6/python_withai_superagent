"""Refit the remaining four searched candidates to bound the value of shortlist selection."""
import argparse
import os
from pathlib import Path
import shutil

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.data import load_training_task
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.frontier_study import expect, receipt
from tools.finalist_headroom import feature_weights, predict_finalist, scoring_cache
from superagent.minilm_features import feature_usage
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("archive","reference","finalists","output"):
        parser.add_argument("--"+arg,required=True,type=Path)
    parser.add_argument("--arm",required=True)
    parser.add_argument("--minilm-weights",type=Path)
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    os.environ["SUPERAGENT_OFFLINE"]="1"
    os.environ.pop("SUPERAGENT_PRETRAINED_RUNTIME",None)
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    reference=read(args.reference/"protocol.json")
    parent=read(args.finalists/"protocol.json")
    expect(digest(args.archive)==reference["archive_sha256"] and parent["arm"]==args.arm and
           parent["reference_sha256"]==digest(args.reference/"protocol.json"),"Mismatched finalist reference")
    records=[]
    for i,block in enumerate(reference["locations"]):
        source=Path(block[args.arm])
        prior=args.finalists/f"fold_{i}"
        rec=receipt(source)
        kept=read(prior/"decision.json")["components"]
        history=read(source/"trials.json")
        trials={r["candidate"]:r["trial"] for r in history}
        extra=[name for name in trials if name not in kept]
        expect(len(extra)==4 and len(kept)==3 and kept[0]==rec["winner"],"Unexpected candidate group")
        records.append(dict(source=str(source),shortlist=kept,extra=extra,trials=trials,
                            original_model=rec["model_sha256"],previous_full=read(prior/"fits.json"),
                            payload_hash=digest(prior/"diagnostic_training.joblib"),history_hash=digest(source/"trials.json")))
    src=Path(__file__).resolve().parents[1]
    freeze_file(root/"protocol.json",dict(reference_sha256=digest(args.reference/"protocol.json"),
                finalist_protocol_sha256=digest(args.finalists/"protocol.json"),records=records,arm=args.arm,
                scoring_feature_weights=feature_weights(args.minilm_weights),
                source={str(p.relative_to(src)):digest(p) for p in [Path(__file__),*sorted((src/'superagent').glob('*.py'))]},
                rule="Refit all four candidates outside the original shortlist on exactly the same fitting rows. The best whole model per fold after opening its outer labels is a retrospective upper bound. Ties by candidate name. No production policy may use this oracle."))
    if args.freeze:return
    opened=[]
    with train_only(args.archive,opened):task=load_training_task(args.archive)
    for i,rec in enumerate(records):
        dest=root/f"fold_{i}"
        dest.mkdir(exist_ok=True)
        source=Path(rec["source"])
        old=args.finalists/f"fold_{i}"
        payload=old/"diagnostic_training.joblib"
        expect(digest(payload)==rec["payload_hash"],"Training payload changed")
        for path in [*old.glob("method_*.json"),*old.glob("*vectors.sqlite"),old/"pretrained_runtime.json"]:
            if path.is_file() and not (dest/path.name).exists():shutil.copyfile(path,dest/path.name)
        for name in rec["extra"]:
            n=rec["trials"][name]
            target=dest/f"extra_{n}.joblib"
            proof=dest/f"extra_{n}.json"
            if proof.exists():
                expect(digest(target)==read(proof)["sha256"],"Prior diagnostic refit changed")
                continue
            model,metrics=train_candidate(payload,name,target,480)
            expect(model is not None and metrics["classifier_fits"]==1,str(metrics))
            write(proof,dict(candidate=name,sha256=digest(target)))
        expect(not (dest/"downstream_usage.jsonl").exists() and not (dest/"llm_usage.jsonl").exists(),
               "An additional forward was recorded")
        print("Frozen all seven full-data models on fold",i+1,flush=True)
    folds=[]
    for i,rec in enumerate(records):
        dest=root/f"fold_{i}"
        old=args.finalists/f"fold_{i}"
        source=Path(rec["source"])
        ids=reference["indices"][i]["outer"]
        x=take(task.X_train,ids)
        y=task.y_train[ids]
        winner,*other=rec["shortlist"]
        models={winner:source/"selected_model.joblib"}
        for j,name in enumerate(other,1):
            path=old/f"full_{j}.joblib"
            expect(digest(path)==rec["previous_full"][name],"Shortlisted model changed")
            models[name]=path
        models.update({n:dest/f"extra_{rec['trials'][n]}.joblib" for n in rec["extra"]})
        cache=(scoring_cache(dest/'diagnostic_prediction_features',[old/'diagnostic_prediction_features',source.parent/(args.arm+'_prediction_features')])
               if args.minilm_weights else None)
        preds={n:predict_finalist(task.kind,joblib.load(p)["model"],x,args.minilm_weights,cache) for n,p in models.items()}
        before=read(old/"outer_predictions.json")
        for n in rec["shortlist"]:np.testing.assert_array_equal(before["full_finalists"][n],preds[n])
        scores={n:int((v==y).sum()) for n,v in preds.items()}
        rank=lambda names:min(names,key=lambda n:(-scores[n],n))
        names=dict(incumbent=winner,oracle_three=rank(rec["shortlist"]),oracle_seven=rank(scores))
        write(dest/"outer_predictions.json",dict(rows=ids,truth=y.tolist(),**{k:preds[n].tolist() for k,n in names.items()}))
        folds.append(dict(rows=len(ids),names=names,all_seven_correct=scores,correct={k:scores[n] for k,n in names.items()}))
    usages=[feature_usage(root/f'fold_{i}'/'diagnostic_prediction_features') for i in range(len(folds))]
    result=dict(rows=sum(f["rows"] for f in folds),folds=folds,correct={k:sum(f["correct"][k] for f in folds) for k in folds[0]["correct"]},
                additional_diagnostic_fits=4*len(folds),api_calls=0,additional_encoder_forwards=sum(u['batches'] for u in usages),
                additional_minilm_input_tokens=sum(u['input_tokens'] for u in usages),training_members=opened)
    write(root/"results.json",result)
    print(result["correct"],flush=True)


if __name__=="__main__":main()
