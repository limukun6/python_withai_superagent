"""Independent, training-only comparison of frozen ordinary runs and the screened recipe."""
import argparse
import csv
import io
from pathlib import Path
import time
import zipfile

import joblib
import numpy as np

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.attribute_alignment import AttributeAlignmentTree
from superagent.joint_context import ContextTokenTree
from superagent.table_foundation import FrozenTableClassifier

PROTOCOL = ROOT / "research/task1_feedback_extension_1127/outer_comparison.json"
COVERAGE = PROTOCOL.with_name("coverage_addendum.json")
DIAGNOSTIC = ("diagnostic_anchor", "diagnostic_screened")


def arrays(path, members):
    with zipfile.ZipFile(path) as archive:
        return [np.load(io.BytesIO(archive.read(member)), allow_pickle=False) for member in members]


def diagnostic(out):
    info = read(out / "inputs.json")
    training = out / "anonymous/training.zip"
    query = out / "anonymous/queries.zip"
    require(sha(training) == info["entry_inputs"] and sha(query) == info["entry_queries"], "Changed packets")
    elected = read(out / "election/results.json")
    require(elected["advance"] and elected["selected"], "No preregistered recipe graduated")
    folder = out / "diagnostics"
    folder.mkdir(exist_ok=False)
    save(folder / "launch.json", dict(protocol=sha(PROTOCOL), election=sha(out/"election/results.json"),
                                     training=sha(training), query=sha(query), label_access=["y_train.npy"]))
    x, y = arrays(training, ["C_train.npy", "y_train.npy"])
    query_rows, = arrays(query, ["C_test.npy"])
    classes = np.unique(y)
    estimators = {"anchor": ContextTokenTree(dict(pair_limit=3)),
                  "attribute_alignment": AttributeAlignmentTree(dict(pair_limit=3)),
                  "pretrained_icl": FrozenTableClassifier("categorical_tabular")}
    report, scores = {}, {}
    for name in ("anchor", *elected["selected"]):
        start = time.monotonic()
        model = estimators[name].fit(x, y)
        order = [list(model.classes_).index(label) for label in classes]
        probabilities = np.asarray(model.predict_proba(query_rows), dtype=float)[:,order]
        require(probabilities.shape == (len(query_rows),len(classes)) and np.isfinite(probabilities).all()
                and np.allclose(probabilities.sum(1),1,atol=1e-5), "Invalid class probability matrix")
        scores[name] = probabilities
        path = folder / (name+".npy")
        np.save(path,probabilities,allow_pickle=False)
        report[name] = dict(fits=1, seconds=round(time.monotonic()-start,2), probabilities=sha(path),
                            details=getattr(model,"training_metadata",getattr(model,"audit_",{})))
        save(folder / (name+".json"),report[name])
        print(name, report[name]["seconds"], flush=True)
    share = elected["share"]
    pooled = (1-share)*scores["anchor"] + share*np.mean([scores[m] for m in elected["selected"]],axis=0)
    seals = {}
    for name, matrix in zip(DIAGNOSTIC,[scores["anchor"],pooled],strict=True):
        path=folder/(name+".npy")
        np.save(path,classes[matrix.argmax(axis=1)],allow_pickle=False)
        seals[name] = sha(path)
    save(folder/"sealed.json",dict(launch=sha(folder/"launch.json"),rows=len(query_rows),fits=len(scores),
         methods=report,predictions=seals,model_api_requests=0,course_validation_opened=False,official_test_opened=False))


def prediction_csv(path, count):
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    require([item["row_id"] for item in rows]==list(map(str,range(count))),"Prediction order mismatch")
    return np.array([item["prediction"] for item in rows])


def compare(a,b,groups):
    _, inverse=np.unique(groups,return_inverse=True)
    n=int(inverse.max())+1
    sizes=np.bincount(inverse,minlength=n)
    delta=a.astype(int)-b.astype(int)
    diffs=np.bincount(inverse,weights=delta,minlength=n)
    rng=np.random.default_rng(672391)
    replicates=[]
    for _ in range(2000):
        ix=rng.integers(n,size=n)
        replicates.append(diffs[ix].sum()/sizes[ix].sum())
    return dict(fixed=int((a & ~b).sum()), lost=int((~a & b).sum()), net=int(delta.sum()),
                gain=float(delta.mean()), group_bootstrap_95=np.quantile(replicates,[.025,.975]).tolist())


def grade(out):
    info=read(out/"inputs.json")
    size=info["roles"]["external"]["rows"]
    names=[*read(PROTOCOL)["ordinary"],"coverage32"]
    predictions, seals, methods = {}, {}, {}
    for name in names:
        folder=out/name
        receipt=read(folder/"sealed.json")
        require(receipt["independent_reload"] and receipt["rows"]==size and receipt["status"]=="completed"
                and receipt["prediction"]==sha(folder/"solve.csv")==sha(folder/"reload.csv"),"Missing completed entry")
        predictions[name]=prediction_csv(folder/"solve.csv",size)
        methods[name]=dict(members=receipt["members"],fits=receipt["fits"],seconds=round(receipt["seconds"],2))
        seals[name]=sha(folder/"sealed.json")
    folder=out/"diagnostics"
    receipt=read(folder/"sealed.json")
    require(receipt["rows"]==size,"Diagnostic size mismatch")
    for name in DIAGNOSTIC:
        path=folder/(name+".npy")
        require(sha(path)==receipt["predictions"][name],"Diagnostic changed")
        predictions[name]=np.load(path,allow_pickle=False).astype(str)
    seals["diagnostics"]=sha(folder/"sealed.json")
    require(sha(out/"outer_answers.joblib")==info["outer_answers"],"Unexpected outer answers")
    save(out/"outer_score_lock.json",dict(protocol=sha(PROTOCOL),coverage=sha(COVERAGE),seals=seals,answers=info["outer_answers"]))
    answer=joblib.load(out/"outer_answers.joblib")
    correct={n:y==np.asarray(answer["y"]).astype(str).reshape(-1) for n,y in predictions.items()}
    paired={}
    for name,ref in [("original32","original10"),("expanded32","original32"),("expanded32","original10"),
                     ("diagnostic_screened","diagnostic_anchor"),("coverage32","original10"),
                     ("coverage32","original32"),("coverage32","expanded32")]:
        paired[name+"_vs_"+ref]=compare(correct[name],correct[ref],answer["groups"])
    expanded_ok = paired["expanded32_vs_original32"]["net"]>0 and paired["expanded32_vs_original10"]["net"]>=0
    coverage_ok = (paired["coverage32_vs_original32"]["net"]>0 and paired["coverage32_vs_original10"]["net"]>=0
                   and paired["coverage32_vs_expanded32"]["net"]>0)
    release = "coverage32" if coverage_ok else "expanded32" if expanded_ok else None
    result=dict(lock=sha(out/"outer_score_lock.json"),rows=size,groups=len(np.unique(answer["groups"])),ordinary=methods,
                scores={n:dict(correct=int(hits.sum()),accuracy=float(hits.mean())) for n,hits in correct.items()},
                comparisons=paired,advance=release,model_api_requests=0,
                designated_validation_opened=False,official_test_opened=False)
    save(out/"outer_results.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["diagnostic","grade"])
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    (diagnostic if args.action=="diagnostic" else grade)(args.out)
