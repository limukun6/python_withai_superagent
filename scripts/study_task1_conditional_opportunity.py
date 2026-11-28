"""Compare two generic challengers against the actual frozen seed-policy exports."""
import argparse
import gc
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.study_task1_conditional_models import CrossedLogistic, PosteriorAlignmentTree
from scripts.study_task1_extensions import key
from scripts.study_task1_outer import arrays, compare, prediction_csv
from scripts.study_task1_seed_completion import origin
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.attribute_alignment import comparable_fields
from superagent.candidate_refinement import resolve_registered
from superagent.joint_context import blocks
from superagent.models import _clean_categories, probability_with_model
from superagent.splits import sample_groups
from superagent.training_randomness import fitted_seed, training_seed

PLAN = ROOT / "research/task1_feedback_extension_1127/conditional_attribute_opportunity.json"
STUDY = "conditional_opportunity_1128"
METHODS = {"posterior":PosteriorAlignmentTree,"crossed":CrossedLogistic}
ROLES = ("nominate","confirm")


def normalized(matrix):
    matrix = np.asarray(matrix,dtype=float)
    require(matrix.ndim==2 and matrix.shape[1]==2 and np.isfinite(matrix).all() and (matrix>=0).all()
            and np.allclose(matrix.sum(1),1.,atol=1e-5), "Invalid class probabilities")
    return matrix[:,1]/matrix.sum(axis=1)


def ordered(model, rows, classes):
    order,p = probability_with_model("categorical_tabular",model,rows)
    return normalized(p[:,[list(order).index(k) for k in classes]])


def sources(out,i):
    fold = origin(out,i)
    side = fold/"replicated"
    src = side/"run/agent"
    rc = read(side/"sealed.json")
    info = read(fold/"inputs.json")
    require(rc["independent_reload"] and rc["prediction"]==sha(side/"solve.csv")==sha(side/"reload.csv")
            and sha(fold/"anonymous/training.zip")==info["entry_inputs"] and sha(fold/"anonymous/queries.zip")==info["entry_queries"],
            "Unverified original run")
    x,y = arrays(fold/"anonymous/training.zip",["C_train.npy","y_train.npy"])
    q, = arrays(fold/"anonymous/queries.zip",["C_test.npy"])
    native = joblib.load(src/"refit_data.joblib")
    require(np.array_equal(native["X_train"],x) and np.array_equal(native["y_train"],y), "Row indices differ from frozen agent")
    return fold,src,x,y,q,rc


def panels(src,x,y):
    groups = sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    seen_rows,seen_groups=set(),set()
    for name in ("primary","confirmation_1","confirmation_2"):
        prior = read(src/(name+"_split.json"))
        held = prior["held"]
        require(not seen_rows.intersection(held) and not seen_groups.intersection(groups[held]),"Repeated original panel")
        seen_rows.update(held)
        seen_groups.update(groups[held])
    remaining = np.asarray(sorted(set(range(len(y)))-seen_rows))
    result = {}
    splitter = StratifiedGroupKFold(2,shuffle=True,random_state=87241)
    for role,(_,idx) in zip(ROLES,splitter.split(x[remaining],y[remaining],groups[remaining]),strict=True):
        held=remaining[idx]
        fit=np.setdiff1d(np.arange(len(y)),held)
        require(not set(groups[fit]).intersection(groups[held]) and not seen_groups.intersection(groups[held])
                and set(y[held])==set(y[fit]),"Leakage or missing class in new panel")
        seen_groups.update(groups[held])
        result[role]=(fit,held)
    return result


def record_matrix(area, tag, values, record):
    p = area/(tag+".npy")
    np.save(p,values,allow_pickle=False)
    save(area/(tag+".json"),dict(**record,predictions=sha(p),rows=len(values)))


def load_matrix(area,tag):
    meta=read(area/(tag+".json"))
    p=area/(tag+".npy")
    require(meta["predictions"]==sha(p),"Modified cached predictions")
    return np.load(p,allow_pickle=False)


def controls(out,i):
    fold,src,x,y,q,rc=sources(out,i)
    area=out/STUDY/str(i)/"control"
    area.mkdir(exist_ok=False,parents=True)
    classes=np.unique(y)
    registry=read(src/"candidate_registry.json")
    repeat=read(src/"seed_replication.json")
    save(area/"launch.json",dict(protocol=sha(PLAN),incumbent=sha(fold/"replicated/sealed.json"),
         registry=sha(src/"candidate_registry.json"),repeat=sha(src/"seed_replication.json"),base_fits=rc["fits"]))
    for role,(fit,held) in panels(src,x,y).items():
        components=[]
        for j,member in enumerate(rc["members"]):
            seeds=[None]
            if repeat["action"]=="adopt" and member in repeat["eligible"]:
                seeds.extend(repeat["seeds"])
            method=resolve_registered("categorical_tabular",member,registry)
            probs=[]
            for n,seed in enumerate(seeds):
                tag=f"{role}_{j}_{n}"
                save(area/(tag+"_attempt.json"),dict(candidate=member,seed=seed,fit_rows=len(fit),holdout_rows=len(held)))
                start=time.monotonic()
                with training_seed(seed):
                    model,_=method.fit_predict(x[fit],y[fit],x[held])
                require(seed is None or seed==fitted_seed(model),"Requested seed not applied")
                p=ordered(model,x[held],classes)
                record_matrix(area,tag,p,dict(fits=1,member=member,elapsed_seconds=round(time.monotonic()-start,3)))
                probs.append(p)
                del model
            components.append(np.mean(probs,axis=0))
        record_matrix(area,role,np.mean(components,axis=0),dict(train=key(fit),held=key(held)))
    finish_control(out,i)


def finish_control(out,i):
    """Complete inference from recorded fits if the host stopped between training and export."""
    fold,src,x,y,q,rc=sources(out,i)
    area=out/STUDY/str(i)/"control"
    classes=np.unique(y)
    repeat=read(src/"seed_replication.json")
    for role,(fit,held) in panels(src,x,y).items():
        meta=read(area/(role+".json"))
        require(meta["train"]==key(fit) and meta["held"]==key(held),"Cached panel uses another partition")
        members=[]
        for j,m in enumerate(rc["members"]):
            seeds=[None]+(repeat["seeds"] if repeat["action"]=="adopt" and m in repeat["eligible"] else [])
            values=[]
            for n,s in enumerate(seeds):
                tag=f"{role}_{j}_{n}"
                receipt=read(area/(tag+"_attempt.json"))
                require(receipt==dict(candidate=m,seed=s,fit_rows=len(fit),holdout_rows=len(held)),"Stale component")
                values.append(load_matrix(area,tag))
            members.append(np.mean(values,axis=0))
        np.testing.assert_array_equal(np.mean(members,axis=0),load_matrix(area,role))
    selection=read(src/"final_selection.json")
    path=src/selection["model_file"]
    require(sha(path)==rc["model"]==selection["model_sha256"],"Wrong deployed incumbent")
    p=ordered(joblib.load(path)["model"],q,classes)
    require(np.array_equal(classes[(p>.5).astype(int)].astype(str),prediction_csv(fold/"replicated/solve.csv",len(q))),
            "Reconstructed incumbent decisions differ")
    record_matrix(area,"external",p,dict(model=rc["model"],external_features=sha(fold/"anonymous/queries.zip")))
    n=len(list(area.glob("*_attempt.json")))
    save(area/"sealed.json",dict(launch=sha(area/"launch.json"),additional_control_fits=n,base_fits=rc["fits"],
         members=rc["members"],probabilities={k:sha(area/(k+".json")) for k in (*ROLES,"external")}))
    print("control",i,n,flush=True)


def applicable_methods(rows,labels):
    if len(np.unique(labels))!=2:
        return set()
    x=_clean_categories(rows)
    try:
        profile,_=blocks(x)
    except ValueError:
        return set()
    methods={"crossed"}
    if comparable_fields(x,profile):
        methods.add("posterior")
    return methods


def challengers(out,i):
    fold,src,x,y,q,_=sources(out,i)
    root=out/STUDY/str(i)
    classes=np.unique(y)
    parts=panels(src,x,y)
    available=applicable_methods(x,y)
    for fit,_ in parts.values():
        available.intersection_update(applicable_methods(x[fit],y[fit]))
    for name,model_type in METHODS.items():
        area=root/name
        area.mkdir(parents=True,exist_ok=False)
        save(area/"launch.json",dict(protocol=sha(PLAN),inputs=sha(fold/"inputs.json"),candidate=name))
        if name not in available:
            save(area/"sealed.json",dict(launch=sha(area/"launch.json"),available=False,classifier_fits=0,tables=0,
                 reason="Required recurring or comparable fields were absent from at least one fitting partition"))
            continue
        for role in (*ROLES,"external"):
            if role=="external":
                fit=np.arange(len(y))
                queries=q
                row_identity=sha(fold/"anonymous/queries.zip")
            else:
                fit,held=parts[role]
                queries=x[held]
                row_identity=key(held)
            tag=area/(role+"_attempt.json")
            save(tag,dict(train=key(fit),held=row_identity,requested_classifier_fits=1))
            start=time.monotonic()
            model=model_type().fit(x[fit],y[fit])
            values=normalized(model.predict_proba(queries)[:,[list(model.classes_).index(k) for k in classes]])
            stamp=dict(attempt=sha(tag),classifier_fits=1,train=key(fit),held=row_identity,
                       audit=model.audit,elapsed_seconds=round(time.monotonic()-start,3))
            if role=="external":
                path=area/"model.joblib"
                joblib.dump(model,path,compress=3)
                stamp["model"]=sha(path)
            record_matrix(area,role,values,stamp)
            del model
        save(area/"sealed.json",dict(launch=sha(area/"launch.json"),classifier_fits=3,classes=classes.tolist(),
             tables=sum(read(area/(r+".json"))["audit"]["supervised_map_constructions"] for r in (*ROLES,"external")),
             probabilities={k:sha(area/(k+".json")) for k in (*ROLES,"external")}))
        gc.collect()
        print("challenger",i,name,flush=True)


def metrics(p,truth,p0):
    return dict(rows=len(p),correct=int(np.sum((p>.5)==truth)),net=int(np.sum((p>.5)==truth)-np.sum((p0>.5)==truth)),
                brier=float(np.mean((p-truth)**2)),brier_change=float(np.mean((p-truth)**2-(p0-truth)**2)))


def elect(out,i):
    fold,src,x,y,q,rc=sources(out,i)
    area=out/STUDY/str(i)
    parts=panels(src,x,y)
    classes=np.unique(y)
    available=[n for n in METHODS if read(area/n/"sealed.json").get("available",True)]
    bank={n:{r:load_matrix(area/n,r) for r in (*ROLES,"external")} for n in ("control",*available)}
    for n in available:
        model_path=area/n/"model.joblib"
        meta=read(area/n/"external.json")
        require(sha(model_path)==meta["model"],"Unbound full-data challenger")
        model=joblib.load(model_path)
        p=normalized(model.predict_proba(q)[:,[list(model.classes_).index(k) for k in classes]])
        np.testing.assert_allclose(p,bank[n]["external"],rtol=1e-7,atol=1e-8)
        for r,(fit,held) in parts.items():
            review=read(area/n/(r+".json"))
            require(review["train"]==key(fit) and review["held"]==key(held)==read(area/"control"/(r+".json"))["held"],
                    "Mismatched experiment rows")
    options=[("control",0.)]+[(n,w) for n in available for w in (.125,.25,.5,1.)]
    def mixture(n,w,r):
        return (1-w)*bank["control"][r]+w*bank[n][r]
    def evaluate(n,w,r):
        return metrics(mixture(n,w,r),y[parts[r][1]]==classes[1],bank["control"][r])
    ranking={opt:evaluate(*opt,"nominate") for opt in options}
    nominee,weight=min(options,key=lambda a:(-ranking[a]["correct"],ranking[a]["brier"],a[1],a[0]))
    first=ranking[nominee,weight]
    second=evaluate(nominee,weight,"confirm")
    nrows=first["rows"]+second["rows"]
    accept=bool(weight and min(first["net"],second["net"])>=0 and (first["net"]+second["net"])/nrows>=.002
                and sum(r["brier_change"]*r["rows"] for r in (first,second))<=0.)
    save(area/"decision.json",dict(protocol=sha(PLAN),controls=sha(area/"control/sealed.json"),
         candidate_evidence={n:sha(area/n/"sealed.json") for n in METHODS},nominee=nominee,weight=weight,
         nomination=first,confirmation=second,action="adopt" if accept else "retain"))
    predictions={"control":bank["control"]["external"],"chosen":mixture(nominee,weight if accept else 0.,"external")}
    for n in METHODS:
        candidate=n if n in bank else "control"
        predictions[n]=bank[candidate]["external"]
        predictions[n+"_quarter"]=mixture(candidate,.25,"external")
    seals={}
    for tag,p in predictions.items():
        dst=area/(tag+"_prediction.npy")
        np.save(dst,classes[(p>.5).astype(int)].astype(str),allow_pickle=False)
        seals[tag]=sha(dst)
    added=read(area/"control/sealed.json")["additional_control_fits"]
    research_cost=added+sum(read(area/n/"sealed.json")["classifier_fits"] for n in METHODS)
    path_cost=rc["fits"]+added+2*len(available)+int(accept)
    require(path_cost<=64 and rc["fits"]+research_cost<=64,"Exceeded per-arm experimental ceiling")
    save(area/"sealed.json",dict(decision=sha(area/"decision.json"),predictions=seals,base_fits=rc["fits"],
         actual_new_classifier_fits=research_cost,simulated_path_fits=path_cost,independent_challenger_reload=True,api_requests=0))
    print("decision",i,nominee,weight,accept,first["net"],second["net"],flush=True)


def grade(out):
    root=out/STUDY
    files={str(i):sha(root/str(i)/"sealed.json") for i in range(5)}
    save(root/"score_lock.json",dict(protocol=sha(PLAN),folds=files))
    rows,groups,all_hits=[],[],{}
    cumulative_tables=0
    for i in range(5):
        src=origin(out,i)
        area=root/str(i)
        seal=read(area/"sealed.json")
        path=src/"outer_answers.joblib"
        require(sha(path)==read(src/"inputs.json")["outer_answers"] and sha(area/"decision.json")==seal["decision"],
                "Incorrect score packet")
        packet=joblib.load(path)
        y=packet["y"].astype(str)
        groups.extend(packet["groups"])
        scores={}
        for k,digest in seal["predictions"].items():
            path=area/(k+"_prediction.npy")
            require(sha(path)==digest,"Changed sealed answer")
            labels=np.load(path,allow_pickle=False)
            require(y.shape==labels.shape,"Mismatched output length")
            h=labels==y
            scores[k]=int(h.sum())
            all_hits.setdefault(k,[]).extend(h)
        rows.append(dict(fold=i,rows=len(y),correct=scores,decision=read(area/"decision.json"),
                         added_fits=seal["actual_new_classifier_fits"],simulated_path_fits=seal["simulated_path_fits"]))
        cumulative_tables+=read(area/"posterior/sealed.json")["tables"]
    totals={k:int(np.sum(v)) for k,v in all_hits.items()}
    anchor=np.asarray(all_hits["control"])
    paired={k:compare(np.asarray(v),anchor,groups) for k,v in all_hits.items() if k!="control"}
    delta=[(r["correct"]["chosen"]-r["correct"]["control"])/r["rows"] for r in rows]
    keep_advancing=bool(paired["chosen"]["gain"]>=.003 and sum(d>0 for d in delta)>=3 and min(delta)>=-.003)
    require(len(groups)==8117 and totals["control"]==read(out/"seed_completion/result.json")["correct"]["replicated"],
            "Wrong frozen five-fold baseline")
    value=dict(lock=sha(root/"score_lock.json"),rows=len(groups),folds=rows,correct=totals,versus_incumbent=paired,
               prospective_cold_qualification=keep_advancing,actual_new_fits=sum(r["added_fits"] for r in rows),
               supervised_response_maps=cumulative_tables,model_api_requests=0,designated_validation_opened=False,
               official_test_opened=False,h2_opened=False)
    save(root/"result.json",value)
    print(totals,paired["chosen"],keep_advancing,value["actual_new_fits"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("controls","finish-control","challengers","elect","grade"))
    parser.add_argument("--out",type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument("--folds",type=int,choices=range(5),nargs="+",default=list(range(5)))
    args=parser.parse_args()
    if args.action=="grade":
        grade(args.out)
    else:
        for idx in args.folds:
            {"controls":controls,"finish-control":finish_control,"challengers":challengers,"elect":elect}[args.action](args.out,idx)
