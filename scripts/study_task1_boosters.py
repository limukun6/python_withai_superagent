"""Run native categorical XGBoost and an interpretable additive interaction learner."""
import argparse
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.study_task1_candidates import ROLES,bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.attribute_alignment import aligned_columns,comparable_fields,fitted_routes
from superagent.joint_context import blocks,tokens

PLAN=ROOT/"research/task1_feedback_extension_1127/alternative_boosting.json"
METHODS=("xgb_raw","xgb_structure","additive")


def training_views(raw,y,groups,queries,name):
    if name=="xgb_raw":
        return raw,queries
    profile,context=blocks(raw)
    def extend(x):
        return np.column_stack((x,tokens(x[:,profile]),tokens(x[:,context])))
    train=extend(raw)
    held={role:extend(x) for role,x in queries.items()}
    if name=="xgb_structure":
        families=comparable_fields(raw,profile)
        if families:
            extra=np.empty((len(raw),2*len(families)),dtype=object)
            for fit,inner in GroupKFold(3).split(raw,y,groups):
                routes=fitted_routes(raw[fit],y[fit],families,context)
                extra[inner]=aligned_columns(raw[inner],routes)
            full=fitted_routes(raw,y,families,context)
            train=np.column_stack((train,extra))
            held={role:np.column_stack((x,aligned_columns(queries[role],full))) for role,x in held.items()}
    return train,held


def train(out,name):
    folder=out/("boost_"+name)
    folder.mkdir(exist_ok=False)
    info=read(out/"inputs.json")
    fitting=bound(out,"fitting.joblib",info)
    y=fitting["y"]
    queries={role:bound(out,role+"_features.joblib",info) for role in ROLES}
    x,queries=training_views(fitting["x"],y,fitting["groups"],queries,name)
    if name.startswith("xgb"):
        from xgboost import XGBClassifier
        require(version("xgboost")=="3.2.0","Unexpected XGBoost build")
        levels=[np.unique(col) for col in x.T]
        def nominal(rows):
            return pd.DataFrame({str(i):pd.Categorical.from_codes(pd.Index(v).get_indexer(col),categories=v) for i,(col,v) in enumerate(zip(rows.T,levels))})
        x=nominal(x)
        queries={role:nominal(v) for role,v in queries.items()}
        model=XGBClassifier(enable_categorical=True,tree_method="hist",n_estimators=600,learning_rate=.04,max_depth=6,
                            min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_lambda=4,
                            max_cat_to_onehot=4,max_cat_threshold=64,n_jobs=2,random_state=2081)
    else:
        from interpret.glassbox import ExplainableBoostingClassifier
        require(version("interpret-core")=="0.7.8","Unexpected EBM build")
        model=ExplainableBoostingClassifier(feature_types=["nominal"]*x.shape[1],interactions=12,outer_bags=4,
                                            inner_bags=0,max_rounds=2500,early_stopping_rounds=75,n_jobs=2,random_state=2081)
    save(folder/"attempt.json",dict(protocol=sha(PLAN),fitting_rows=len(y),features=x.shape[1],estimator_calls=1))
    start=time.monotonic()
    model.fit(x,y)
    pred={}
    for role,rows in queries.items():
        probs=np.asarray(model.predict_proba(rows),dtype=float)
        require(probs.shape==(len(rows),len(model.classes_)) and np.isfinite(probs).all()
                and np.allclose(probs.sum(1),1,atol=1e-5),"Invalid candidate probabilities")
        path=folder/(role+".npy")
        np.save(path,probs,allow_pickle=False)
        pred[role]=sha(path)
    seal=dict(attempt=sha(folder/"attempt.json"),classes=model.classes_.tolist(),predictions=pred,
              classifier_fits=1,internal_bags=4 if name=="additive" else None,
              seconds=round(time.monotonic()-start,2),api_requests=0)
    save(folder/"sealed.json",seal)
    print(name,seal,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=(*METHODS,"elect"))
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    if args.action=="elect":
        elect(args.out,METHODS,"boost",PLAN)
    else:
        train(args.out,args.action)
