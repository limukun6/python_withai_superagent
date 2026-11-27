"""Repackage and replay a previously delivered model without training or label access."""
import argparse
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.study_task1_outer import arrays,prediction_csv
from scripts.unified_released_study import read,require,save,sha
from superagent.models import probability_with_model


def check(fold,out,mode):
    src=fold/"regular"
    seal=read(src/"sealed.json")
    model=src/"run/agent/selected_model.joblib"
    require(sha(model)==seal["model"] and seal["independent_reload"],"Original model was not sealed")
    compact=out/"compact.joblib"
    if mode=="pack":
        out.mkdir(exist_ok=False)
        joblib.dump(joblib.load(model),compact)
        save(out/"packed.json",dict(source=sha(model),compact=sha(compact),old_bytes=model.stat().st_size,
                                     new_bytes=compact.stat().st_size))
        return
    packet=read(fold/"inputs.json")
    query=fold/"anonymous/queries.zip"
    require(sha(query)==packet["entry_queries"],"Unexpected query features")
    x,=arrays(query,["C_test.npy"])
    start=time.monotonic()
    bundle=joblib.load(model if mode=="native" else compact)
    labels,p=probability_with_model(bundle["kind"],bundle["model"],x)
    pred=np.asarray(labels)[p.argmax(1)].astype(str)
    expected=prediction_csv(src/"solve.csv",len(x))
    require(np.array_equal(expected,pred),"Predictions changed during packaging")
    spent=round(time.monotonic()-start,2)
    if mode=="native":
        np.save(out/"native_probs.npy",p,allow_pickle=False)
        save(out/"native.json",dict(rows=len(x),seconds=spent,predictions=seal["prediction"]))
    else:
        prior=np.load(out/"native_probs.npy",allow_pickle=False)
        np.testing.assert_allclose(prior,p,rtol=1e-6,atol=1e-6)
        result=dict(**read(out/"packed.json"),rows=len(x),hard_prediction_changes=int(np.sum(expected!=pred)),
                    max_abs_probability_delta=float(np.max(abs(prior-p))),native_cold_seconds=read(out/"native.json")["seconds"],
                    compact_cold_seconds=spent,additional_training_fits=0,api_requests=0)
        save(out/"result.json",result)
        print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["pack","native","replay"])
    for name in ("fold","out"):
        parser.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    check(a.fold,a.out,a.mode)
