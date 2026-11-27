"""Compare native numerics and additional independently seeded views of a frozen tabular transformer."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_measurements import interpret, recognize
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.table_foundation import SHA256, checkpoint, device

PLAN=ROOT/"research/task1_feedback_extension_1127/tabicl_typed_views.json"
SEEDS=(173,683,1607,3091)
METHODS=("raw16","typed4","typed16")


def typed(x, kinds):
    columns={}
    for j in range(x.shape[1]):
        if j not in kinds:
            columns[str(j)]=x[:,j].astype(object)
            continue
        entries=[interpret(v) for v in x[:,j]]
        value=np.asarray([e[1] if e and e[0]==kinds[j] else np.nan for e in entries],dtype=float)
        columns[str(j)]=value
        if kinds[j]=="clock":
            phase=value*(2*np.pi/1440.)
            columns[str(j)+"_sin"]=np.sin(phase)
            columns[str(j)+"_cos"]=np.cos(phase)
    return pd.DataFrame(columns)


def run(out):
    import torch
    from tabicl import TabICLClassifier
    info=read(out/"inputs.json")
    fit=bound(out,"fitting.joblib",info)
    x,y=fit["x"],fit["y"]
    classes=np.unique(y)
    panels=[bound(out,r+"_features.joblib",info) for r in ROLES]
    src=out/"trial_pretrained_icl"
    anchor=read(src/"sealed.json")
    weights=checkpoint()
    require(weights is not None and sha(weights)==SHA256 and version("tabicl")=="2.0.3"
            and anchor["classes"]==classes.tolist(),"Changed foundation baseline")
    archives=out/"typed_icl_components"
    archives.mkdir(exist_ok=True)
    start=time.monotonic()
    torch.set_num_threads(3)
    successes={}
    matrix={}
    kinds=recognize(x)
    for representation in ("raw","typed"):
        encode=(lambda data:pd.DataFrame(data,dtype=object)) if representation=="raw" else (lambda data:typed(data,kinds))
        train=encode(x)
        features=[encode(a) for a in panels]
        for seed in SEEDS:
            dest=src if (representation,seed)==("raw",173) else archives/(representation+"_"+str(seed))
            if dest==src:
                seals=anchor
            elif (dest/"sealed.json").is_file():
                seals=read(dest/"sealed.json")
                require(seals["protocol"]==sha(PLAN) and seals["source_fit"]==sha(out/"fitting.joblib")
                        and seals["classes"]==classes.tolist(),"Unbound partial component")
            else:
                dest.mkdir(exist_ok=False)
                model=TabICLClassifier(n_estimators=4,batch_size=1,model_path=weights,allow_auto_download=False,
                        device=device(),use_amp=False,use_fa3=False,kv_cache=True,n_jobs=1,random_state=seed)
                model.fit(train,y)
                require(np.array_equal(model.classes_,classes),"Unaligned TabICL classes")
                hashes={}
                for role,part in zip(ROLES,features):
                    values=np.asarray(model.predict_proba(part),dtype=float)
                    require(np.isfinite(values).all() and np.allclose(values.sum(1),1.,atol=1e-5),"Invalid TabICL posterior")
                    p=dest/(role+".npy")
                    np.save(p,values,allow_pickle=False)
                    hashes[role]=sha(p)
                seals=dict(protocol=sha(PLAN),source_fit=sha(out/"fitting.joblib"),classes=classes.tolist(),
                           representation=representation,seed=seed,views=4,context_fits=1,predictions=hashes)
                save(dest/"sealed.json",seals)
                del model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                print(representation,seed,round(time.monotonic()-start,2),flush=True)
            successes[representation,seed]=sha(dest/"sealed.json")
            bank=[]
            for role in ROLES:
                p=dest/(role+".npy")
                require(sha(p)==seals["predictions"][role],"Changed component output")
                bank.append(np.load(p,allow_pickle=False))
            matrix[representation,seed]=bank
    for name,rep,seeds in (("raw16","raw",SEEDS),("typed4","typed",SEEDS[:1]),("typed16","typed",SEEDS)):
        folder=out/("typedicl_"+name)
        folder.mkdir(exist_ok=False)
        hashes={}
        for i,role in enumerate(ROLES):
            p=folder/(role+".npy")
            np.save(p,np.mean([matrix[rep,s][i] for s in seeds],axis=0),allow_pickle=False)
            hashes[role]=sha(p)
        record=dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),classes=classes.tolist(),
                    component_seals=[successes[rep,s] for s in seeds],views=4*len(seeds),
                    field_types=list(kinds.values()) if rep=="typed" else [],predictions=hashes,
                    local_weight_sha256=SHA256,device=device(),api_requests=0)
        save(folder/"sealed.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","elect"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    a=parser.parse_args()
    if a.action=="run":
        run(a.out)
    else:
        elect(a.out,METHODS,"typedicl",PLAN)
