"""Train two local TabM representations and assess any complementary predictions."""
import argparse
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.joint_context import blocks,tokens

PLAN=ROOT/"research/task1_feedback_extension_1127/tabm.json"
METHODS=("raw","structure")


def train(out,name):
    import torch
    from tabm import TabM
    require(version("tabm")=="0.0.3" and version("rtdl-num-embeddings")=="0.0.12","Changed neural implementation")
    folder=out/("tabm_"+name)
    folder.mkdir(exist_ok=False)
    info=read(out/"inputs.json")
    fit=bound(out,"fitting.joblib",info)
    x,y=fit["x"],fit["y"]
    an=np.unique(y)
    profiles,context=blocks(x) if name=="structure" else ((),())
    def represent(x):
        return np.column_stack((x,tokens(x[:,profiles]),tokens(x[:,context]))) if profiles else x
    encoder=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.int64)
    rows=encoder.fit_transform(represent(x))+1
    sizes=[len(values)+1 for values in encoder.categories_]
    device="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    seed=61839
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng=np.random.default_rng(seed)
    features=torch.tensor(rows,dtype=torch.long,device=device)
    truth=torch.tensor(np.searchsorted(an,y),dtype=torch.long,device=device)
    k=8
    model=TabM.make(n_num_features=0,cat_cardinalities=sizes,d_out=len(an),k=k,n_blocks=3,d_block=128,dropout=.15).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
    save(folder/"attempt.json",dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),device=device,
         optimizer_trainings=1,shared_heads=k,rows=len(y),fields=len(sizes),downloaded_weights=False))
    start=time.monotonic()
    model.train()
    for _ in range(32):
        order=rng.permutation(len(y))
        for i in range(0,len(y),256):
            ix=torch.tensor(order[i:i+256],dtype=torch.long,device=device)
            logits=model(None,features[ix])
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),truth[ix].repeat_interleave(k))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    bank={}
    with torch.inference_mode():
        for role in ROLES:
            query=bound(out,role+"_features.joblib",info)
            encoded=encoder.transform(represent(query))+1
            chunks=[]
            for batch in range(0,len(query),128):
                tensor=torch.tensor(encoded[batch:batch+128],dtype=torch.long,device=device)
                chunks.append(torch.softmax(model(None,tensor),dim=-1).mean(1).cpu().numpy())
            p=np.concatenate(chunks).astype(float)
            require(p.shape==(len(query),len(an)) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5),"Invalid output")
            path=folder/(role+".npy")
            np.save(path,p,allow_pickle=False)
            bank[role]=sha(path)
    result=dict(attempt=sha(folder/"attempt.json"),predictions=bank,classes=an.tolist(),
                optimizer_trainings=1,heads=k,device=device,seconds=round(time.monotonic()-start,2),api_requests=0)
    save(folder/"sealed.json",result)
    print(name,result,flush=True)


def elect(out, methods=METHODS, prefix="tabm", protocol=PLAN):
    info=read(out/"inputs.json")
    folder=out/(prefix+"_election")
    folder.mkdir(exist_ok=False)
    paths={"aligned":out/"trial_attribute_alignment","icl":out/"trial_pretrained_icl",
           **{n:out/(prefix+"_"+n) for n in methods}}
    seals={k:read(p/"sealed.json") for k,p in paths.items()}
    classes=np.asarray(seals["aligned"]["classes"])
    require(all(d["classes"]==classes.tolist() for d in seals.values()),"Changed class order")
    def panel(role,options):
        bank={}
        for k,p in paths.items():
            f=p/(role+".npy")
            require(sha(f)==seals[k]["predictions"][role],"Prediction changed")
            bank[k]=np.load(f,allow_pickle=False)
        ref=(bank["aligned"]+bank["icl"])/2
        y=bound(out,role+"_answers.joblib",info)
        base=classes[ref.argmax(1)]==y
        results={}
        for n,w in options:
            pred=ref if not w else (1-w)*ref+w*bank[n]
            results[n,w]=paired(classes[pred.argmax(1)]==y,base)
        return results,len(y),int(base.sum())
    choices=[("retain",0.)]+[(n,w) for n in methods for w in (.125,.25,.5,1.)]
    measured,size,ref=panel("nomination",choices)
    winner=min(choices,key=lambda v:(-measured[v]["net"],v[1],v[0]))
    save(folder/"locked.json",dict(protocol=sha(protocol),winner=list(winner),rows=size,reference=ref,
         alternatives=[dict(method=n,share=w,**r) for (n,w),r in measured.items()]))
    subsequent=[]
    for role in ROLES[1:]:
        evidence,rows,base=panel(role,[winner])
        subsequent.append(dict(role=role,rows=rows,reference=base,**evidence[winner]))
    gain=sum(r["net"] for r in subsequent)/sum(r["rows"] for r in subsequent)
    result=dict(lock=sha(folder/"locked.json"),winner=list(winner),checks=subsequent,gain=gain,
                eligible=min(r["net"] for r in subsequent)>=0 and gain>=.003,api_requests=0)
    save(folder/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=(*METHODS,"elect"))
    parser.add_argument("--out",required=True,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    if args.action=="elect":
        elect(args.out)
    else:
        train(args.out,args.action)
