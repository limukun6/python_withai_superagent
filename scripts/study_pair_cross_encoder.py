"""Training-only test of jointly encoded text pairs in both directions."""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix,hstack,vstack
from sklearn.linear_model import LogisticRegression
import torch
from tokenizers import Tokenizer
from transformers import BertForSequenceClassification

from scripts.cache_paired_encoders import configure_weights
from scripts.fetch_ranker_probe import FILES,REVISION,verified
from scripts.study_paired_encoders import evaluate,order_key
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.paired_encoder import PairedEncoderFeatures, text_key
from superagent.prediction_evidence import fingerprint


PROTOCOL=ROOT/"research/paired_encoder_search_1127/cross_protocol.json"
METHOD="gte_with_joint_text_pair"


def pair_key(left,right):
    return text_key(left)+text_key(right)


def encode(bank_path, pairs, folder):
    require(all(verified(folder/n,*identity) for n,identity in FILES.items()),"Unverified public weights")
    tokenizer=Tokenizer.from_file(str(folder/"tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    model, info=BertForSequenceClassification.from_pretrained(folder,local_files_only=True,
                                        use_safetensors=True,output_loading_info=True)
    require(not (info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]),"Incomplete cross encoder")
    model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    canonical=sorted({tuple(sorted(pair)) for pair in pairs})
    packed=tokenizer.encode_batch([order for a,b in canonical for order in ((a,b),(b,a))])
    order=sorted(range(len(packed)),key=lambda i:len(packed[i].ids))
    width=model.config.hidden_size
    outputs=np.empty((len(packed),2*width+1),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,len(order),64):
            indices=order[start:start+64]
            shape=(len(indices),len(packed[indices[-1]].ids))
            ids=torch.zeros(shape,dtype=torch.long)
            types=torch.zeros_like(ids)
            attention=torch.zeros_like(ids)
            ordinary=torch.zeros(shape,dtype=torch.bool)
            for r,i in enumerate(indices):
                item=packed[i]
                n=len(item.ids)
                ids[r,:n]=torch.tensor(item.ids)
                types[r,:n]=torch.tensor(item.type_ids)
                attention[r,:n]=1
                ordinary[r,:n]=~torch.tensor(item.special_tokens_mask,dtype=torch.bool)
            result=model(input_ids=ids,token_type_ids=types,attention_mask=attention,output_hidden_states=True)
            hidden=result.hidden_states[-1]
            def pooled(segment):
                mask=(ordinary & (types==segment)).float()
                return torch.einsum("bt,btd->bd",mask,hidden)/mask.sum(1,keepdim=True).clamp_min(1.)
            outputs[indices,:width]=hidden[:,0].numpy()
            outputs[indices,width:2*width]=(pooled(0)-pooled(1)).numpy()
            outputs[indices,-1]=result.logits[:,0].numpy()
    values=0.5*(outputs[::2]-outputs[1::2])
    values[[a==b for a,b in canonical]]=0.
    require(np.isfinite(values).all(),"Nonfinite paired features")
    with bank_path.open("xb") as stream:
        np.savez(stream, revision=REVISION, keys=np.array([pair_key(*p) for p in canonical]),vectors=values)
    return values,canonical


def study(args):
    configure_weights(args.weights)
    os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(args.cache)
    args.output.mkdir(parents=True,exist_ok=False)
    incumbent="gte_paired_linear"
    ref=read(args.reference/"results.json")
    require(ref["advance_to_raw_solve"] and ref["candidate_locked"]==[incumbent],"Unexpected research reference")
    save(args.output/"protocol.json",dict(plan=sha(PROTOCOL),reference=sha(args.reference/"results.json"),
         method=METHOD,model=REVISION,model_api_requests=0))
    primary=joblib.load(args.agent/"primary.joblib")
    values,canonical=encode(args.output/"cross_bank.npz",[*primary["X_train"],*primary["X_search"]],
                            args.weights/"ms-marco-TinyBERT-L2-v2")
    index={pair:i for i,pair in enumerate(canonical)}
    def raw(rows):
        ids=[index[tuple(sorted(pair))] for pair in rows]
        sign=np.asarray([1. if a<=b else -1. for a,b in rows])
        return values[ids]*sign[:,None]
    fits=0
    def fold(name):
        nonlocal fits
        directory=args.output/name
        directory.mkdir()
        payload=args.agent/(name+".joblib")
        data=joblib.load(payload)
        spec=read(args.agent/(name+"_split.json"))
        require(sha(payload)==spec["payload_sha256"],"Incorrect training input")
        y=np.asarray(data["y_train"])
        held=np.asarray(data["y_search"])
        original=joblib.load(args.reference/name/incumbent/"trial.joblib")
        pr=original["prediction_record"]
        require(pr["evaluation_payload_sha256"]==spec["payload_sha256"] and pr["label_sha256"]==fingerprint(held),
                "Incomparable reference")
        reference=np.asarray(original["search_scores"])[:,list(original["score_classes"]).index(1)]
        train_rows,held_rows=data["X_train"],data["X_search"]
        features=PairedEncoderFeatures("gte","linear").fit(train_rows)
        differences=raw(train_rows)
        scale=np.maximum(np.sqrt(np.mean(differences**2,axis=0)),1e-4)
        width=(differences.shape[1]-1)//2
        block_scale=np.r_[np.full(2*width,np.sqrt(width)),1.]
        def matrix(rows,dense):
            return hstack([features.transform(rows),csr_matrix(np.clip(dense/scale,-6.,6.)/block_scale)],format="csr")
        x=matrix(train_rows,differences)
        save(directory/"started.json",dict(candidate=METHOD,actual_fits=1,split=spec["payload_sha256"]))
        model=LogisticRegression(C=.3,fit_intercept=False,max_iter=2000,random_state=42)
        model.fit(vstack([x,-x]),np.r_[y,1-y])
        fits+=1
        scores=model.predict_proba(matrix(held_rows,raw(held_rows)))[:,1]
        require(np.isfinite(scores).all(),"Invalid model predictions")
        path=directory/"fitted.joblib"
        joblib.dump(dict(encoder=features,classifier=model,scale=scale,block_scale=block_scale),path)
        np.save(directory/"scores.npy",scores)
        save(directory/"completed.json",dict(fits=1,model=sha(path),scores=sha(directory/"scores.npy"),rows=spec["row_sha256"]))
        return {METHOD:scores,incumbent:reference},held
    scores,held=fold("primary")
    options=[evaluate(g,scores,held,incumbent) for g in ([METHOD],[incumbent,METHOD])]
    best=min(options,key=order_key)
    selected=best["members"] if best["delta"]>0 else []
    save(args.output/"primary_lock.json",dict(options=options,selected=selected))
    confirmations=[]
    if selected:
        for name in ("confirmation_1","confirmation_2"):
            scores,held=fold(name)
            check=evaluate(selected,scores,held,incumbent)
            confirmations.append(check)
            if check["delta"]<=0:break
    checks=[best,*confirmations]
    pooled=sum(r["gained"]-r["lost"] for r in checks)/sum(r["rows"] for r in checks)
    result=dict(primary=options,selected=selected,confirmations=confirmations,pooled_gain=pooled,
                passed=len(confirmations)==2 and min(r["delta"] for r in checks)>0 and pooled>=.003,
                fits=fits,new_api_requests=0,designated_validation_opened=False)
    require(len(list(args.output.glob("*/started.json")))==len(list(args.output.glob("*/completed.json")))==fits,
            "Missing fit receipts")
    save(args.output/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("agent","weights","cache","reference","output"):
        p.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    study(p.parse_args())
