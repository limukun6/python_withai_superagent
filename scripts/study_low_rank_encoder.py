"""Fit a last-block attention update as a residual around the intact preference model."""
import argparse
import copy
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from scripts.cache_paired_encoders import configure_weights
from scripts.cache_training_tokens import last_block_pool
from scripts.study_paired_encoders import evaluate,order_key
from scripts.unified_released_study import ROOT,read,require,save,sha
from superagent.ab_sentence import ALTERNATIVES,_encoder,verified_files
from superagent.paired_encoder import text_key
from superagent.prediction_evidence import fingerprint


NAME="last_block_low_rank_update"
SEED=14731


class LowRankLinear(nn.Module):
    def __init__(self,frozen):
        super().__init__()
        self.frozen=frozen
        self.down=nn.Linear(frozen.in_features,8,bias=False)
        self.up=nn.Linear(8,frozen.out_features,bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self,x):
        return self.frozen(x)+self.up(self.down(x))


class PrefixBank:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            require(str(data["revision"])==ALTERNATIVES["gte"]["revision"] and str(data["layer"])=="penultimate",
                    "Wrong contextual activations")
            self.vectors=torch.from_numpy(data["vectors"])
            self.original=torch.from_numpy(data["original_pooled"])
            self.offsets=data["offsets"]
            self.index={key:i for i,key in enumerate(data["keys"].tolist())}
        require(len(self.original)==len(self.index) and self.offsets[-1]==len(self.vectors),"Corrupt prefix bank")

    def pairs(self,rows):
        return np.array([[self.index[text_key(a)],self.index[text_key(b)]] for a,b in rows])

    def collate(self,items):
        seq=[self.vectors[self.offsets[i]:self.offsets[i+1]] for i in items]
        lengths=torch.tensor([len(x) for x in seq])
        hidden=nn.utils.rnn.pad_sequence(seq,batch_first=True).float()
        mask=torch.arange(hidden.shape[1])[None,:]<lengths[:,None]
        return hidden,mask


def fit_fold(bank, base_block, parent, query_scores, data, folder):
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    block=copy.deepcopy(base_block).eval().requires_grad_(False)
    attention=block.attention.self
    attention.query=LowRankLinear(attention.query)
    attention.value=LowRankLinear(attention.value)
    trained=[v for v in block.parameters() if v.requires_grad]
    require(sum(t.numel() for t in trained)==12288,"Unexpected trainable parameter set")
    features=parent.named_steps["features"]
    width=len(features.center_)
    require(features.variant=="gte" and features.head=="linear" and len(features.semantic_blocks(
            [data["X_train"][0]],list(data["X_train"][0]))[0])==2,"Unknown parent feature layout")
    direction=torch.as_tensor(parent.named_steps["classifier"].coef_[0,-width-1:-1],dtype=torch.float32)/np.sqrt(width)
    center=torch.from_numpy(features.center_)
    scale=torch.from_numpy(features.scale_)
    def semantic(z):
        return torch.clamp((z-center)/scale,-6.,6.)@direction
    def correction(pairs):
        unique,inverse=np.unique(pairs.reshape(-1),return_inverse=True)
        hidden,mask=bank.collate(unique)
        new=last_block_pool(block,hidden,mask)
        change=semantic(new)-semantic(bank.original[unique])
        sides=change[torch.as_tensor(inverse)].reshape(-1,2)
        return sides[:,0]-sides[:,1]
    fit_ids,held_ids=bank.pairs(data["X_train"]),bank.pairs(data["X_search"])
    train_p=parent.predict_proba(data["X_train"])[:,1]
    to_logit=lambda p: torch.logit(torch.as_tensor(p,dtype=torch.float32),eps=1e-7)
    offsets=to_logit(train_p)
    y=torch.as_tensor(data["y_train"],dtype=torch.float32)
    with torch.no_grad():
        initial=float(correction(fit_ids[:32]).abs().max())
    require(initial<2e-4,"Zero adaptation failed to reproduce the parent")
    save(folder/"started.json",dict(fits=1,trainable_parameters=12288,initial_max_logit_change=initial))
    optimizer=torch.optim.AdamW(trained,lr=.001,weight_decay=.04)
    for epoch in range(2):
        indexes=rng.permutation(len(fit_ids))
        for offset in range(0,len(indexes),64):
            ids=indexes[offset:offset+64]
            logits=offsets[ids]+correction(fit_ids[ids])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y[ids])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trained,1.)
            optimizer.step()
        print(folder.name,"epoch",epoch+1,flush=True)
    with torch.no_grad():
        probe=held_ids[:12]
        torch.testing.assert_close(correction(probe),-correction(probe[:,::-1]),atol=2e-5,rtol=2e-5)
        all_changes=torch.cat([correction(held_ids[i:i+64]) for i in range(0,len(held_ids),64)])
        scores=torch.sigmoid(to_logit(query_scores)+all_changes).numpy()
    weights={n:p.detach().cpu() for n,p in block.named_parameters() if p.requires_grad}
    torch.save(weights,folder/"adapters.pt")
    np.save(folder/"scores.npy",scores)
    save(folder/"completed.json",dict(fits=1,parameters=12288,weights=sha(folder/"adapters.pt"),
         predictions=sha(folder/"scores.npy"),mean_absolute_logit_change=float(all_changes.abs().mean())))
    return scores


def run(args):
    torch.set_num_threads(2)
    configure_weights(args.weights)
    os.environ["SUPERAGENT_PAIRED_FEATURE_CACHE"]=str(args.cache)
    spec=ALTERNATIVES["gte"]
    root=verified_files(spec["env"],spec["files"])
    require(root is not None,"Missing verified GTE")
    base_block=_encoder(root,512)[1].encoder.layer[-1]
    bank=PrefixBank(args.bank)
    args.output.mkdir(parents=True,exist_ok=False)
    parent=read(args.reference/"results.json")["candidate_locked"][0]
    save(args.output/"protocol.json",dict(plan_sha256=sha(ROOT/"research/paired_encoder_search_1127/low_rank_protocol.json"),
         reference=parent,bank_sha256=sha(args.bank),new_model_requests=0))
    def fold(name):
        folder=args.output/name
        folder.mkdir()
        path=args.agent/(name+".joblib")
        spec=read(args.agent/(name+"_split.json"))
        data=joblib.load(path)
        labels=np.asarray(data["y_search"])
        original=joblib.load(args.reference/name/parent/"trial.joblib")
        observed=original["prediction_record"]
        require(sha(path)==spec["payload_sha256"]==observed["evaluation_payload_sha256"] and
                fingerprint(labels)==observed["label_sha256"] and original["metrics"]["resolved_configuration"]["id"]==parent,
                "Parent and training groups differ")
        p=np.asarray(original["search_scores"])[:,list(original["score_classes"]).index(1)]
        scores=fit_fold(bank,base_block,original["model"],p,data,folder)
        return {parent:p,NAME:scores},labels
    columns,y=fold("primary")
    options=[evaluate(g,columns,y,parent) for g in ([NAME],[parent,NAME])]
    best=min(options,key=order_key)
    selected=best["members"] if best["delta"]>0 else []
    save(args.output/"primary_lock.json",dict(options=options,selected=selected))
    confirmed=[]
    if selected:
        for name in ("confirmation_1","confirmation_2"):
            columns,y=fold(name)
            row=evaluate(selected,columns,y,parent)
            confirmed.append(row)
            if row["delta"]<=0:break
    checked=[best,*confirmed]
    pooled=sum(r["gained"]-r["lost"] for r in checked)/sum(r["rows"] for r in checked)
    fits=len(list(args.output.glob("*/started.json")))
    require(fits==len(list(args.output.glob("*/completed.json")))<=3,"Missing training receipt")
    result=dict(primary=options,selected=selected,confirmations=confirmed,pooled_gain=pooled,
                passed=len(confirmed)==2 and min(r["delta"] for r in checked)>0 and pooled>=.003,
                actual_fits=fits,trainable_parameters=12288,model_api_requests=0,designated_validation_opened=False)
    save(args.output/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("weights","cache","bank","agent","reference","output"):
        parser.add_argument("--"+n,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
