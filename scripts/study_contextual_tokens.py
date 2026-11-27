"""Train shared token preference heads using only the predeclared training folds."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from scripts.study_paired_encoders import evaluate, order_key
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.ab_sentence import ALTERNATIVES
from superagent.models import style_differences
from superagent.paired_encoder import text_key
from superagent.prediction_evidence import fingerprint


PROTOCOL = ROOT/"research/paired_encoder_search_1127/token_protocol.json"
HEADS = ("token_attention","token_mean")
SEED = 11419


class TokenHead(nn.Module):
    def __init__(self, width, styles, attention):
        super().__init__()
        self.mean = nn.Linear(width,1,bias=False)
        self.style = nn.Linear(styles,1,bias=False)
        self.drop = nn.Dropout(.12)
        self.attention = attention
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.style.weight)
        if attention:
            self.key = nn.Linear(width,24,bias=False)
            self.query = nn.Linear(width,24)
            self.detail = nn.Linear(width,1,bias=False)
            nn.init.zeros_(self.detail.weight)

    def forward(self, tokens, mask, styles):
        weights = mask.float()/mask.sum(1,keepdim=True)
        pooled = torch.einsum("bt,btd->bd",weights,tokens)
        left, right = pooled[::2], pooled[1::2]
        logits = self.mean(self.drop(left-right)) + self.style(self.drop(styles))
        if self.attention:
            query = self.query((left+right)*.5).repeat_interleave(2,dim=0)
            attention = torch.einsum("btd,bd->bt",self.key(tokens),query)/np.sqrt(query.shape[1])
            weight = torch.softmax(attention.masked_fill(~mask,-torch.inf),dim=1)
            detail = torch.einsum("bt,btd->bd",weight-weights,tokens)
            logits = logits+self.detail(self.drop(detail[::2]-detail[1::2]))
        return logits.squeeze(-1)


class TokenBank:
    def __init__(self, path):
        with np.load(path,allow_pickle=False) as archive:
            require(str(archive["revision"])==ALTERNATIVES["gte"]["revision"],"Encoder revision mismatch")
            self.offsets = archive["offsets"]
            self.vectors = torch.from_numpy(archive["vectors"])
            self.ids = {k:i for i,k in enumerate(archive["keys"].tolist())}
        lengths = np.diff(self.offsets)
        require(np.all(lengths>0) and self.offsets[-1]==len(self.vectors),"Incomplete token cache")
        summed = np.add.reduceat(self.vectors.numpy(),self.offsets[:-1],axis=0,dtype=np.float32)
        self.means = summed/lengths[:,None].astype(np.float32)

    def ids_for(self, rows):
        return np.array([[self.ids[text_key(s)] for s in row] for row in rows])

    def batch(self, pairs, center, scale):
        tensors = [self.vectors[self.offsets[i]:self.offsets[i+1]] for i in pairs.reshape(-1)]
        lengths = torch.tensor([len(v) for v in tensors])
        series = nn.utils.rnn.pad_sequence(tensors,batch_first=True).float()
        mask = torch.arange(series.shape[1])[None,:]<lengths[:,None]
        return torch.clamp((series-center)/scale,-8.,8.),mask


def fit_head(bank, data, name, path):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows, query = data["X_train"],data["X_search"]
    ids, held = bank.ids_for(rows),bank.ids_for(query)
    true = torch.as_tensor(data["y_train"],dtype=torch.float32)
    means = bank.means[np.unique(ids)]
    center = torch.from_numpy(means.mean(axis=0))
    scale = torch.from_numpy(np.maximum(means.std(axis=0),1e-4))
    feat = style_differences(rows)
    style_scale = np.sqrt(np.maximum(np.mean(feat**2,axis=0),1e-6))
    style = torch.as_tensor(np.clip(feat/style_scale,-8.,8.),dtype=torch.float32)
    held_style = torch.as_tensor(np.clip(style_differences(query)/style_scale,-8.,8.),dtype=torch.float32)
    model = TokenHead(len(center),style.shape[1],name=="token_attention")
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.04)
    with path.with_suffix(".started.json").open("x") as out:
        json.dump(dict(candidate=name,actual_supervised_fits=1,seed=SEED,epochs=3),out)
    for _ in range(3):
        order = rng.permutation(len(ids))
        for offset in range(0,len(ids),96):
            idx=order[offset:offset+96]
            logits=model(*bank.batch(ids[idx],center,scale),style[idx])
            loss=nn.functional.binary_cross_entropy_with_logits(logits,true[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        check = min(8,len(held))
        original=model(*bank.batch(held[:check],center,scale),held_style[:check])
        swapped=model(*bank.batch(held[:check,::-1],center,scale),-held_style[:check])
        torch.testing.assert_close(original,-swapped,atol=3e-5,rtol=3e-5)
        chunks=[]
        for start in range(0,len(held),96):
            idx=slice(start,start+96)
            chunks.append(torch.sigmoid(model(*bank.batch(held[idx],center,scale),held_style[idx])).numpy())
    torch.save(dict(weights=model.state_dict(),center=center,scale=scale,style_scale=torch.as_tensor(style_scale)),path)
    return np.concatenate(chunks)


def study(args):
    torch.set_num_threads(2)
    bank = TokenBank(args.bank)
    args.output.mkdir(parents=True,exist_ok=False)
    ref = read(args.reference/"results.json")
    require(ref["advance_to_raw_solve"] and len(ref["candidate_locked"])==1,"Missing reference winner")
    incumbent = ref["candidate_locked"][0]
    save(args.output/"protocol.json",dict(protocol_sha256=sha(PROTOCOL),bank_sha256=sha(args.bank),
         reference_sha256=sha(args.reference/"results.json"),model_api_requests=0))
    fits=0
    def fold(name, heads):
        nonlocal fits
        directory=args.output/name
        directory.mkdir()
        payload=args.agent/(name+".joblib")
        spec=read(args.agent/(name+"_split.json"))
        require(sha(payload)==spec["payload_sha256"],"Fold changed")
        data=joblib.load(payload)
        labels=np.asarray(data["y_search"])
        previous=joblib.load(args.reference/name/incumbent/"trial.joblib")
        checked=previous["prediction_record"]
        require(checked["evaluation_payload_sha256"]==sha(payload) and checked["label_sha256"]==fingerprint(labels),
                "Misaligned reference probabilities")
        columns={incumbent:np.asarray(previous["search_scores"])[:,list(previous["score_classes"]).index(1)]}
        for head in heads:
            path=directory/(head+".pt")
            probs=fit_head(bank,data,head,path)
            require(probs.shape==labels.shape and np.isfinite(probs).all(),"Unusable scores")
            np.save(directory/(head+".npy"),probs)
            save(directory/(head+".finished.json"),dict(fits=1,model_sha256=sha(path),
                 rows=spec["row_sha256"],scores_sha256=sha(directory/(head+".npy"))))
            fits+=1
            columns[head]=probs
        return columns,labels
    columns,labels=fold("primary",HEADS)
    primary=[evaluate(g,columns,labels,incumbent) for head in HEADS for g in ([head],[incumbent,head])]
    best=min(primary,key=order_key)
    selected=best["members"] if best["delta"]>0 else []
    save(args.output/"discovery_lock.json",dict(primary=primary,selected=selected))
    print("primary",[(r["members"],r["correct"],r["gained"]-r["lost"]) for r in primary],flush=True)
    confirmations=[]
    if selected:
        for name in ("confirmation_1","confirmation_2"):
            columns,labels=fold(name,[m for m in selected if m!=incumbent])
            result=evaluate(selected,columns,labels,incumbent)
            confirmations.append(result)
            if result["delta"]<=0:
                break
    checked=[best,*confirmations]
    pooled=sum(x["gained"]-x["lost"] for x in checked)/sum(x["rows"] for x in checked)
    passed=len(confirmations)==2 and min(r["delta"] for r in checked)>0 and pooled>=.003
    require(fits==len(list(args.output.glob("*/*.started.json")))==len(list(args.output.glob("*/*.finished.json"))),
            "Fit accounting changed")
    result=dict(primary=primary,selected=selected,confirmations=confirmations,pooled_gain=pooled,passed=passed,
                actual_fits=fits,new_api_requests=0,opened_designated_labels=False)
    save(args.output/"results.json",result)
    print("confirmations",confirmations,"passed",passed,"fits",fits,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("agent","bank","reference","output"):
        parser.add_argument("--"+n,required=True,type=lambda s:Path(s).resolve())
    study(parser.parse_args())
