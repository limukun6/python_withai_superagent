"""Compare two shared item utilities with the portfolio elected by training feedback."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from scripts.compare_regrouped_portfolios import lock
from scripts.study_resource_audition import paired
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.models import text_statistics
from superagent.prediction_evidence import fingerprint
from superagent.preference_analogy import embedding_function
from superagent.splits import sample_groups, take


PLAN = ROOT / "research/paired_encoder_search_1127/item_utility_protocol.json"


class ItemUtility(nn.Module):
    def __init__(self, width, nonlinear):
        super().__init__()
        self.linear = nn.Linear(width, 1, bias=False)
        self.hidden = nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,1,bias=False)) if nonlinear else None

    def forward(self, items):
        scores = self.linear(items)
        if self.hidden is not None:
            scores = scores + self.hidden(items)
        return scores[:,0]


def documents(pairs):
    texts = list(dict.fromkeys(s for row in pairs for s in row))
    indices = {s:i for i,s in enumerate(texts)}
    return texts, torch.tensor([[indices[a],indices[b]] for a,b in pairs], dtype=torch.long)


def representation(texts, providers, fitted=None):
    transform, blocks = [], []
    values = [np.asarray(text_statistics(t), dtype=np.float32) for t in texts]
    for i in range(len(providers)+1):
        block = np.asarray(values) if i == len(providers) else np.asarray(providers[i](texts),dtype=np.float32)
        if fitted is None:
            pca = None
            if i < len(providers):
                pca = PCA(n_components=min(64, len(block)-1, block.shape[1]),svd_solver="randomized",random_state=19037)
                pca.fit(block)
            projected = block if pca is None else pca.transform(block)
            center, scale = projected.mean(0), np.maximum(projected.std(0),1e-5)
        else:
            pca, center, scale = fitted[i]
            projected = block if pca is None else pca.transform(block)
        transform.append((pca,center,scale))
        blocks.append(np.clip((projected-center)/scale,-6,6)/np.sqrt(projected.shape[1]))
    result = np.column_stack(blocks).astype(np.float32)
    require(np.isfinite(result).all(), "Nonfinite item features")
    return torch.from_numpy(result),transform


def train(x, pairs, truth, nonlinear):
    torch.manual_seed(19037)
    network = ItemUtility(x.shape[1],nonlinear)
    optimizer = torch.optim.AdamW(network.parameters(),lr=.01,weight_decay=.02)
    generator = torch.Generator().manual_seed(27183)
    y = torch.tensor(truth,dtype=torch.float32)
    for _ in range(16):
        for ids in torch.randperm(len(y),generator=generator).split(128):
            a,b = pairs[ids].T
            logits = network(x[a])-network(x[b])
            loss = nn.functional.binary_cross_entropy_with_logits(logits,y[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {k:v.detach().numpy().copy() for k,v in network.state_dict().items()}


def run(root):
    old = root/"resource-audition"
    out = root/"item-utility"
    out.mkdir(exist_ok=True)
    prior_path = root/"semantic-tree/inputs.json"
    prior = read(prior_path)
    data_file = root/"arctic-menu/original/run/agent/refit_data.joblib"
    require(sha(data_file) == prior["source"] and prior["selection"] == sha(old/"predictions.json")
            and sha(old/"confirmation.json") == prior["reference"], "Earlier election changed")
    training = joblib.load(data_file)
    metric = MetricSpec.from_payload(training)
    require(training["kind"]=="pair_text" and tuple(metric.labels)==(0,1) and metric.name=="accuracy",
            "Requires an ordered binary comparison with the evaluated class order")
    rows,y = training["X_train"],np.asarray(training["y_train"])
    g = sample_groups(TrainingTask(training["kind"],rows,y),"shared_text")
    split = read(old/"inputs.json")
    folds = [held for _,held in StratifiedGroupKFold(5,shuffle=True,random_state=split["seed"]).split(g,y,g)]
    fitting = np.sort(np.concatenate(folds[3:]))
    panels = dict(fitting=fitting,nomination=folds[0],confirmation=folds[1],audit=folds[2])
    require(all(fingerprint(v)==split["partitions"][k]["key"] for k,v in panels.items())
            and sum(len(set(g[ix])) for ix in panels.values())==len(set(g)),"Group boundaries changed")
    activate_resource_directory(root/"arctic-library")
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(root/"arctic-menu/fit_features"),
                      HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    models = {}
    for name in sorted(set(prior["elected"]) | set(prior["weights"])):
        path = old/name/"trial.joblib"
        require(sha(path)==read(path.parent/"receipt.json")["trial_sha256"],"Changed upstream fit")
        models[name]=joblib.load(path)["model"]
    providers = [embedding_function(models[name].named_steps["features"]) for name in prior["elected"]]
    require(len(providers)==2,"Expected two elected feature providers")
    lock(out/"inputs.json",dict(plan=sha(PLAN),prior=sha(prior_path),elected=prior["elected"],
         reference=prior["weights"],source=sha(data_file),partitions=split["partitions"]))
    model_file = out/"heads.joblib"
    seal = out/"trained.json"
    if not seal.exists():
        texts,pairs = documents(take(rows,fitting))
        matrix,transform = representation(texts,providers)
        heads = {}
        for name in ("linear","nonlinear"):
            with (out/(name+"_started.json")).open("x") as track:
                track.write('{"supervised_fits":1}\n')
            heads[name]=train(matrix,pairs,y[fitting],name=="nonlinear")
        joblib.dump(dict(transform=transform,weights=heads),model_file)
        lock(seal,dict(inputs=sha(out/"inputs.json"),model=sha(model_file),training_pairs=len(fitting),
             unique_items=len(texts),features=matrix.shape[1],supervised_fits=len(heads),unsupervised_pca_fits=2))
    require(sha(model_file)==read(seal)["model"],"Trained heads changed")
    saved = joblib.load(model_file)
    digests = read(old/"predictions.json")["components"]

    def evaluate(role,options):
        idx = panels[role]
        batch = take(rows,idx)
        texts,pairs = documents(batch)
        x,_ = representation(texts,providers,saved["transform"])
        probs = {}
        for name in sorted({m for m,_ in options if m}):
            network=ItemUtility(x.shape[1],name=="nonlinear")
            network.load_state_dict({k:torch.from_numpy(v) for k,v in saved["weights"][name].items()})
            with torch.inference_mode():
                utility=network.eval()(x).numpy().astype(float)
            a,b=pairs.numpy().T
            logits=utility[a]-utility[b]
            pos=expit(logits)
            require(np.allclose(pos+expit(utility[b]-utility[a]),1,rtol=0,atol=2e-15),
                    "Order reversal does not complement the probability")
            probs[name]=np.c_[1-pos,pos]
        members={}
        for name in prior["weights"]:
            model=models[name]
            cols=[list(model.classes_).index(c) for c in metric.labels]
            p=model.predict_proba(batch)[:,cols]
            require(hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()==digests[name][role],
                    "Protected probabilities changed")
            members[name]=p
        base=sum(w*members[m] for m,w in prior["weights"].items())
        right=base.argmax(1)==y[idx]
        unanimous=np.logical_and.reduce([p.argmax(1)!=y[idx] for p in members.values()])
        scores={}
        for name,w in options:
            p=base if not name else (1-w)*base+w*probs[name]
            correct=p.argmax(1)==y[idx]
            scores[name,w]=dict(**paired(correct,right),fixed_unanimous=int((correct&unanimous).sum()))
        return dict(rows=len(idx),protected=int(right.sum()),all_original_wrong=int(unanimous.sum())),scores

    options=[("",0.)]+[(name,w) for name in ("linear","nonlinear") for w in (.25,.5,1.)]
    first,measured=evaluate("confirmation",options)
    elected=min(options,key=lambda opt:(-measured[opt]["correct"],opt[1],opt[0]))
    lock(out/"choice.json",dict(seal=sha(seal),first=first,
         alternatives=[dict(name=n,share=w,**v) for (n,w),v in measured.items()],
         elected=list(elected)))
    follow=None
    advance=False
    if elected[1]>0:
        info,scores=evaluate("audit",[elected])
        follow=dict(**info,**scores[elected])
        pooled=(measured[elected]["net"]+follow["net"])/(first["rows"]+follow["rows"])
        advance=follow["net"]>0 and pooled>=.002
    result=dict(choice=sha(out/"choice.json"),elected=list(elected),first=first,nominee=measured[elected],
         follow=follow,advance=advance,supervised_fits=2,unsupervised_pca_fits=2,new_base_fits=0,
         model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(p.parse_args().root)
