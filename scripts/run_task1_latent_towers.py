"""Training-only recommendation ablation on the remaining two group folds."""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
from torch import nn

from scripts.run_task1_profile_screen import digest, load_split, save
from superagent.joint_context import ContextTokenTree, blocks, tokens
from superagent.splits import sample_groups


class Tower(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n+1,8,padding_idx=0) for n in sizes[:-1]])
        self.project = nn.Sequential(nn.Linear(8*len(self.embeddings),32),nn.LayerNorm(32),nn.SiLU(),
                                     nn.Dropout(.12),nn.Linear(32,16))
        self.offset = nn.Embedding(sizes[-1]+1,16,padding_idx=0)
        nn.init.normal_(self.offset.weight,std=.04)
        with torch.no_grad():
            self.offset.weight[0].zero_()

    def forward(self, x):
        attrs = torch.cat([layer(x[:,i]) for i,layer in enumerate(self.embeddings)],dim=1)
        ids = x[:,-1]
        if self.training:
            ids = ids.masked_fill(torch.rand(len(ids)) < .18,0)
        return self.project(attrs) + self.offset(ids)


class ResponseNetwork(nn.Module):
    def __init__(self, profile_sizes, context_sizes):
        super().__init__()
        self.profile = Tower(profile_sizes)
        self.context = Tower(context_sizes)
        self.joint = nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Dropout(.15),nn.Linear(32,1))

    def forward(self,p,c):
        a,b = self.profile(p),self.context(c)
        cross = a*b
        return self.joint(torch.cat((a,b,cross),dim=1)).flatten() + cross.sum(dim=1)/4


def encoded(x,q,cols):
    def table(a):
        fields = a[:,cols]
        return np.column_stack((fields,tokens(fields)))
    enc = OrdinalEncoder(dtype=np.int64,handle_unknown='use_encoded_value',unknown_value=-1)
    train = enc.fit_transform(table(x))
    return (torch.from_numpy(train+1),torch.from_numpy(enc.transform(table(q))+1),
            [len(levels) for levels in enc.categories_])


def predict_neural(x,y,q):
    profile,context = blocks(x)
    p,qp,ps = encoded(x,q,profile)
    c,qc,cs = encoded(x,q,context)
    torch.manual_seed(26471)
    model = ResponseNetwork(ps,cs)
    opt = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.03)
    target = torch.from_numpy(y.astype(np.float32))
    for epoch in range(24):
        if epoch == 16:
            for group in opt.param_groups:
                group['lr'] = .0007
        for ids in torch.randperm(len(y)).split(256):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(p[ids],c[ids]),target[ids])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(qp,qc)).numpy()


def run(task,out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    start = time.monotonic()
    x,y,hashes = load_split(task,'train')
    groups = sample_groups(SimpleNamespace(kind='categorical_tabular',X_train=x))
    splits = list(StratifiedGroupKFold(5,shuffle=True,random_state=18643).split(x,y,groups))
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__),root/'superagent/joint_context.py',root/'superagent/profile_history.py',
               root/'docs/batches/TASK1_LATENT_TOWERS_PLAN_20261127_zh.md']
    save(out/'protocol.json',dict(members=hashes,sources={p.name:digest(p.read_bytes()) for p in sources},
        torch=torch.__version__, seed=18643, folds=[3,4], epochs=24, fit_cap=4,device='cpu',
        held_sha=[digest(h.astype('<i8').tobytes()) for _,h in splits],validation_opened=False,test_opened=False))
    rows = []
    fits = 0
    try:
        for fold,(train,held) in enumerate(splits[3:],3):
            row = dict(fold=fold,held=len(held))
            stages = []
            for name in ('tree','neural','blend'):
                t = time.monotonic()
                if name!='blend':
                    fits+=1
                    save(out/'last_attempt.json',dict(fold=fold,name=name,admitted=fits))
                if name=='tree':
                    model = ContextTokenTree(dict(pair_limit=3)).fit(x[train],y[train])
                    p = model.predict_proba(x[held])[:,1]
                elif name=='neural':
                    p = predict_neural(x[train],y[train],x[held])
                else:
                    p = np.mean(stages,axis=0)
                if not np.isfinite(p).all():
                    raise ValueError('Nonfinite prediction')
                stages.append(p)
                correct = (p>.5)==y[held]
                base = (stages[0]>.5)==y[held]
                row[name] = dict(correct=int(correct.sum()),accuracy=float(correct.mean()),
                    delta_pp=float(100*(correct.mean()-base.mean())),fixed=int((correct&~base).sum()),
                    broken=int((~correct&base).sum()),seconds=round(time.monotonic()-t,3),
                    prediction_sha=digest((p>.5).astype('<i8').tobytes()))
                save(out/'results.json',rows+[row])
                print(fold,name,row[name]['accuracy'],flush=True)
            rows.append(row)
        summary = {}
        for name in ('neural','blend'):
            delta = [r[name]['delta_pp'] for r in rows]
            summary[name] = dict(mean_delta_pp=float(np.mean(delta)),promote=bool(min(delta)>0 and np.mean(delta)>=1))
        save(out/'completion.json',dict(status='complete',fits=fits,api_calls=0,tokens=0,course_yuan=0,
                                        seconds=round(time.monotonic()-start,3),summary=summary))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),admitted_fits=fits,
                                        seconds=round(time.monotonic()-start,3),api_calls=0))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.task,args.output)
