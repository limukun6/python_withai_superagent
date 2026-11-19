"""A bounded supervised MLP trained from random initialization; portable NumPy inference."""
import time
import statistics
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import SparseRandomProjection
from .models import _clean_categories
from .general_models import MixedPreprocessor, TextTablePreprocessor


def dense(x):
    return np.ascontiguousarray(x.toarray() if sparse.issparse(x) else x,dtype=np.float32)


def network(width,hidden,classes):
    from torch import nn
    return nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,classes))


def device_probe(width,hidden,classes,batch,steps):
    import torch
    devices=['cpu']
    if torch.cuda.is_available(): devices.append('cuda')
    elif torch.backends.mps.is_available(): devices.append('mps')
    estimates,measurements,errors={},{},{}
    for device in devices:
        try:
            net=network(width,hidden,classes).to(device)
            opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
            x=torch.ones((batch,width))
            y=torch.arange(batch)%classes
            def sync():
                if device=='cuda':torch.cuda.synchronize()
                elif device=='mps':torch.mps.synchronize()
            def step():
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(x.to(device)),y.to(device)).backward()
                opt.step()
            start=time.perf_counter()
            step()
            sync()
            warm=time.perf_counter()-start
            samples=[]
            for _ in range(5):
                start=time.perf_counter()
                for _ in range(3):step()
                sync()
                samples.append((time.perf_counter()-start)/3)
            median=statistics.median(samples)
            estimates[device]=warm+steps*median
            measurements[device]=dict(warm_seconds=warm,median_step_seconds=median,estimated_seconds=estimates[device])
        except (RuntimeError,MemoryError,NotImplementedError) as exc:
            errors[device]=str(exc)[:300]
    if not estimates:raise RuntimeError(f'No working training device: {errors}')
    best=min(estimates,key=estimates.get)
    if 'cpu' in estimates and estimates[best]>=.9*estimates['cpu']: best='cpu'
    return best,dict(selected=best,probes=measurements,probe_errors=errors,estimated_training_steps=steps,
                     rule='Prefer CPU unless a working accelerator is estimated at least 10% faster; 16 synthetic optimizer steps per probed device.')


class ScratchClassifier:
    def __init__(self,kind,params):
        self.kind,self.params=kind,params

    def features(self,rows,fit=False):
        k=self.kind
        if k=='categorical_tabular':
            raw=_clean_categories(rows)
            if fit:self.transformer=OneHotEncoder(handle_unknown='ignore',max_categories=64,dtype=np.float32)
        elif k in ('numeric_tabular','image'):
            raw=np.asarray(rows)
            if k=='image':
                raw=raw[:,::max(1,(raw.shape[1]+15)//16),::max(1,(raw.shape[2]+15)//16)].reshape(len(raw),-1)
            if fit:self.transformer=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler())
        elif k in ('mixed_tabular','text_tabular'):
            raw=rows
            if fit:self.transformer=MixedPreprocessor() if k=='mixed_tabular' else TextTablePreprocessor()
        else:
            if fit:self.transformer=HashingVectorizer(n_features=512,analyzer='char_wb',ngram_range=(2,4),dtype=np.float32)
            if k=='pair_text':
                return self.transformer.transform([r[0] for r in rows])-self.transformer.transform([r[1] for r in rows])
            raw=rows
        mat=self.transformer.fit_transform(raw) if fit else self.transformer.transform(raw)
        if fit:
            self.projection=SparseRandomProjection(n_components=256,random_state=42).fit(mat) if mat.shape[1]>512 else None
        return self.projection.transform(mat) if self.projection is not None else mat

    def fit(self,rows,labels):
        import torch
        x=self.features(rows,True)
        self.classes_,y=np.unique(labels,return_inverse=True)
        p=self.params
        torch.set_num_threads(min(4,torch.get_num_threads()))
        n=len(y)
        batch=min(p['batch_size'],n)
        device,record=device_probe(x.shape[1],p['hidden'],len(self.classes_),batch,((n+batch-1)//batch)*p['epochs'])
        torch.manual_seed(p['seed'])
        net=network(x.shape[1],p['hidden'],len(self.classes_)).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
        rng=np.random.default_rng(p['seed'])
        started=time.monotonic()
        for _ in range(p['epochs']):
            order=rng.permutation(n)
            for start in range(0,n,batch):
                indices=order[start:start+batch]
                batch_x=torch.from_numpy(dense(x[indices])).to(device)
                batch_y=torch.from_numpy(y[indices]).long().to(device)
                opt.zero_grad()
                loss=torch.nn.functional.cross_entropy(net(batch_x),batch_y)
                loss.backward()
                opt.step()
        self.weights=[(l.weight.detach().cpu().numpy().T.copy(),l.bias.detach().cpu().numpy().copy()) for l in net if isinstance(l,torch.nn.Linear)]
        if not all(np.isfinite(v).all() for pair in self.weights for v in pair):raise ValueError('Non-finite neural weights')
        self.training_metadata=dict(device_decision=record,epochs=p['epochs'],input_width=x.shape[1],
                                    initialized='random',pretrained_tokens=0,optimizer_seconds=time.monotonic()-started)
        return self

    def predict_proba(self,rows):
        x=self.features(rows)
        batches=[]
        for start in range(0,x.shape[0],512):
            h=dense(x[start:start+512])
            for w,b in self.weights[:-1]:h=np.maximum(h@w+b,0)
            logits=h@self.weights[-1][0]+self.weights[-1][1]
            ex=np.exp(logits-logits.max(axis=1,keepdims=True))
            batches.append(ex/ex.sum(axis=1,keepdims=True))
        return np.vstack(batches) if batches else np.empty((0,len(self.classes_)))

    def predict(self,rows):
        return self.classes_[self.predict_proba(rows).argmax(axis=1)]
