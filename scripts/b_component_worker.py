"""Isolated training and inference for a budgeted binary preference portfolio."""
import argparse
import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.sparse import vstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import torch


sys.path.insert(0,os.environ['SUPERAGENT_B_COMPONENT_RUNTIME'])
from superagent.minilm_features import DEEP_FILES, FILES, DeeperMiniLMTextClassifier, MiniLMTextClassifier
from superagent.pair_attention import PairAttentionEncoder
from superagent.pretrained_choice import digest
from tools.milestone_validation import CachedSentenceBank
from tools.preference_adapter import HiddenBank, PreferenceAdapter, encode_items, predict, train_adapter
from tools.token_pooling_study import logits
from tools.token_value_head import text_key


COST={'word':1,'base':1,'deep':1,'plain_equal':2,'stream_adapter':4,'adapter':4,
      'stream_hybrid':5,'full_hybrid':5}


def write(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2)


def check_weights():
    shallow=Path(os.environ['SUPERAGENT_MINILM_WEIGHTS'])
    deep=Path(os.environ['SUPERAGENT_DEEP_MINILM_WEIGHTS'])
    for folder,files in [(shallow,FILES),(deep,DEEP_FILES)]:
        if not all(digest(folder/name)==sha for name,(_,sha,_) in files.items()):
            raise ValueError('Pinned downstream encoder does not match')
    return shallow,deep


class FitMeter:
    def __init__(self,folder,cap,wall,timeout):
        self.path=folder/'component_fits.jsonl'
        self.path.touch(exist_ok=False)
        self.cap=cap
        self.deadline=time.monotonic()+wall
        self.timeout=timeout
        self.attempts=0
        self.current=None
        logistic_fit=LogisticRegression.fit
        adam_step=torch.optim.AdamW.step
        def counted_fit(model,*args,**kwargs):
            self.observe('logistic',model)
            return logistic_fit(model,*args,**kwargs)
        def counted_step(optimizer,*args,**kwargs):
            self.observe('adapter',optimizer,repeated=True)
            return adam_step(optimizer,*args,**kwargs)
        LogisticRegression.fit=counted_fit
        torch.optim.AdamW.step=counted_step

    def add(self,event):
        with self.path.open('a') as f:
            f.write(json.dumps(event)+'\n')
            f.flush()
            os.fsync(f.fileno())

    def observe(self,kind,obj,repeated=False):
        if self.current is None or self.current['kind']!=kind:
            raise RuntimeError('Unadmitted supervised optimization')
        if self.current['object'] is not None:
            if repeated and self.current['object'] is obj:
                return
            raise RuntimeError('More supervised fits than admitted')
        self.current['object']=obj
        self.add(dict(event='physical_fit',number=self.attempts,kind=kind))

    def run(self,name,kind,fn,reserve=0):
        if self.current is not None or self.attempts+1+reserve>self.cap:
            raise RuntimeError('Insufficient fitting allowance')
        seconds=min(self.timeout,self.deadline-time.monotonic())
        if seconds<=0:
            raise TimeoutError('Path deadline reached')
        self.attempts+=1
        self.current={'kind':kind,'object':None}
        self.add(dict(event='attempt',number=self.attempts,name=name,reserve=reserve))
        def expired(*_):
            raise TimeoutError('Supervised work exceeded its deadline')
        if hasattr(signal,'setitimer'):
            signal.signal(signal.SIGALRM,expired)
            signal.setitimer(signal.ITIMER_REAL,seconds)
        try:
            result=fn()
            if self.current['object'] is None:
                raise RuntimeError('No matching supervised operation observed')
            self.add(dict(event='completed',number=self.attempts,name=name))
            return result
        except BaseException as exc:
            self.add(dict(event='failed',number=self.attempts,name=name,error=type(exc).__name__))
            raise
        finally:
            if hasattr(signal,'setitimer'):
                signal.setitimer(signal.ITIMER_REAL,0)
            self.current=None


def word_features(vectorizer,rows):
    return vectorizer.transform([a for a,_ in rows])-vectorizer.transform([b for _,b in rows])


def fit_word(rows,y):
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2,max_features=40000)
    vec.fit([t for pair in rows for t in pair])
    x=word_features(vec,rows)
    return vec,LogisticRegression(C=.05,max_iter=2000,random_state=42).fit(vstack([x,-x]),np.r_[y,1-y])


class Features:
    def __init__(self,rows,scratch,weights,need_base=True):
        self.shallow,self.deep=weights
        self.bank=self.layer=None
        self.usage={}
        if need_base:
            texts=sorted({t for pair in rows for t in pair},key=text_key)
            hidden=scratch/'hidden.npz'
            self.usage['L6']=encode_items(texts,self.shallow,hidden)
            self.bank=HiddenBank(hidden)
        self.deep_runtime=scratch/'deep'/'runtime.json'
        self.deep_runtime.parent.mkdir()
        write(self.deep_runtime,dict(deep_minilm_weights=str(self.deep.resolve())))

    def last_layer(self):
        if self.layer is None:
            self.layer=PairAttentionEncoder(self.shallow).model.encoder.layer[-1]
        return self.layer

    def bind(self,model,kind):
        if kind=='deep':
            model.runtime=str(self.deep_runtime)
            model.encoder=None
        else:
            model.encoder=CachedSentenceBank(self.bank)
        return model


def new_head(rows,y,features,kind):
    cls=DeeperMiniLMTextClassifier if kind=='deep' else MiniLMTextClassifier
    model=features.bind(cls('pair_text',str(features.deep_runtime),False),kind)
    return model.fit(rows,y)


def fit_residual(rows,y,groups,features,meter,stage,reserve):
    offsets=np.full(len(y),np.nan)
    coefs=np.full((len(y),384),np.nan,dtype=np.float32)
    folds=[]
    partition=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=78101)
    for i,(fit,held) in enumerate(partition.split(np.zeros(len(y)),y,groups)):
        if set(groups[fit])&set(groups[held]):
            raise ValueError('Crossfit groups overlap')
        head=meter.run(f'{stage}/offset_{i+1}','logistic',
              lambda:new_head([rows[j] for j in fit],y[fit],features,'base'),reserve)
        offsets[held]=logits(head,[rows[j] for j in held])
        coefs[held]=head.classifier.coef_[0,-384:]
        folds.append(dict(fitted=len(fit),held=len(held),disjoint_groups=True))
    if not np.isfinite(offsets).all() or not np.isfinite(coefs).all():
        raise ValueError('Incomplete out-of-fit observations')
    learner,loss=meter.run(stage+'/rank8','adapter',lambda:train_adapter(
        'inside',features.bank,features.bank.pairs(rows),offsets,coefs,y,features.last_layer()),reserve)
    return dict(weights=learner.state_dict(),folds=folds,loss=loss)


def on_rows(components,rows,features,method,threshold):
    if method=='word':
        vec,clf=components['word']
        return clf.predict_proba(word_features(vec,rows))[:,1]
    if method=='deep':
        return features.bind(components['deep'],'deep').predict_proba(rows)[:,1]
    base=features.bind(components['base'],'base')
    z=logits(base,rows)
    p=expit(z)
    if method=='base':
        return p
    if method=='plain_equal':
        return (p+on_rows(components,rows,features,'deep',threshold))/2
    selected=(np.abs(z-base.classifier.intercept_[0])<=threshold if method.startswith('stream_')
              else np.ones(len(rows),dtype=bool))
    if selected.any():
        opened=[row for row,use in zip(rows,selected,strict=True) if use]
        learner=PreferenceAdapter('inside').eval()
        learner.load_state_dict(components['residual']['weights'])
        delta=predict(learner,features.bank,features.bank.pairs(opened),base.classifier.coef_[0,-384:],features.last_layer())
        q=expit(z[selected]+delta)
        if method in ('stream_hybrid','full_hybrid'):
            q=(q+on_rows(components,opened,features,'deep',threshold))/2
        p[selected]=q
    return p


def train(args):
    data=joblib.load(args.data)
    rows=data['rows']
    y=np.asarray(data['labels'],dtype=int)
    groups=np.array(data['groups'])
    fit_ids,held_ids=map(np.asarray,(data['fit_ids'],data['held_ids']))
    if set(np.unique(y))!={0,1} or set(groups[fit_ids])&set(groups[held_ids]):
        raise ValueError('Unexpected training domain or overlapping groups')
    probe=np.array([int(g[:2],16)<128 for g in groups[held_ids]])
    if not probe.any() or not (~probe).any():
        raise ValueError('Need two disjoint groups for a paired review')
    folder=args.out
    meter=FitMeter(folder,args.limit,args.wall,args.timeout)
    actions=[]
    def record(event,**items):
        actions.append(dict(action=event,**items))
        with (folder/'actions.jsonl').open('a') as f:
            f.write(json.dumps(actions[-1])+'\n')
    x=[rows[i] for i in fit_ids]
    query=[rows[i] for i in held_ids]
    yf,truth=y[fit_ids],y[held_ids]
    weights=check_weights()
    with tempfile.TemporaryDirectory() as t:
        features=Features(rows,Path(t),weights)
        components={}
        predictions={}
        threshold=0.
        def observe(name):
            pred=on_rows(components,query,features,name,threshold)>.5
            predictions[name]=pred
            record('measure',candidate=name,probe_correct=int((pred[probe]==truth[probe]).sum()),
                   probe_rows=int(probe.sum()),refit_fits=COST[name])
        def component_gate(names,increment,refit):
            shared=np.logical_and.reduce([predictions[k][probe]!=truth[probe] for k in names])
            required=max(8,math.ceil(.05*int(probe.sum())))
            admit=int(shared.sum())>=required and meter.attempts+increment+refit<=args.limit
            record('consider_expansion',observed=names,shared_probe_errors=int(shared.sum()),
                   required=required,additional_fits=increment,reserve=refit,open=bool(admit))
            return admit
        components['word']=meter.run('search/word','logistic',lambda:fit_word(x,yf),1)
        observe('word')
        components['base']=meter.run('search/base','logistic',lambda:new_head(x,yf,features,'base'),1)
        z=logits(components['base'],[r for r,m in zip(query,probe,strict=True) if m])
        threshold=float(np.quantile(np.abs(z-components['base'].classifier.intercept_[0]),.25,method='inverted_cdf'))
        observe('base')
        if component_gate(['word','base'],1,2):
            components['deep']=meter.run('search/deep','logistic',lambda:new_head(x,yf,features,'deep'),2)
            observe('deep')
            observe('plain_equal')
            if component_gate(['word','base','deep','plain_equal'],3,4):
                components['residual']=fit_residual(x,yf,groups[fit_ids],features,meter,'search',4)
                for name in ('stream_adapter','adapter','stream_hybrid','full_hybrid'):
                    if meter.attempts+COST[name]<=args.limit:
                        observe(name)
        grades={k:dict(probe=int((p[probe]==truth[probe]).sum()),
                       confirmation=int((p[~probe]==truth[~probe]).sum()),total=int((p==truth).sum()))
                for k,p in predictions.items()}
        finalists=[name for name in predictions if meter.attempts+COST[name]<=args.limit]
        selected=max(finalists,key=lambda k:(grades[k]['total'],-COST[k],-list(COST).index(k)))
        record('stop_and_refit',selected=selected,scores=grades,spent=meter.attempts,
               reserved_for_selected=COST[selected],threshold=threshold)
        write(folder/'inner_evidence.json',dict(held_row_ids=held_ids.tolist(),probe=probe.tolist(),
              observed_predictions={k:v.astype(int).tolist() for k,v in predictions.items()},
              scores=grades,selected=selected,threshold=threshold))
        final={}
        if selected=='word':
            final['word']=meter.run('final/word','logistic',lambda:fit_word(rows,y))
        else:
            if selected!='deep':
                final['base']=meter.run('final/base','logistic',lambda:new_head(rows,y,features,'base'))
            if selected in ('deep','plain_equal','full_hybrid','stream_hybrid'):
                final['deep']=meter.run('final/deep','logistic',lambda:new_head(rows,y,features,'deep'))
            if selected in ('adapter','stream_adapter','full_hybrid','stream_hybrid'):
                final['residual']=fit_residual(rows,y,groups,features,meter,'final',0)
        joblib.dump(dict(components=final,selected=selected,threshold=threshold),folder/'component_payload.joblib',compress=3)
        deep_log=features.deep_runtime.parent/'downstream_usage.jsonl'
        events=[json.loads(v) for v in deep_log.read_text().splitlines()] if deep_log.exists() else []
        write(folder/'worker_result.json',dict(selected=selected,actual_fits=meter.attempts,training_rows=len(y),
              inner_train_rows=len(fit_ids),probe_rows=int(probe.sum()),confirm_rows=int((~probe).sum()),
              scores=grades,L6=features.usage['L6'],L12_input_tokens=sum(e['input_tokens'] for e in events),
              model_api_calls=0,payload_sha256=digest(folder/'component_payload.joblib')))


def inference(args):
    payload=joblib.load(args.data)
    rows=json.loads(args.rows.read_text())
    with tempfile.TemporaryDirectory() as t:
        features=Features(rows,Path(t),check_weights(),need_base=payload['selected'] not in ('word','deep'))
        p=on_rows(payload['components'],rows,features,payload['selected'],payload['threshold'])
    if len(p)!=len(rows) or not np.isfinite(p).all():
        raise ValueError('Invalid probabilities')
    np.save(args.out,p,allow_pickle=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['train','predict','preflight'])
    for key in ('data','out','rows'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--limit',type=int,default=11)
    p.add_argument('--wall',type=float,default=1200)
    p.add_argument('--timeout',type=float,default=300)
    a=p.parse_args()
    torch.set_num_threads(3)
    if a.phase=='train':
        train(a)
    elif a.phase=='predict':
        inference(a)
    else:
        check_weights()
        print('Pinned encoders available')
