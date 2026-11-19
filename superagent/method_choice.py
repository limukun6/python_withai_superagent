"""Allocate two experiments after the ordinary five-step search."""
from pathlib import Path
import importlib.util
import math
import os
import json
import hashlib

import joblib
import numpy as np
from .confirmation import shortlist
from .models import Candidate
from .tabular_encoder import BACKBONES, SEMANTIC, LevelClassifier, MISSING, eligible_columns, MODEL as QWEN, FILES as QWEN_FILES
from .minilm_features import MiniLMEncoder, CANDIDATE as MINILM, MODEL as MINI_MODEL, FILES as MINI_FILES
from .task_budget import BudgetStateError
from .pretrained_choice import consider, budget, CANDIDATE, PARAMS

NEURAL = 'scratch_neural_classifier'
COOCCUR = 'training_category_cooccurrence'
SUPPORTED = {'categorical_tabular','numeric_tabular','mixed_tabular','text_tabular','pair_text','single_text','image'}


def select_action(evidence, feasible, mode, used):
    """Return None to spend the slot on the ordinary next proposal."""
    if mode == 'fixed':
        priorities = [MINILM, evidence['semantic'], NEURAL, COOCCUR]
        reason = 'Fixed ordering of feasible methods.'
    elif evidence['shared_errors'] < evidence['required_shared_errors']:
        return None, 'The ordinary models rarely all miss the same example.'
    else:
        aligned = [MINILM, evidence['semantic']] if evidence['short_latin_share']>=.8 else [evidence['semantic']]
        can_pool = evidence['low_support_shared_errors']>=max(4,math.ceil(.10*evidence['shared_errors']))
        if not used and evidence['nonlinear_gain']>=.025:
            priorities = [NEURAL, *aligned, COOCCUR]
            reason = 'Nonlinear models lead linear models by at least 2.5 points; test learned nonlinear features.'
        elif used and evidence['previous_delta']>=evidence['margin']:
            return None, 'The last challenger improved the baseline by at least one point; retain it and use the remaining slot for ordinary search.'
        elif used and evidence['previous_delta'] < -evidence['margin'] and used[-1] in (MINILM,SEMANTIC):
            priorities = [NEURAL, COOCCUR]
            reason = 'The semantic challenger lost by over one point; explore a different source of features.'
        elif can_pool or evidence['kind'] in ('pair_text','single_text'):
            priorities = [*aligned, COOCCUR, NEURAL]
            reason = 'Shared errors persist on text or lower-support names; try a compatible frozen encoder.'
        else:
            priorities = [COOCCUR, NEURAL]
            reason = 'Limited evidence for semantic transfer; learn features from training rows.'
    for name in priorities:
        if name in feasible and name not in used:
            return name, reason
    return None, 'No eligible untried alternative supported by the observations.'


def weights_available(config, key, files):
    if not config or not config.get(key):
        return False
    if os.environ.get('SUPERAGENT_OFFLINE')!='1':
        return True
    folder=Path(config[key])
    return all((folder/k).is_file() and (folder/k).stat().st_size==size for k,(_,_,size) in files.items())


def plan(kind, history, folder, payload, config, mode):
    originals = shortlist(history[:5],5)
    if not originals:
        return dict(decision='ordinary',reason='No successful baseline.')
    incumbent = originals[0]
    labels = np.asarray(payload['y_search'])
    shared = np.ones(len(labels),bool)
    for row in originals:
        predictions = joblib.load(folder/f"trial_{row['trial']}.joblib")['search_predictions']
        if np.shape(predictions)!=labels.shape: raise ValueError('Misaligned prediction evidence')
        shared &= predictions!=labels
    x, held = payload['X_train'], payload['X_search']
    semantic = SEMANTIC if kind=='categorical_tabular' else CANDIDATE
    transfer = dict(decision='skip',reason='No compatible Qwen input and weights.')
    eligible = {}
    if kind in ('pair_text','single_text') and config and config.get('weights'):
        transfer=consider(kind,history[:5],folder,labels,config,train_features=x,search_features=held)
        if transfer['decision']=='try': eligible[semantic]=PARAMS
    n = len(payload['y_train'])
    feature_count = int(x.shape[1]) if hasattr(x,'shape') and len(x.shape)>1 else 1
    torch_available = importlib.util.find_spec('torch') is not None
    if kind in SUPPORTED and n>=256 and feature_count<=2048 and torch_available and len(np.unique(payload['y_train']))<=256:
        eligible[NEURAL] = dict(hidden=64,epochs=22,batch_size=256,seed=42)
    rare_fraction, short_latin_share, low_errors = 0., 0., 0
    if kind=='categorical_tabular':
        cols=eligible_columns(x)
        mask=np.zeros(len(labels),bool)
        for i,counts in cols:
            median=np.median(list(counts.values()))
            maximum=max(counts.values())
            names = [str(v).strip() for v in held[:,i]]
            mask |= np.array([s.casefold() not in MISSING and (s not in counts or (counts[s]<=median and counts[s]<maximum)) for s in names])
        rare_fraction=(float(np.mean(mask[shared]))/float(np.mean(mask))) if shared.any() and mask.any() else 0.
        if cols and incumbent['candidate'].startswith(BACKBONES):
            base=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'],
                columns=[i for i,_ in cols],assignment='semantic',dimensions_per_column=3)
            eligible[COOCCUR]=base
            names={s for _,counts in cols for s in counts}
            short_latin_share=sum(s.isascii() and any(c.isalpha() for c in s) and len(s.split())<=16 for s in names)/len(names)
            low_errors=int((mask & shared).sum())
            if torch_available and weights_available(config,'minilm_weights',MINI_FILES):
                eligible[MINILM]=dict(base,model=MINI_MODEL)
            if torch_available and weights_available(config,'tabular_weights',QWEN_FILES):
                allowance=sum(min(64,len(s.encode())+2) for s in names)
                try:
                    budget(config,allowance,model_id=QWEN)
                    eligible[SEMANTIC]=dict(base,model=QWEN)
                    transfer=dict(decision='try',candidate=SEMANTIC,params=eligible[SEMANTIC],
                                  reason='Verified affordable Qwen features for readable levels.')
                except BudgetStateError as exc:
                    transfer=dict(decision='skip',reason=str(exc))
    linear=[r['search_score'] for r in originals if any(p in r['candidate'] for p in ('logistic','word','char'))]
    nonlinear=[r['search_score'] for r in originals if any(p in r['candidate'] for p in ('trees','catboost','rbf'))]
    extra=history[5:]
    used=[r['candidate'] for r in extra]
    delta=(extra[-1]['search_score']-incumbent['search_score'] if extra and extra[-1]['status']=='completed' else -1.)
    ev=dict(rows=n, kind=kind, class_count=len(np.unique(payload['y_train'])), semantic=semantic,
            shared_errors=int(shared.sum()), required_shared_errors=max(8,math.ceil(.05*len(labels))),
            nonlinear_gain=max(nonlinear)-max(linear) if nonlinear and linear else 0.,rare_enrichment=rare_fraction,
            previous_delta=delta if extra else 0.,margin=.01,torch_available=torch_available,
            short_latin_share=short_latin_share,low_support_shared_errors=low_errors)
    name,reason=select_action(ev,eligible,mode,used)
    return dict(decision='try' if name else 'ordinary',candidate=name,params=eligible.get(name),reason=reason,
                evidence=ev,previous_methods=used,eligible=list(eligible),transfer=transfer,
                protected_incumbent=incumbent['candidate'],mode=mode)


class Cooccurrence:
    """Fold-trained hashed context counts for each level, excluding its own column."""
    def __init__(self, rows, columns):
        self.vectors={}
        raw=np.asarray(rows).astype(str)
        ctx=np.empty(raw.shape,dtype=np.int64)
        for i in range(raw.shape[1]):
            distinct,inv=np.unique(raw[:,i],return_inverse=True)
            bins=[int.from_bytes(hashlib.sha256(f'{i}:{s}'.encode()).digest()[:4],'big')%256 for s in distinct]
            ctx[:,i]=np.asarray(bins)[inv]
        for i in columns:
            for s in np.unique(raw[:,i]):
                other=np.delete(ctx[raw[:,i]==s],i,axis=1)
                key=s.strip()
                counts=np.bincount(other.ravel(),minlength=256)
                self.vectors[key]=self.vectors.get(key,0)+counts

    def encode(self,names):
        a=np.stack([self.vectors[s] for s in names])
        return np.sqrt(a/np.maximum(1,a.sum(axis=1,keepdims=True))).astype('float32')


def candidate(kind, name, folder):
    proposal=json.loads((Path(folder)/f'method_{name}.json').read_text())
    params=proposal['params']
    if proposal['candidate']!=name or proposal['decision']!='try':raise ValueError('Unregistered method')
    def fit(rows,labels,search):
        if name==NEURAL:
            from .scratch_classifier import ScratchClassifier
            model=ScratchClassifier(kind,params).fit(rows,labels)
        elif name in (COOCCUR,MINILM) and kind=='categorical_tabular':
            encoder=MiniLMEncoder(Path(folder)/'pretrained_runtime.json') if name==MINILM else Cooccurrence(rows,params['columns'])
            model=LevelClassifier(params).fit(rows,labels,encoder)
        else:
            raise ValueError('Unsupported method')
        return model,model.predict(search)
    return Candidate(name,params,fit)
