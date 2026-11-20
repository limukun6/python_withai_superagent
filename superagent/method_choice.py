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
from .models import Candidate, _clean_categories
from .tabular_encoder import BACKBONES, SEMANTIC, LevelClassifier, MISSING, eligible_columns, MODEL as QWEN, FILES as QWEN_FILES
from .minilm_features import MiniLMEncoder, CANDIDATE as MINILM, MODEL as MINI_MODEL, FILES as MINI_FILES
from .task_budget import BudgetStateError
from .pretrained_choice import consider, budget, CANDIDATE, PARAMS

NEURAL = 'scratch_neural_classifier'
COOCCUR = 'training_category_cooccurrence'
DIRECT = 'direct_category_splits'
DIRECT_CONTEXT = 'direct_splits_with_category_context'
CROSSES = 'category_cross_C0.1'
TYPED = 'typed_category_features'
TYPED_DIRECT = 'typed_features_with_direct_splits'
SUPPORTED = {'categorical_tabular','numeric_tabular','mixed_tabular','text_tabular','pair_text','single_text','image'}
MODES = ('fixed', 'adaptive', 'paired', 'context-control', 'representation', 'typed-values')


def select_action(evidence, feasible, mode, used):
    """Return None to spend the slot on the ordinary next proposal."""
    if mode == 'context-control' and evidence['kind']=='categorical_tabular':
        priorities = [MINILM, COOCCUR]
        reason = 'Fixed comparison using frozen semantic features followed by training-only category co-occurrence.'
    elif mode == 'fixed':
        priorities = [MINILM, evidence['semantic'], NEURAL, COOCCUR]
        reason = 'Fixed ordering of feasible methods.'
    elif evidence['shared_errors'] < evidence['required_shared_errors']:
        return None, 'The ordinary models rarely all miss the same example.'
    else:
        aligned = [MINILM, evidence['semantic']] if evidence['short_latin_share']>=.8 else [evidence['semantic']]
        can_pool = evidence['low_support_shared_errors']>=max(4,math.ceil(.10*evidence['shared_errors']))
        repr_mode=mode in ('representation','typed-values')
        pair_aware = mode in ('paired','context-control','representation','typed-values') and evidence.get('primary_metric')=='accuracy'
        if mode=='typed-values' and not used and TYPED in feasible:
            priorities=[TYPED]
            reason='The table has repeated values with a consistent quantity or clock scale. Test their numerical features on the leading classifier.'
        elif mode=='typed-values' and used and used[-1]==TYPED:
            a,b=evidence['corrected'],evidence['lost']
            supported=pair_aware and a+b>=8 and a-b>=math.sqrt(a+b)
            priorities=[TYPED_DIRECT,DIRECT,CROSSES,COOCCUR] if supported else [DIRECT,CROSSES,COOCCUR]
            reason=('The numerical features show a paired gain. Retain them while testing direct splits.' if supported else
                    'The numerical feature gain was inconclusive. Test a different categorical representation.')
        elif repr_mode and not used and evidence['kind']=='categorical_tabular' and (DIRECT in feasible or CROSSES in feasible):
            priorities=[DIRECT,CROSSES]
            reason='The original models share errors on a bounded categorical table. Test a different way to represent repeated categories.'
        elif repr_mode and used and used[-1]==DIRECT:
            corrected,lost=evidence['corrected'],evidence['lost']
            supported=(pair_aware and corrected+lost>=8 and corrected-lost>=math.sqrt(corrected+lost))
            priorities=[DIRECT_CONTEXT,CROSSES] if supported else [CROSSES,COOCCUR]
            reason=('Direct category splits gained more correct rows than they lost. Try context on that backbone.' if supported else
                    'The improvement from direct splits was uncertain. Test a regularized model of category combinations.')
        elif (pair_aware and not used and can_pool and evidence['rare_enrichment']>=1.
                and any(name in feasible for name in aligned)):
            priorities = [*aligned, COOCCUR, NEURAL]
            reason = 'Shared errors disproportionately include lower-support names; probe category semantics before a new classifier.'
        elif pair_aware and used and evidence['discordant']>=max(8,.02*evidence['search_rows']):
            a,b=evidence['corrected'],evidence['lost']
            boundary=math.sqrt(a+b)
            if a-b>=boundary:
                return None, 'The challenger has more paired gains than losses by one discordant-count standard deviation; protect it and continue ordinary search.'
            if b-a>=boundary and used[-1] in (SEMANTIC,MINILM):
                priorities=[COOCCUR,NEURAL]
                reason='The last encoder introduced materially more paired losses; switch feature family.'
            else:
                priorities=[*aligned, COOCCUR, NEURAL]
                reason='The last candidate changes individual predictions; test an untried compatible representation.'
        elif not used and evidence['nonlinear_gain']>=.025:
            priorities = [NEURAL, *aligned, COOCCUR]
            reason = 'Nonlinear models lead linear models by at least 2.5 points; test learned nonlinear features.'
        elif used and evidence['previous_delta']>=evidence['margin']:
            return None, 'The last challenger improved the baseline by at least one point; retain it and use the remaining slot for ordinary search.'
        elif used and evidence['previous_delta'] < -evidence['margin'] and used[-1] in (MINILM,SEMANTIC):
            priorities = [NEURAL, COOCCUR]
            reason = 'The semantic challenger lost by over one point; explore a different source of features.'
        elif can_pool or evidence['kind'] in ('pair_text','single_text'):
            priorities = [*aligned, COOCCUR, NEURAL]
            reason = 'Shared errors persist on text or lower-support names, giving compatible frozen encoders first priority.'
        else:
            priorities = [COOCCUR, NEURAL]
            reason = 'Limited evidence for semantic transfer; learn features from training rows.'
    for name in priorities:
        if name in feasible and name not in used:
            if name!=priorities[0]:
                reason+=f' Earlier choices in this order were unavailable or already tried. Selected {name}.'
            return name, reason
    return None, 'No eligible untried alternative supported by the observations.'


def weights_available(config, key, files):
    if not config or not config.get(key):
        return False
    if os.environ.get('SUPERAGENT_OFFLINE')!='1':
        return True
    folder=Path(config[key])
    return all((folder/k).is_file() and (folder/k).stat().st_size==size for k,(_,_,size) in files.items())


def plan(kind, history, folder, payload, config, mode, protected_count=5):
    originals = shortlist(history[:protected_count],protected_count)
    if not originals:
        return dict(decision='ordinary',reason='No successful baseline.')
    incumbent = originals[0]
    labels = np.asarray(payload['y_search'])
    shared = np.ones(len(labels),bool)
    reference_predictions=None
    for row in originals:
        predictions = joblib.load(folder/f"trial_{row['trial']}.joblib")['search_predictions']
        if np.shape(predictions)!=labels.shape: raise ValueError('Misaligned prediction evidence')
        shared &= predictions!=labels
        if row['candidate']==incumbent['candidate']:reference_predictions=predictions
    x, held = payload['X_train'], payload['X_search']
    semantic = SEMANTIC if kind=='categorical_tabular' else CANDIDATE
    transfer = dict(decision='skip',reason='No compatible Qwen input and weights.')
    eligible = {}
    encoder_installed=all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','tokenizers','safetensors'))
    if kind in ('pair_text','single_text') and encoder_installed and config and config.get('weights'):
        transfer=consider(kind,history[:protected_count],folder,labels,config,train_features=x,search_features=held)
        if transfer['decision']=='try': eligible[semantic]=PARAMS
    n = len(payload['y_train'])
    feature_count = int(x.shape[1]) if hasattr(x,'shape') and len(x.shape)>1 else 1
    torch_available = importlib.util.find_spec('torch') is not None
    if kind in SUPPORTED and n>=256 and feature_count<=2048 and torch_available and len(np.unique(payload['y_train']))<=256:
        eligible[NEURAL] = dict(hidden=64,epochs=22,batch_size=256,seed=42)
    rare_fraction, short_latin_share, low_errors = 0., 0., 0
    if kind=='categorical_tabular':
        cols=eligible_columns(x)
        if mode in ('representation','typed-values'):
            pairs=feature_count*(feature_count-1)//2
            if 2<=feature_count<=32 and len(x)*(feature_count+pairs)<=6000000:
                from .challengers import cross_candidate
                eligible[CROSSES]=cross_candidate().params
            if incumbent['candidate'].startswith('catboost_depth'):
                small=sum(3<=len(np.unique(column))<=64 for column in _clean_categories(x).T)
                if small>=1:
                    eligible[DIRECT]=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'],one_hot_max_size=64)
                if mode=='typed-values':
                    from .typed_categories import column_types
                    typed=column_types(x)
                    if typed:
                        params=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'],
                                    search_column_types={str(i):u for i,u in typed.items()})
                        eligible[TYPED]=params
                        if DIRECT in eligible:eligible[TYPED_DIRECT]=dict(params,one_hot_max_size=64)
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
            if DIRECT in eligible:
                eligible[DIRECT_CONTEXT]=dict(base,one_hot_max_size=64)
            names={s for _,counts in cols for s in counts}
            short_latin_share=sum(s.isascii() and any(c.isalpha() for c in s) and len(s.split())<=16 for s in names)/len(names)
            low_errors=int((mask & shared).sum())
            if encoder_installed and weights_available(config,'minilm_weights',MINI_FILES):
                eligible[MINILM]=dict(base,model=MINI_MODEL)
            if encoder_installed and weights_available(config,'tabular_weights',QWEN_FILES):
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
    extra=history[protected_count:]
    corrected=lost=0
    if extra and extra[-1]['status']=='completed':
        after=joblib.load(folder/f"trial_{extra[-1]['trial']}.joblib")['search_predictions']
        if after.shape!=labels.shape:raise ValueError('Misaligned challenger predictions')
        corrected=int(np.sum((reference_predictions!=labels)&(after==labels)))
        lost=int(np.sum((reference_predictions==labels)&(after!=labels)))
    used=[r['candidate'] for r in extra]
    delta=(extra[-1]['search_score']-incumbent['search_score'] if extra and extra[-1]['status']=='completed' else -1.)
    ev=dict(rows=n, kind=kind, class_count=len(np.unique(payload['y_train'])), semantic=semantic,
            shared_errors=int(shared.sum()), required_shared_errors=max(8,math.ceil(.05*len(labels))),
            nonlinear_gain=max(nonlinear)-max(linear) if nonlinear and linear else 0.,rare_enrichment=rare_fraction,
            previous_delta=delta if extra else 0.,margin=.01,torch_available=torch_available,
            short_latin_share=short_latin_share,low_support_shared_errors=low_errors,primary_metric=incumbent['primary_metric'],
            corrected=corrected,lost=lost,discordant=corrected+lost,search_rows=len(labels))
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
        elif name in (TYPED,TYPED_DIRECT) and kind=='categorical_tabular':
            from .typed_categories import TypedCategoryClassifier
            model=TypedCategoryClassifier(params).fit(rows,labels)
        elif name==DIRECT and kind=='categorical_tabular':
            from .models import _tabular_catboost
            base=params['backbone_params']
            model=_tabular_catboost(labels,base['depth'],base['l2_leaf_reg'])
            model.set_params(one_hot_max_size=params['one_hot_max_size'])
            model.fit(_clean_categories(rows),labels,cat_features=list(range(rows.shape[1])))
            return model,model.predict(_clean_categories(search)).reshape(-1)
        elif name in (COOCCUR,DIRECT_CONTEXT,MINILM) and kind=='categorical_tabular':
            encoder=MiniLMEncoder(Path(folder)/'pretrained_runtime.json') if name==MINILM else Cooccurrence(rows,params['columns'])
            model=LevelClassifier(params).fit(rows,labels,encoder)
        else:
            raise ValueError('Unsupported method')
        return model,model.predict(search)
    return Candidate(name,params,fit)
