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
RESIDUAL = 'category_conditional_distributions'
MARGINAL = 'category_marginal_frequency'
PERMUTED_RESIDUAL = 'permuted_category_distributions'
RESIDUAL_DIRECT = 'category_distributions_with_direct_splits'
HORIZON = 'feedback_boosting_iterations'
SHRINKAGE = 'boosting_smaller_steps'
HIST = 'native_histogram_categories'
HIST_ADJUSTED = 'feedback_histogram_categories'
PAIR_POSTERIOR = 'cross_fitted_category_pair_effects'
SINGLE_POSTERIOR = 'cross_fitted_category_single_effects'
TEXT_MINILM = 'minilm_text_with_lexical_features'
TEXT_MINILM_ONLY = 'minilm_text_dense_only'
PAIR_OPPONENT = 'minilm_opponent_interactions'
PAIR_ITEM = 'minilm_single_item_interactions'
TEXT_DEEP = 'minilm_l12_text_with_lexical_features'
TEXT_REGULARIZED = 'minilm_text_feedback_regularization'
SUPPORTED = {'categorical_tabular','numeric_tabular','mixed_tabular','text_tabular','pair_text','single_text','image'}
MODES = ('fixed', 'adaptive', 'paired', 'context-control', 'representation', 'typed-values','distribution','distribution-control','boost-curve','histogram','anchored','anchored-control','posteriors','text-encoder','text-encoder-control','auto','opponent','item-control','encoder-depth','encoder-depth-control','text-regularization','encoder-mixture')


def resolved_mode(mode, feasible):
    if mode=='encoder-mixture':
        return 'encoder-depth' if TEXT_MINILM in feasible else 'paired'
    if mode != 'auto':
        return mode
    return 'text-encoder' if TEXT_MINILM in feasible else 'paired'


def select_action(evidence, feasible, mode, used):
    """Return None to spend the slot on the ordinary next proposal."""
    mode=resolved_mode(mode,feasible)
    if mode=='encoder-depth-control' and TEXT_MINILM in feasible:
        priorities=[TEXT_MINILM,TEXT_DEEP]
        reason='The matched depth control tests the smaller and then deeper encoder when each fits the resource limit.'
    elif mode=='text-encoder-control' and TEXT_MINILM in feasible:
        priorities=[TEXT_MINILM,TEXT_MINILM_ONLY]
        reason='The fixed text control compares lexical plus frozen sentence features and frozen features alone.'
    elif mode=='anchored-control' and evidence['kind']=='categorical_tabular' and COOCCUR in feasible:
        priorities=[COOCCUR,DIRECT]
        reason='The matched control tests training-only co-occurrence first, then direct category splits.'
    elif mode=='distribution-control' and evidence['kind']=='categorical_tabular' and PERMUTED_RESIDUAL in feasible:
        priorities=[PERMUTED_RESIDUAL,MARGINAL]
        reason='Matched diagnostic using shuffled category coordinates and marginal frequencies.'
    elif mode == 'context-control' and evidence['kind']=='categorical_tabular':
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
        repr_mode=mode in ('representation','typed-values','distribution','distribution-control')
        pair_aware = (repr_mode or mode in ('paired','context-control','boost-curve','histogram','anchored','anchored-control','posteriors')) and evidence.get('primary_metric')=='accuracy'
        if mode=='text-regularization' and used and used[-1]==TEXT_MINILM:
            if TEXT_REGULARIZED in feasible:
                priorities=[TEXT_REGULARIZED]
                reason='Grouped search probabilities support testing a different regularization strength with the same training features.'
            else:
                return None,'Search confidence gives no clear regularization direction. Continue ordinary search.'
        elif mode=='encoder-depth' and used and used[-1]==TEXT_MINILM:
            gain,loss=evidence['corrected'],evidence['lost']
            justified=(evidence['primary_metric']=='accuracy' and gain+loss>=8 and gain-loss>=math.sqrt(gain+loss)
                       and evidence['residual_shared_errors']>=max(12,.05*evidence['search_rows']))
            if justified and TEXT_DEEP in feasible:
                priorities=[TEXT_DEEP]
                reason='The smaller frozen model gained paired decisions and shared errors remain. Test deeper public weights with the same classifier within the estimated trial time.'
            else:
                return None,'Evidence or the time and weight allowance does not justify a deeper encoder. Continue ordinary search.'
        elif mode in ('opponent','item-control') and used and used[-1]==TEXT_MINILM and PAIR_OPPONENT in feasible:
            gain,loss=evidence['corrected'],evidence['lost']
            enough=evidence['residual_shared_errors']>=max(12,.05*evidence['search_rows'])
            if not enough or loss-gain>math.sqrt(max(1,gain+loss)):
                return None,'The hybrid lacks either shared residual errors or competitive accuracy. Continue ordinary search.'
            priorities=[PAIR_OPPONENT if mode=='opponent' else PAIR_ITEM]
            reason=('The text models share residual errors. Test whether the two opposing texts interact.'
                    if mode=='opponent' else 'With the same residual evidence, test nonlinear features of each item separately.')
        elif mode in ('text-encoder','opponent','item-control','encoder-depth','text-regularization') and not used and TEXT_MINILM in feasible:
            priorities=[TEXT_MINILM]
            reason='Several text classifiers miss the same rows and the available frozen encoder matches the observed writing system. Test sentence features beside word and character features.'
        elif mode in ('text-encoder','opponent','item-control') and used and used[-1]==TEXT_MINILM:
            gain,loss=evidence['corrected'],evidence['lost']
            supported=evidence['primary_metric']=='accuracy' and gain+loss>=8 and gain-loss>=math.sqrt(gain+loss)
            if not supported:
                return None, 'The hybrid lacks a clear paired accuracy gain. Retain it for confirmation and spend the last trial on an ordinary alternative.'
            priorities=[TEXT_MINILM_ONLY]
            reason='The hybrid improves paired predictions. Test how much a classifier using sentence features alone can recover.'
        elif mode=='posteriors' and not used and PAIR_POSTERIOR in feasible:
            priorities=[PAIR_POSTERIOR]
            reason='The available training groups support repeated category combinations. Test label effects estimated exclusively from other groups.'
        elif mode=='posteriors' and used and used[-1]==PAIR_POSTERIOR:
            gain,loss=evidence['corrected'],evidence['lost']
            supported=pair_aware and gain+loss>=8 and gain-loss>=math.sqrt(gain+loss)
            priorities=[SINGLE_POSTERIOR,COOCCUR] if supported else [COOCCUR,SINGLE_POSTERIOR]
            reason=('The combination features improved paired predictions. Test whether single fields account for that improvement.'
                    if supported else 'The combination features did not show a clear paired gain. Try category co-occurrence.')
        elif mode=='anchored' and not used and COOCCUR in feasible:
            priorities=[COOCCUR]
            reason='The ordinary models share errors. Test locally learned category context before allocating the final slot.'
        elif mode=='anchored' and used and used[-1]==COOCCUR:
            a,b=evidence['corrected'],evidence['lost']
            harmed=pair_aware and a+b>=8 and b-a>=math.sqrt(a+b)
            if harmed and can_pool and any(name in feasible for name in aligned):
                priorities=[*aligned,DIRECT]
                reason='Local context introduced more paired losses. Probe the independent meaning of repeated category names.'
            else:
                priorities=[DIRECT,*aligned]
                reason='Keep the local candidate eligible and test an independent split representation.'
        elif mode=='histogram' and not used and HIST in feasible:
            priorities=[HIST]
            reason='Try a different tree learner that can partition categories into sets using training labels.'
        elif mode=='histogram' and used and used[-1]==HIST and HIST_ADJUSTED in feasible:
            a,b=evidence['corrected'],evidence['lost']
            if pair_aware and b-a>=2*math.sqrt(max(1,a+b)) and COOCCUR in feasible:
                priorities=[COOCCUR]
                reason='The alternate tree learner lost many paired decisions. Return to the incumbent with different features.'
            else:
                priorities=[HIST_ADJUSTED,COOCCUR]
                reason='The alternate tree learner remains competitive. Use its train-to-search gap to adjust leaf capacity.'
        elif mode=='boost-curve' and not used and SHRINKAGE in feasible:
            priorities=[HORIZON,SHRINKAGE]
            reason='Use validation log loss at four stages of the existing tree model to choose its next training horizon.'
        elif mode=='boost-curve' and used and used[-1] in (HORIZON,SHRINKAGE):
            a,b=evidence['corrected'],evidence['lost']
            if used[-1]==HORIZON and a-b>-math.sqrt(max(1,a+b)):
                priorities=[SHRINKAGE,COOCCUR]
                reason='The changed horizon was competitive. Test smaller boosting updates at matched nominal shrinkage.'
            else:
                priorities=[COOCCUR,MINILM]
                reason='Explore different features after testing the boosting curve.'
        elif mode=='distribution' and not used and RESIDUAL in feasible:
            priorities=[RESIDUAL]
            reason='Repeated categorical fields and common model errors make their training-only conditional distributions testable.'
        elif mode=='distribution' and used and used[-1]==RESIDUAL:
            a,b=evidence['corrected'],evidence['lost']
            boundary=math.sqrt(max(1,a+b))
            if pair_aware and a+b>=8 and a-b>=boundary:
                priorities=[RESIDUAL_DIRECT,MARGINAL]
                reason='The new coordinates gained more rows than they lost. Try direct splits with the same coordinates.'
            elif pair_aware and a+b>=8 and b-a>=boundary:
                priorities=[COOCCUR,DIRECT,CROSSES]
                reason='The new coordinates caused more errors. Return to a different representation.'
            else:
                priorities=[MARGINAL,COOCCUR]
                reason='Net gains are uncertain. Test whether the frequency of each category alone is sufficient.'
        elif mode=='typed-values' and not used and TYPED in feasible:
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
    requested_mode=mode
    if mode=='encoder-mixture':
        mode='encoder-depth' if kind in ('single_text','pair_text') else 'paired'
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
    text_latin_share=None
    confidence=None
    if mode in ('text-encoder','text-encoder-control','auto','opponent','item-control','encoder-depth','encoder-depth-control','text-regularization') and kind in ('pair_text','single_text') and encoder_installed and weights_available(config,'minilm_weights',MINI_FILES) and 128<=len(x)<=20000:
        sampled=[x[i] for i in np.linspace(0,len(x)-1,min(256,len(x)),dtype=int)]
        texts=[s for pair in sampled for s in pair] if kind=='pair_text' else sampled
        letters=[c for s in texts for c in str(s)[:512] if c.isalpha()]
        text_latin_share=sum(c.isascii() for c in letters)/max(1,len(letters))
        if text_latin_share>=.8:
            base=dict(model=MINI_MODEL,C=.3,dimensions=384,max_tokens=256)
            eligible[TEXT_MINILM]=dict(base,dense_only=False)
            eligible[TEXT_MINILM_ONLY]=dict(base,dense_only=True)
            if mode=='text-regularization' and history[-1]['candidate']==TEXT_MINILM and history[-1]['status']=='completed':
                from .text_confidence import observe_model
                trained=joblib.load(folder/f"trial_{history[-1]['trial']}.joblib")['model']
                confidence=observe_model(trained,kind,held,labels)
                if confidence['next_c'] is not None:
                    eligible[TEXT_REGULARIZED]=dict(base,C=confidence['next_c'],dense_only=False)
            if mode in ('encoder-depth','encoder-depth-control') and history[-1]['candidate']==TEXT_MINILM and history[-1]['status']=='completed':
                from .minilm_features import DEEP_MODEL,DEEP_FILES
                estimated=2*history[-1]['worker_wall_seconds']
                if estimated<=config.get('trial_timeout_seconds',90) and weights_available(config,'deep_minilm_weights',DEEP_FILES):
                    eligible[TEXT_DEEP]=dict(base,model=DEEP_MODEL,max_tokens=128,dense_only=False,estimated_seconds=estimated)
            if mode in ('opponent','item-control') and kind=='pair_text' and len(x)>=256:
                common=dict(base,rank=96,projection_seed=9181)
                eligible[PAIR_OPPONENT]=dict(common,opponent_context=True)
                eligible[PAIR_ITEM]=dict(common,opponent_context=False)
    if kind in ('pair_text','single_text') and encoder_installed and config and config.get('weights'):
        transfer=consider(kind,history[:protected_count],folder,labels,config,train_features=x,search_features=held)
        if transfer['decision']=='try': eligible[semantic]=PARAMS
    n = len(payload['y_train'])
    feature_count = int(x.shape[1]) if hasattr(x,'shape') and len(x.shape)>1 else 1
    torch_available = importlib.util.find_spec('torch') is not None
    if kind in SUPPORTED and n>=256 and feature_count<=2048 and torch_available and len(np.unique(payload['y_train']))<=256:
        eligible[NEURAL] = dict(hidden=64,epochs=22,batch_size=256,seed=42)
    rare_fraction, short_latin_share, low_errors = 0., 0., 0
    curve=None
    if kind=='categorical_tabular':
        cols=eligible_columns(x)
        if mode=='posteriors' and incumbent['candidate'].startswith('catboost_depth') and n>=256 and 2<=feature_count<=32 and len(np.unique(payload['y_train']))<=8:
            from itertools import combinations
            counts=[len(np.unique(col)) for col in _clean_categories(x).T]
            repeated=[count for count in counts if 2<=count<=256]
            if any(a*b<=n//4 for a,b in combinations(repeated,2)):
                base=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'],prior_count=20)
                eligible[PAIR_POSTERIOR]=dict(base,include_pairs=True)
                eligible[SINGLE_POSTERIOR]=dict(base,include_pairs=False)
        if mode=='histogram' and n>=256 and 2<=feature_count<=64 and n*feature_count<=4000000:
            leaf=max(15,min(40,math.ceil(n/220)))
            eligible[HIST]=dict(iterations=320,leaf_nodes=31,minimum_leaf=leaf,l2=2.)
            if len(history)>protected_count and history[-1]['candidate']==HIST and history[-1]['status']=='completed':
                gap=history[-1]['train_score']-history[-1]['search_score']
                tight=gap>=.08
                eligible[HIST_ADJUSTED]=dict(iterations=320,leaf_nodes=15 if tight else 63,minimum_leaf=leaf*2 if tight else leaf,
                                            l2=4. if tight else 2.,observed_gap=gap)
        if mode=='boost-curve' and incumbent['candidate'].startswith('catboost_depth'):
            from .boosting_curve import measured_curve,suggested_trees
            current=joblib.load(folder/f"trial_{incumbent['trial']}.joblib")['model']
            curve=measured_curve(current,held,labels)
            horizon=suggested_trees(curve)
            base=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'])
            count=int(current.tree_count_)
            rate=float(incumbent['params']['learning_rate'])
            if horizon is not None and horizon!=count:eligible[HORIZON]=dict(base,iterations=horizon,learning_rate=rate)
            eligible[SHRINKAGE]=dict(base,iterations=2*count,learning_rate=rate/2)
        if mode in ('representation','typed-values','distribution','distribution-control','anchored','anchored-control'):
            if mode.startswith('distribution') and incumbent['candidate'].startswith(BACKBONES) and feature_count<=40 and len(x)*feature_count<=2000000:
                from .distribution_features import column_codes
                encodable=column_codes(x)
                if len(encodable)>=2 and any(len(v[0])>=3 for v in encodable.values()):
                    base=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'],prior_count=20.)
                    eligible[MARGINAL]=dict(base,representation='frequency')
                    if mode=='distribution-control':
                        eligible[PERMUTED_RESIDUAL]=dict(base,representation='permuted')
                    else:
                        eligible[RESIDUAL]=dict(base,representation='conditional')
            pairs=feature_count*(feature_count-1)//2
            if 2<=feature_count<=32 and len(x)*(feature_count+pairs)<=6000000:
                from .challengers import cross_candidate
                eligible[CROSSES]=cross_candidate().params
            if incumbent['candidate'].startswith('catboost_depth'):
                small=sum(3<=len(np.unique(column))<=64 for column in _clean_categories(x).T)
                if small>=1:
                    eligible[DIRECT]=dict(backbone=incumbent['candidate'],backbone_params=incumbent['params'],one_hot_max_size=64)
                    if RESIDUAL in eligible:eligible[RESIDUAL_DIRECT]=dict(eligible[RESIDUAL],one_hot_max_size=64)
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
    if curve is not None:ev['boosting_curve']=curve
    if confidence is not None:ev['text_confidence']=confidence
    if text_latin_share is not None:ev['sampled_ascii_letter_fraction']=text_latin_share
    if mode in ('opponent','item-control','encoder-depth') and extra and extra[-1]['status']=='completed':
        ev['residual_shared_errors']=int(np.sum((reference_predictions!=labels)&(after!=labels)))
    elif mode in ('opponent','item-control','encoder-depth'):
        ev['residual_shared_errors']=0
    name,reason=select_action(ev,eligible,requested_mode,used)
    result=dict(decision='try' if name else 'ordinary',candidate=name,params=eligible.get(name),reason=reason,
                evidence=ev,previous_methods=used,eligible=list(eligible),transfer=transfer,
                protected_incumbent=incumbent['candidate'],mode=requested_mode)
    if requested_mode in ('auto','encoder-mixture'):
        result['resolved_mode']=resolved_mode(requested_mode,eligible)
    return result


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
        elif name in (TEXT_MINILM,TEXT_MINILM_ONLY,TEXT_REGULARIZED) and kind in ('pair_text','single_text'):
            from .minilm_features import MiniLMTextClassifier
            model=MiniLMTextClassifier(kind,Path(folder)/'pretrained_runtime.json',params['dense_only'])
            if name==TEXT_REGULARIZED:
                model.regularization=params['C']
            model.fit(rows,labels)
        elif name==TEXT_DEEP and kind in ('pair_text','single_text'):
            from .minilm_features import DeeperMiniLMTextClassifier
            model=DeeperMiniLMTextClassifier(kind,Path(folder)/'pretrained_runtime.json',False).fit(rows,labels)
        elif name in (PAIR_OPPONENT,PAIR_ITEM) and kind=='pair_text':
            from .sentence_interactions import NonlinearSentenceClassifier
            model=NonlinearSentenceClassifier(Path(folder)/'pretrained_runtime.json',params).fit(rows,labels)
        elif name in (PAIR_POSTERIOR,SINGLE_POSTERIOR) and kind=='categorical_tabular':
            from .category_posteriors import PosteriorCategoryClassifier
            model=PosteriorCategoryClassifier(params).fit(rows,labels)
        elif name in (HIST,HIST_ADJUSTED) and kind=='categorical_tabular':
            from .histogram_categories import HistogramCategoryClassifier
            model=HistogramCategoryClassifier(params).fit(rows,labels)
        elif name in (RESIDUAL,RESIDUAL_DIRECT,PERMUTED_RESIDUAL,MARGINAL) and kind=='categorical_tabular':
            from .distribution_features import DistributionClassifier
            model=DistributionClassifier(params).fit(rows,labels)
        elif name in (TYPED,TYPED_DIRECT) and kind=='categorical_tabular':
            from .typed_categories import TypedCategoryClassifier
            model=TypedCategoryClassifier(params).fit(rows,labels)
        elif name in (DIRECT,HORIZON,SHRINKAGE) and kind=='categorical_tabular':
            from .models import _tabular_catboost
            base=params['backbone_params']
            model=_tabular_catboost(labels,base['depth'],base['l2_leaf_reg'])
            model.set_params(**{k:params[k] for k in ('one_hot_max_size','iterations','learning_rate') if k in params})
            model.fit(_clean_categories(rows),labels,cat_features=list(range(rows.shape[1])))
            return model,model.predict(_clean_categories(search)).reshape(-1)
        elif name in (COOCCUR,DIRECT_CONTEXT,MINILM) and kind=='categorical_tabular':
            encoder=MiniLMEncoder(Path(folder)/'pretrained_runtime.json') if name==MINILM else Cooccurrence(rows,params['columns'])
            model=LevelClassifier(params).fit(rows,labels,encoder)
        else:
            raise ValueError('Unsupported method')
        return model,model.predict(search)
    return Candidate(name,params,fit)
