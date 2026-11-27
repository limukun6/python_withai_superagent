"""Create executable pipelines from the installed estimator inventory and observed training evidence."""
from functools import lru_cache, partial
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

from .ab_preference_output import CanonicalPreferencePipeline
from .ab_symmetric import UnorderedPairFeatures
from .estimator_facts import observed_classifier
from .generated_pipeline import KINDS, make_preprocessor
from .metrics import score_of
from .models import Candidate, search_candidates
from .runtime_inventory import ProviderRequirement, installed_extensions, runtime_classifiers
from .task_constraints import permitted


VERSION=4
PROJECTIONS=('native','latent','fourier')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class LatentFeatures(TransformerMixin,BaseEstimator):
    def __init__(self,dimensions):
        self.dimensions=dimensions

    def fit(self,x,y=None):
        count=min(self.dimensions,x.shape[0]-1,x.shape[1]-1)
        self.reducer_=TruncatedSVD(n_components=count,random_state=42).fit(x) if count>=1 else None
        return self

    def transform(self,x):
        if self.reducer_ is not None:
            return self.reducer_.transform(x)
        return x.toarray() if hasattr(x,'toarray') else np.asarray(x)


class PreferencePipeline(CanonicalPreferencePipeline):
    def predict(self,rows,**params):
        if hasattr(self.steps[-1][1],'predict_proba'):
            return super().predict(rows,**params)
        ordered=[(a,b) if a<=b else (b,a) for a,b in rows]
        labels=np.asarray(Pipeline.predict(self,ordered,**params))
        return np.where([a<=b for a,b in rows],labels,1-labels)


def valid_spec(kind,spec):
    required={'kind','semantics','class','kwargs','projection','dimensions'}
    if not isinstance(spec,dict) or set(spec)!=required or kind not in KINDS or spec['kind']!=kind:
        raise ValueError('Invalid open pipeline task or fields')
    if spec['semantics'] not in (('preference','symmetric') if kind=='pair_text' else (None,)):
        raise ValueError('Incompatible open pipeline pair semantics')
    if spec['projection'] not in PROJECTIONS or type(spec['dimensions']) is not int or not 2<=spec['dimensions']<=96:
        raise ValueError('Invalid open projection')
    cls=runtime_classifiers().get(spec['class'])
    args=spec['kwargs']
    if cls is None or not isinstance(args,dict) or not set(args)<=cls().get_params(deep=False).keys():
        raise ValueError('Unknown runtime classifier or constructor parameters')
    # Only scalar parameters used by the synthesizer are accepted in a saved registry.
    bounds={'C':(.001,64),'alpha':(.000001,8),'n_estimators':(24,192),'max_depth':(2,18),'min_samples_leaf':(1,64),
            'n_neighbors':(1,32),'max_iter':(1,2000),'n_jobs':(1,1),'random_state':(42,42),'cache_size':(128,128),
            'nu':(.001,.5),'n_restarts_optimizer':(0,0)}
    for key,value in args.items():
        if key=='outlier_label' and value=='most_frequent':
            continue
        if key not in bounds or type(value) not in (float,int) or not math.isfinite(value) or not bounds[key][0]<=value<=bounds[key][1]:
            raise ValueError('Open constructor parameter outside its resource boundary')
    estimator=cls(**args)
    if hasattr(estimator,'_parameter_constraints') and hasattr(estimator,'_validate_params'):
        estimator._validate_params()
    return estimator


def fit_pipeline(spec,x,y,query):
    estimator=valid_spec(spec['kind'],spec)
    pref=spec['semantics']=='preference'
    base=(UnorderedPairFeatures(('word','char'),bins=4096) if spec['semantics']=='symmetric' else make_preprocessor(spec['kind']))
    steps=[('features',base)]
    if spec['projection']=='latent':
        steps.append(('latent',LatentFeatures(spec['dimensions'])))
    elif spec['projection']=='fourier':
        steps.append(('fourier',RBFSampler(n_components=spec['dimensions'],gamma='scale',random_state=42)))
    steps.append(('classifier',estimator))
    model=PreferencePipeline(steps) if pref else Pipeline(steps)
    if pref:
        x=[*x,*[(b,a) for a,b in x]]
        y=np.r_[y,1-np.asarray(y)]
    model.fit(x,y)
    if spec['class'] in installed_extensions()[0]:
        model.provider_requirement_=ProviderRequirement(spec['class'])
    return model,model.predict(query)


def materialize(spec):
    estimator=valid_spec(spec['kind'],spec)
    key='open_'+type(estimator).__name__+'_'+digest(spec)[:20]
    params=dict(spec,representation=spec['projection']+'_'+spec['kind'],runtime_composed=True,
                classifier_fits_per_evaluation=1,swap_augmentation=spec['semantics']=='preference')
    return Candidate(key,params,partial(fit_pipeline,spec))


def capability(spec):
    candidate=materialize(spec)
    estimator=valid_spec(spec['kind'],spec)
    facts=observed_classifier(estimator)
    family=facts['family'] or type(estimator).__name__
    rule={'preference':'flips','symmetric':'preserves',None:'not_applicable'}[spec['semantics']]
    return dict(id=candidate.family,params=candidate.params,route='open_'+type(estimator).__name__,input_kind=spec['kind'],
                kind=spec['kind'],representation=candidate.params['representation'],estimator=family,estimator_class=type(estimator).__name__,
                supports_probabilities=hasattr(estimator,'predict_proba'),classifier_fits=1,refit_classifier_fits=1,
                expected_classifier_fits=1,origin='runtime_composed',resource='cpu',swap_label_rule=rule)


@lru_cache(maxsize=1)
def discovered():
    return {n:(cls,get_tags(cls()),cls().get_params(deep=False)) for n,cls in runtime_classifiers().items()}


def propose(task,profile,history):
    """The classifier names originate in the installed distribution. No task or dataset name is an input."""
    kind=task['kind']
    semantics=task['input_schema'].get('pair_semantics') if kind=='pair_text' else None
    if kind=='pair_text' and semantics is None and task['input_schema'].get('pair_label_one')=='left':
        semantics='preference'
    primary=[r for r in history if r['split_id']=='primary' and r['status']=='completed']
    best=max(primary,key=score_of,default=None)
    statistics=profile['statistics']
    n=statistics['train_rows']
    shape=statistics.get('feature_shape') or [statistics.get('feature_count',statistics.get('word_count',{}).get('median',8))]
    width=max(1,math.prod(shape))
    gap=max(0.,best['train_score']-score_of(best)) if best else 0.
    error=1-score_of(best) if best else 1-profile['majority_class_fraction']
    strength=round(float(np.clip(math.log1p(n)/(math.sqrt(width)+1)*math.exp(-4*gap)*(1+error),.001,64.)),6)
    dimensions=max(2,min(96,round(math.sqrt(n)*(1+error)/(1+3*gap))))
    sparse=kind in ('single_text','pair_text','categorical_tabular','mixed_tabular','text_tabular')
    nonnegative=kind in ('categorical_tabular','single_text') or semantics=='symmetric'
    minority=profile['minority_class_fraction']
    entries,excluded={},{}
    for name,(cls,tags,defaults) in discovered().items():
        classifier=tags.classifier_tags
        reason=None
        if classifier is None or classifier.poor_score or not tags.target_tags.single_output or cls.__module__.startswith('sklearn.semi_supervised'):
            reason='not an ordinary single-target inductive classifier'
        elif len(task['metric']['labels'])>2 and not classifier.multi_class:
            reason='requires binary labels'
        elif task['metric']['name']=='roc_auc' and not hasattr(cls(),'predict_proba'):
            reason='requires probabilities in frozen class order'
        elif tags.input_tags.pairwise or tags.input_tags.categorical or (tags.input_tags.positive_only and not nonnegative):
            reason='incompatible feature domain'
        elif cls.__module__.startswith('sklearn.gaussian_process.') and (n>180 or len(task['metric']['labels'])>2):
            reason='cubic kernel fit exceeds this worker budget'
        elif ('kernel' in defaults and n>12000) or (cls.__module__.startswith('sklearn.discriminant_analysis') and width>512):
            reason='quadratic workspace exceeds this worker budget'
        if reason:
            excluded[name]=reason
            continue
        projection='native'
        if sparse and not tags.input_tags.sparse:
            if tags.input_tags.positive_only:
                excluded[name]='signed dense projection incompatible with count features'
                continue
            projection='latent'
        elif best and not tags.input_tags.positive_only and 'linear_only' not in task.get('method_restrictions',()):
            if gap>.12:
                projection='latent'
            elif best['train_score']<.9 and width<=256:
                projection='fourier'
        if projection!='native' and n*dimensions*8>128*2**20:
            excluded[name]='dense projection exceeds memory allowance'
            continue
        values=dict(random_state=42,n_jobs=1,C=strength,alpha=round(max(.000001,min(8.,1/(strength+1))),6),
                    n_estimators=max(24,min(192,round(4*math.sqrt(n)))),max_depth=max(2,min(18,round(math.log2(n+1)/(1+3*gap)))),
                    min_samples_leaf=max(1,min(64,round(1+gap*math.sqrt(n)))),
                    n_neighbors=max(1,min(32,n//5 or 1,round(math.sqrt(math.log1p(n))*(1+4*gap)))),
                    nu=max(.001,min(.5,minority/2)),cache_size=128,n_restarts_optimizer=0,outlier_label='most_frequent')
        limit=defaults.get('max_iter')
        if 'max_iter' in defaults and (limit is None or isinstance(limit,(int,float)) and (limit<=0 or limit>2000)):
            values['max_iter']=1500
        kwargs={k:v for k,v in values.items() if k in defaults}
        spec=dict(kind=kind,semantics=semantics,**{'class':name},kwargs=kwargs,projection=projection,dimensions=dimensions if projection!='native' else 2)
        if name in installed_extensions()[0]:
            try:
                candidate=materialize(spec)
            except Exception as error:
                excluded[name]='unsupported generated parameters: '+type(error).__name__
                continue
        else:
            candidate=materialize(spec)
        entries[candidate.family]=spec
    provenance=dict(trials=[r['trial'] for r in primary],best_trial=best['trial'] if best else None,
                    score=score_of(best) if best else None,gap=gap,error=error,rows=n,width=width,
                    inventory=sorted(discovered()),excluded={**installed_extensions()[1],**excluded})
    return entries,provenance


def resolve(kind,name,document):
    if set(document)!={'schema_version','kind','entries'} or document['schema_version']!=VERSION or document['kind']!=kind:
        raise ValueError('Invalid open-search registry')
    if name in document['entries']:
        spec=document['entries'][name]
        if spec['kind']!=kind:
            raise ValueError('Runtime pipeline has the wrong modality')
        if spec.get("operator")=="add_masked_context":
            from .masked_context import materialize as context_method
            method=context_method(spec)
        else:
            method=materialize(spec)
        if method.family!=name:
            raise ValueError('Runtime configuration digest differs')
        return method
    return next(c for c in search_candidates(kind,'all') if c.family==name)


class MethodSpace:
    def __init__(self,spec,profile,folder):
        self.spec,self.profile,self.folder=spec,profile,Path(folder)
        self.entries={}
        self.frontier=()
        self.generations={}

    def refresh(self,history,baseline):
        key=tuple(r['trial'] for r in history if r['split_id']=='primary' and r['status']=='completed')
        if key not in self.generations:
            proposals,why=propose(self.spec,self.profile,history)
            restrictions=self.spec.get('method_restrictions',())
            if restrictions:
                allowed={n:s for n,s in proposals.items() if permitted(capability(s),restrictions)}
                why['excluded_by_task_restrictions']=sorted(proposals.keys()-allowed.keys())
                proposals=allowed
            new=sorted(proposals.keys()-self.entries.keys())
            self.entries.update(proposals)
            self.frontier=tuple(proposals)
            document=dict(schema_version=VERSION,kind=self.spec['kind'],entries=self.entries)
            payload=json.dumps(document,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
            version=self.folder/f'runtime_registry_{len(self.generations)}.json'
            with version.open('xb') as f:
                f.write(payload)
            current=self.folder/'candidate_registry.json'
            temporary=current.with_suffix('.tmp')
            temporary.write_bytes(payload)
            temporary.replace(current)
            self.generations[key]=dict(file=version.name,sha256=hashlib.sha256(payload).hexdigest(),
                                       generated=list(proposals),newly_seen=new,evidence=why)
        visible=set(self.frontier)|{r['candidate'] for r in history}
        caps={**baseline,**{n:capability(self.entries[n]) for n in sorted(visible&self.entries.keys())}}
        return {n:c for n,c in caps.items() if permitted(c,self.spec.get('method_restrictions',()))},self.generations[key]
