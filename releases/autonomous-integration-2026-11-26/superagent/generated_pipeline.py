"""Restricted sklearn pipeline factories, distinct from registered candidates.

This is a checked constructor language, not an arbitrary-Python/OS sandbox.
Only trusted installed classifier constructors and a task-owned preprocessor
can appear. Generated imports and statements are never executed.
"""
import ast
from dataclasses import dataclass
from functools import lru_cache, partial
import hashlib
import math

import numpy as np
from scipy.sparse import hstack
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .general_models import MixedPreprocessor, TextTablePreprocessor, ImageFlattener
from .models import _clean_categories


KINDS={'numeric_tabular','categorical_tabular','mixed_tabular','text_tabular','single_text','pair_text','image'}


class Categories(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):return self
    def transform(self,x):return _clean_categories(x)


class PairFeatures(TransformerMixin,BaseEstimator):
    def __init__(self,max_features=30000):self.max_features=max_features
    def fit(self,x,y=None):
        self.word_=TfidfVectorizer(ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',min_df=1,max_features=self.max_features,sublinear_tf=True)
        self.char_=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=1,max_features=self.max_features,sublinear_tf=True)
        texts=[s for pair in x for s in pair]
        self.word_.fit(texts);self.char_.fit(texts);return self
    def transform(self,x):
        a=[r[0] for r in x];b=[r[1] for r in x]
        return hstack([v.transform(a)-v.transform(b) for v in (self.word_,self.char_)]).tocsr()


def make_preprocessor(kind):
    if kind not in KINDS:raise ValueError('Unknown generated-pipeline modality')
    if kind=='pair_text':return PairFeatures()
    if kind=='single_text':
        return FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b',min_df=1,max_features=30000,sublinear_tf=True)),
                             ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=1,max_features=30000,sublinear_tf=True))])
    if kind=='text_tabular':return TextTablePreprocessor()
    if kind=='mixed_tabular':return MixedPreprocessor()
    if kind=='categorical_tabular':return Pipeline([('clean',Categories()),('encode',OneHotEncoder(handle_unknown='ignore'))])
    if kind=='image':return Pipeline([('flatten',ImageFlattener()),('scale',StandardScaler())])
    return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])


@lru_cache(maxsize=1)
def classifier_inventory():
    from sklearn.utils import all_estimators
    result={}
    for name,cls in all_estimators(type_filter='classifier'):
        try:parameters=cls().get_params(deep=False)
        except (TypeError,ValueError):continue
        # Meta-estimators and implicit CV would need separate component accounting.
        if name.endswith('CV') or any(k in parameters for k in ('estimator','estimators','base_estimator')):continue
        result[name]=cls
    return result


def _literal(node):
    if isinstance(node,ast.Constant) and type(node.value) in (str,int,float,bool,type(None)):
        value=node.value
        if isinstance(value,str) and (len(value)>120 or '/' in value or '\\' in value or '__' in value):
            raise ValueError('Constructor strings cannot contain paths, dunders or long data')
        if type(value) in (int,float) and (not math.isfinite(value) or abs(value)>10000):raise ValueError('Constructor number exceeds bounds')
        return value
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant) and type(node.operand.value) in (int,float):
        return -_literal(node.operand)
    if isinstance(node,(ast.List,ast.Tuple)) and len(node.elts)<=20:
        values=[_literal(v) for v in node.elts];return tuple(values) if isinstance(node,ast.Tuple) else values
    if isinstance(node,ast.Dict) and len(node.keys)<=20:
        return {_literal(k):_literal(v) for k,v in zip(node.keys,node.values)}
    raise ValueError('Only bounded literal constructor parameters are supported')


def checked_factory(code,kind):
    if kind not in KINDS or not isinstance(code,str) or not 0<len(code)<=6000:raise ValueError('Invalid generated code or kind')
    tree=ast.parse(code)
    if len(list(ast.walk(tree)))>400:raise ValueError('Generated constructor tree is too large')
    inventory=classifier_inventory();environment={'Pipeline':Pipeline,'make_preprocessor':make_preprocessor,'__builtins__':{}}
    definitions=[]
    for statement in tree.body:
        if isinstance(statement,ast.ImportFrom):
            if statement.level:raise ValueError('Relative imports are not permitted')
            for alias in statement.names:
                visible=alias.asname or alias.name
                if visible.startswith('_') or visible in ('kind','build_model','make_preprocessor'):raise ValueError('Invalid constructor alias')
                if statement.module=='sklearn.pipeline' and alias.name=='Pipeline' and visible=='Pipeline':continue
                cls=inventory.get(alias.name)
                if cls is None or statement.module not in (cls.__module__,cls.__module__.split('._')[0]):
                    raise ValueError('Import is not a supported installed sklearn classifier')
                if visible in environment:raise ValueError('Duplicate constructor alias')
                environment[visible]=cls
        elif isinstance(statement,ast.FunctionDef):definitions.append(statement)
        else:raise ValueError('Only import declarations and one constructor factory are allowed')
    if len(definitions)!=1:raise ValueError('One build_model factory is required')
    factory=definitions[0];arguments=factory.args
    if (factory.name!='build_model' or factory.decorator_list or factory.returns is not None or getattr(factory,'type_params',[])
        or [a.arg for a in arguments.args]!=['kind'] or arguments.posonlyargs or arguments.kwonlyargs
        or arguments.vararg or arguments.kwarg or arguments.defaults or arguments.kw_defaults
        or arguments.args[0].annotation is not None or len(factory.body)!=1 or not isinstance(factory.body[0],ast.Return)):
        raise ValueError('Factory must be def build_model(kind): return Pipeline(...)')
    call=factory.body[0].value
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='Pipeline' or call.keywords or len(call.args)!=1:
        raise ValueError('Factory must return one two-step Pipeline')
    steps=call.args[0]
    if not isinstance(steps,ast.List) or len(steps.elts)!=2:raise ValueError('Pipeline requires trusted preprocessing and one classifier')
    values=[]
    for step in steps.elts:
        if not isinstance(step,ast.Tuple) or len(step.elts)!=2:raise ValueError('Invalid pipeline step')
        values.append((_literal(step.elts[0]),step.elts[1]))
    if [name for name,_ in values]!=['features','classifier']:raise ValueError('Pipeline step names must be features/classifier')
    pre=values[0][1]
    if (not isinstance(pre,ast.Call) or not isinstance(pre.func,ast.Name) or pre.func.id!='make_preprocessor'
        or pre.keywords or len(pre.args)!=1 or not isinstance(pre.args[0],ast.Name) or pre.args[0].id!='kind'):
        raise ValueError('Use the task-owned make_preprocessor(kind)')
    estimator=values[1][1]
    if not isinstance(estimator,ast.Call) or not isinstance(estimator.func,ast.Name) or estimator.func.id not in environment or estimator.func.id in ('Pipeline','make_preprocessor') or estimator.args:
        raise ValueError('Use an installed classifier with literal keyword parameters')
    params={}
    for keyword in estimator.keywords:
        if keyword.arg is None or keyword.arg in params:raise ValueError('No unpacked/duplicate constructor parameters')
        params[keyword.arg]=_literal(keyword.value)
    cls=environment[estimator.func.id]
    classifier=cls(**params);effective=classifier.get_params(deep=False)
    for name,limit in [('n_estimators',500),('max_iter',2000),('max_depth',20),('cache_size',256),('n_restarts_optimizer',2),('n_neighbors',50)]:
        value=effective.get(name)
        if value is not None and (type(value) not in (int,float) or not 0<=value<=limit):raise ValueError('Generated estimator resource bound: '+name)
    hidden=effective.get('hidden_layer_sizes')
    if hidden is not None:
        sizes=(hidden,) if type(hidden) is int else hidden
        if not isinstance(sizes,(tuple,list)) or not 1<=len(sizes)<=2 or any(type(v) is not int or not 1<=v<=128 for v in sizes) or sum(sizes)>256:
            raise ValueError('Generated MLP hidden layers exceed the small-model bound')
    if effective.get('n_jobs') is not None and (type(effective['n_jobs']) is not int or effective['n_jobs'] not in (1,2)):
        raise ValueError('Generated estimators allow at most2 job workers')
    if 'random_state' in effective and type(effective['random_state']) is not int:raise ValueError('Set an explicit integer random_state')
    if effective.get('probability') is True:raise ValueError('Implicit probability-calibration CV is not supported by this fit accountant')
    # Import declarations were checked against trusted classes and are removed.
    only_factory=ast.Module(body=[factory],type_ignores=[]);ast.fix_missing_locations(only_factory)
    exec(compile(only_factory,'<checked-pipeline-factory>','exec'),environment)
    pipeline=environment['build_model'](kind)
    if not isinstance(pipeline,Pipeline) or type(pipeline.named_steps['classifier']) is not cls:
        raise ValueError('Constructed pipeline differs from checked declaration')
    return pipeline,{'classifier':cls.__name__,'classifier_module':cls.__module__,'parameters':params,
                     'generation_language':'checked_sklearn_pipeline_factory','unrestricted_python':False}


@dataclass
class GeneratedModel:
    kind: str
    pipeline: object
    training_metadata: dict
    classifier_fits: int=1

    @property
    def classes_(self):return np.asarray(self.pipeline.classes_)

    def predict_proba(self,x):
        p=self.pipeline.predict_proba(x)
        if self.kind=='pair_text':
            reverse=self.pipeline.predict_proba([(b,a) for a,b in x])
            positive=(p[:,1]+1-reverse[:,1])/2
            return np.column_stack([1-positive,positive])
        return p

    def predict(self,x):
        if self.kind!='pair_text':return self.pipeline.predict(x)
        if hasattr(self.pipeline,'predict_proba'):
            return self.classes_[np.argmax(self.predict_proba(x),axis=1)]
        if hasattr(self.pipeline,'decision_function'):
            values=(self.pipeline.decision_function(x)-self.pipeline.decision_function([(b,a) for a,b in x]))/2
            return self.classes_[(values>0).astype(int)]
        forward=self.pipeline.predict(x);reverse=self.pipeline.predict([(b,a) for a,b in x])
        return ((forward+1-reverse)>1).astype(int)


def fit_generated(kind,code,x_train,y_train,x_search):
    pipeline,spec=checked_factory(code,kind);y=np.asarray(y_train)
    if kind=='pair_text':
        if set(y.tolist())!={0,1}:raise ValueError('Generated preference model needs both0/1 classes')
        pipeline.fit([*x_train,*[(b,a) for a,b in x_train]],np.concatenate([y,1-y]))
    else:pipeline.fit(x_train,y)
    metadata={**spec,'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'training_rows':len(y),
        'swap_augmentation':kind=='pair_text','classifier_fits':1,'pretrained_weights':False}
    model=GeneratedModel(kind,pipeline,metadata)
    return model,model.predict(x_search)


def materialize_generated(kind,code):
    from .models import Candidate
    _,spec=checked_factory(code,kind)
    digest=hashlib.sha256(code.encode()).hexdigest()
    params={'code_sha256':digest,'kind':kind,'classifier':spec['classifier'],
            'constructor_parameters':spec['parameters'],
            'representation':'task_owned_preprocessor','classifier_fits_per_evaluation':1}
    return Candidate('generated_'+kind+'_'+digest[:16],params,partial(fit_generated,kind,code))


def resolve_generated_registry(kind,candidate_id,document):
    from .models import search_candidates
    if set(document)!={'schema_version','kind','portfolio','entries'} or document['schema_version']!=3 or document['kind']!=kind:
        raise ValueError('Invalid runtime-generated registry')
    if document['portfolio'] not in ('original','expanded') or not isinstance(document['entries'],dict) or len(document['entries'])>1:
        raise ValueError('Runtime generation allows one bounded candidate')
    generated={}
    for key,entry in document['entries'].items():
        if set(entry)!={'spec','params','origin'} or set(entry['spec'])!={'kind','code'} or entry['spec']['kind']!=kind:
            raise ValueError('Malformed generated entry')
        candidate=materialize_generated(kind,entry['spec']['code'])
        if candidate.family!=key or candidate.params!=entry['params']:raise ValueError('Generated source hash or parameters changed')
        generated[key]=candidate
    if candidate_id in generated:return generated[candidate_id]
    static=next((c for c in search_candidates(kind,'all') if c.family==candidate_id),None)
    if static is None:raise ValueError('Candidate absent from generated and static catalogs')
    return static
