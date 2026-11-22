"""Independent parameter observer prototype; not imported by the live agent."""
import math

from superagent.estimator_facts import _unwrap,observed_classifier


FIELDS={
    'LogisticRegression':('C','penalty','solver','max_iter','class_weight','fit_intercept'),
    'LinearSVC':('C','penalty','loss','max_iter','class_weight'),
    'SVC':('C','kernel','gamma','degree','class_weight','probability'),
    'ExtraTreesClassifier':('n_estimators','max_depth','min_samples_leaf','max_features','class_weight'),
    'RandomForestClassifier':('n_estimators','max_depth','min_samples_leaf','max_features','class_weight'),
    'CatBoostClassifier':('depth','l2_leaf_reg','iterations','learning_rate','loss_function'),
}


def scalar(value):
    if value is None or type(value) in (str,bool,int):return value,True
    if type(value) is float and math.isfinite(value):return value,True
    return None,False


def observe(model):
    inner=_unwrap(model);facts=observed_classifier(model);name=type(inner).__name__
    values={};unavailable={}
    parameters=inner.get_params(deep=False) if name in FIELDS else {}
    for key in FIELDS.get(name,()):
        if key not in parameters:unavailable[key]='Not present in fitted object constructor parameters';continue
        value,ok=scalar(parameters[key])
        if ok:values[key]=value
        else:unavailable[key]='Non-scalar parameter; this bounded observer does not serialize it'
    return dict(**facts,configured_parameters=values,unavailable_fields=unavailable,
        learned_tree_count=getattr(inner,'tree_count_',None) if name=='CatBoostClassifier' else None,
        observation_source='known unwrapped fitted object get_params(deep=False)',
        scope='Constructor parameter facts; not a causal explanation or feature-engineering audit',deployed_in_agent=False)


def compare(declared,observed):
    checked=[]
    for key,value in declared.items():
        if key not in observed['configured_parameters']:
            checked.append(dict(parameter=key,declared=value,status='unverified',reason='No bounded fitted-parameter observation'))
            continue
        actual=observed['configured_parameters'][key]
        numeric=type(value) in (int,float) and type(actual) in (int,float)
        equal=math.isclose(value,actual,rel_tol=1e-12,abs_tol=1e-12) if numeric else type(value)==type(actual) and value==actual
        checked.append(dict(parameter=key,declared=value,observed=actual,status='confirmed' if equal else 'contradicted'))
    return checked
