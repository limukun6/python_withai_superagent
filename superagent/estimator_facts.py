"""What was actually fitted, read from the trained artifact.

The declared configuration facts used to be a lookup table keyed by training
function name. That answers "what does this candidate usually build", not "what
did this run build", and it returns nothing for candidates created at runtime.
This module reads the fitted object and reports the classifier family, so a
claim can be checked against the artifact instead of against the catalogue.
"""
from sklearn.pipeline import Pipeline

# Vocabulary shared with targeted_feedback.candidate_facts.
_BY_CLASS = {
    'LogisticRegression': 'logistic',
    'LogisticRegressionCV': 'logistic',
    'RidgeClassifier': 'ridge',
    'RidgeClassifierCV': 'ridge',
    'LinearSVC': 'linear_svm',
    'CatBoostClassifier': 'catboost',
    'ExtraTreesClassifier': 'extra_trees',
    'RandomForestClassifier': 'random_forest',
    'GradientBoostingClassifier': 'gradient_boosting',
    'HistGradientBoostingClassifier': 'gradient_boosting',
    'DummyClassifier': 'dummy',
    'GaussianNB': 'naive_bayes',
    'MultinomialNB': 'naive_bayes',
    'BernoulliNB': 'naive_bayes',
    'DecisionTreeClassifier': 'decision_tree',
    'KNeighborsClassifier': 'nearest_neighbours',
    'AnalogyClassifier': 'preference_analogy',
}


def _unwrap(model):
    """Follow pipelines, our own blend wrappers, and (vectorizer..., classifier) tuples.

    Only wrappers defined in this package are followed by attribute: ensemble
    estimators also expose `.estimator`, and that attribute is their base tree,
    not the fitted model.
    """
    for _ in range(6):
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
            continue
        if isinstance(model, tuple) and model:
            model = model[-1]
            continue
        if (type(model).__module__ or '').startswith('superagent'):
            inner = getattr(model, 'classifier', None) or getattr(model, 'estimator', None)
            if inner is not None and inner is not model:
                model = inner
                continue
        break
    return model


def observed_classifier(model):
    """The classifier family of the object that was actually fitted."""
    inner = _unwrap(model)
    name = type(inner).__name__
    family = _BY_CLASS.get(name)
    kernel = getattr(inner, 'kernel', None) if name in ('SVC','NuSVC') else None
    if family is None and name in ('SVC', 'NuSVC'):
        family = {'linear':'linear_svm','rbf':'rbf_svm','poly':'poly_svm',
                  'sigmoid':'sigmoid_svm','precomputed':'precomputed_svm'}.get(kernel) if isinstance(kernel,str) else None
    return {'family': family, 'class': name, 'module': type(inner).__module__,
            'parameters': {'kernel':kernel if isinstance(kernel,str) else 'callable_or_unknown'} if name in ('SVC','NuSVC') else {},
            'source': 'fitted_estimator'}


def verify_configuration_claim(proposal, observed):
    """Compare a declared classifier with the one that was fitted."""
    claim = (proposal or {}).get('configuration_claim')
    declared = claim.get('classifier') if isinstance(claim, dict) else None
    family = (observed or {}).get('family')
    if declared is None:
        status = 'not_declared'
    elif family is None:
        status = 'unverified'
    elif declared == family:
        status = 'confirmed'
    else:
        status = 'contradicted'
    return {'declared': declared, 'observed': family,
            'observed_class': (observed or {}).get('class'),
            'status': status,
            'scope': 'Declared classifier versus the fitted estimator of this trial'}
