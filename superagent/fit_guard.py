"""Count supervised fits of the closed AB-v1 catalogue before executing them.

One tree ensemble is one classifier. Its internal decision trees and unsupervised
preprocessors are not additional candidate fits. This is not an arbitrary-code sandbox.
"""
from contextlib import contextmanager
from functools import wraps
from importlib.util import find_spec
import json
from pathlib import Path


@contextmanager
def admitted_fits(admission, candidate, *, runtime_estimator=None):
    if admission is None:
        yield None
        return
    limit = admission['maximum_classifier_fits']
    if type(limit) is not int or not 1 <= limit <= 11:
        raise ValueError('Invalid worker classifier fit admission')
    path = Path(admission['ledger_path'])
    if path.exists():
        raise FileExistsError('Worker fit ledger already exists')
    record = dict(candidate=candidate, maximum_classifier_fits=limit,
                  attempted_classifier_fits=0, completed_classifier_fits=0,
                  status='initialized', calls=[])

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
        temporary.replace(path)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import LinearSVC, SVC
    classes = [LogisticRegression, ExtraTreesClassifier, SVC, LinearSVC, BernoulliNB]
    if candidate.startswith('preference_analogy_'):
        from .preference_analogy import AnalogyClassifier
        classes.append(AnalogyClassifier)
    if runtime_estimator is not None and runtime_estimator not in classes:
        classes.append(runtime_estimator)
    if find_spec('catboost') is not None:
        from catboost import CatBoostClassifier
        if CatBoostClassifier not in classes:
            classes.append(CatBoostClassifier)
    originals = []

    def instrument(original):
        @wraps(original)
        def fit(model, *args, **kwargs):
            if record['attempted_classifier_fits'] >= limit:
                raise RuntimeError('Supervised fit exceeds worker admission')
            call = dict(classifier=type(model).__name__, status='started')
            record['calls'].append(call)
            record['attempted_classifier_fits'] += 1
            record['status'] = 'running'
            save()  # Durable attempted count BEFORE the actual classifier call.
            try:
                value = original(model, *args, **kwargs)
            except Exception:
                call['status'] = 'failed'
                save()
                raise
            record['completed_classifier_fits'] += 1
            call['status'] = 'completed'
            save()
            return value
        return fit

    save()
    try:
        originals = [(cls, cls.fit, 'fit' in vars(cls)) for cls in classes]
        for cls, original, _ in originals:
            cls.fit = instrument(original)
        yield record
        if record['attempted_classifier_fits'] != limit or record['completed_classifier_fits'] != limit:
            raise ValueError('Actual supervised fits differ from the registered completed candidate cost')
        record['status'] = 'completed'
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        for cls, original, owned in originals:
            if owned:
                cls.fit = original
            else:
                del cls.fit
        save()
