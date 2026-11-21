"""Count actual sklearn classifier fit entries during small regression suites."""
from contextlib import contextmanager, ExitStack
from functools import wraps
from unittest.mock import patch


@contextmanager
def observe_classifier_fits():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.dummy import DummyClassifier
    calls = []
    def observe(original):
        @wraps(original)
        def fit(model, *args, **kwargs):
            record = dict(classifier=type(model).__name__, status='attempted')
            calls.append(record)
            try:
                result = original(model, *args, **kwargs)
            except Exception:
                record['status'] = 'failed'
                raise
            record['status'] = 'completed'
            return result
        return fit
    with ExitStack() as stack:
        for classifier in (LogisticRegression, ExtraTreesClassifier, SVC, DummyClassifier):
            stack.enter_context(patch.object(classifier, 'fit', observe(classifier.fit)))
        yield calls
