"""A29 only: observe supervised worker fits; never intercept the live planning process."""
import json
import os
from pathlib import Path
import sys


if os.environ.get('AB_A29_OBSERVE_FITS') == '1' and 'superagent.experiment_worker' in sys.orig_argv:
    def offline(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise PermissionError('Network disabled in A29 training worker')
    sys.addaudithook(offline)
    from functools import wraps
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    def observe(cls):
        original = cls.fit
        @wraps(original)
        def fit(model, *args, **kwargs):
            with Path(os.environ['AB_A29_FIT_LOG']).open('a', encoding='utf-8') as file:
                file.write(json.dumps(dict(classifier=cls.__name__, candidate=sys.argv[2],
                    run_name=os.environ['AB_A29_RUN_NAME'])) + '\n')
            return original(model, *args, **kwargs)
        cls.fit = fit

    for classifier in (LogisticRegression, ExtraTreesClassifier, SVC):
        observe(classifier)
    if 'catboost' in sys.argv[2] or sys.argv[2] == 'supported_field_interactions':
        from catboost import CatBoostClassifier
        observe(CatBoostClassifier)
