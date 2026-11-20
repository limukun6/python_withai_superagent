"""Offline observation for the anonymous integration fixture's child processes only."""
import json
import os
from pathlib import Path
import sys


def offline(event, args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
        raise PermissionError('Network access is disabled in the anonymous acceptance test')


if os.environ.get('AB_B1_OFFLINE') == '1':
    sys.addaudithook(offline)
    if 'superagent.experiment_worker' in sys.orig_argv and os.environ.get('AB_B1_FIT_LOG'):
        from functools import wraps
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        def observe(cls):
            original = cls.fit

            @wraps(original)
            def fit(self, *args, **kwargs):
                with Path(os.environ['AB_B1_FIT_LOG']).open('a',encoding='utf-8') as file:
                    file.write(json.dumps({'classifier':cls.__name__,'candidate':sys.argv[2]}) + chr(10))
                return original(self,*args,**kwargs)
            cls.fit = fit

        for cls in (LogisticRegression,ExtraTreesClassifier,SVC):
            observe(cls)
        if len(sys.argv) > 2 and ('catboost' in sys.argv[2] or sys.argv[2] == 'supported_field_interactions'):
            from catboost import CatBoostClassifier
            observe(CatBoostClassifier)
