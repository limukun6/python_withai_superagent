"""Invoke the official CLI prediction entry with training and network forbidden."""
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier


def no_fit(*args,**kwargs):raise RuntimeError('Classifier fitting forbidden in portable prediction validation')


for classifier in (LogisticRegression,ExtraTreesClassifier,SVC,LinearSVC,DummyClassifier):classifier.fit=no_fit
try:
    from catboost import CatBoostClassifier
    CatBoostClassifier.fit=no_fit
except ImportError:pass


def no_network(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
        raise PermissionError('Network forbidden in portable prediction validation')


sys.addaudithook(no_network)
from superagent.cli import main
if __name__=='__main__':
    sys.argv=['superagent','predict',*sys.argv[1:]]
    main()
