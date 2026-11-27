"""Isolate a requested training seed in the fitting worker without altering ordinary method settings."""
from contextlib import contextmanager
from contextvars import ContextVar

from sklearn.pipeline import Pipeline

_seed = ContextVar("superagent_training_seed", default=None)


@contextmanager
def training_seed(value):
    if value is not None and (type(value) is not int or not 0 <= value < 2**31):
        raise ValueError("Invalid classifier training seed")
    token = _seed.set(value)
    try:
        yield
    finally:
        _seed.reset(token)


def selected_seed(default=None):
    value = _seed.get()
    return default if value is None else value


def fitted_seed(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1].get_params(deep=False).get("random_state")
    estimator = getattr(model, "classifier", model)
    if type(estimator).__module__.split(".")[0] == "catboost":
        return estimator.get_params().get("random_seed")
    return None
