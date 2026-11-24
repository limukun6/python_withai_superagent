"""Discover single-fit CPU classifiers advertised by installed Python distributions."""
from collections import Counter
from functools import lru_cache
from importlib import metadata
import inspect
import re

from sklearn.base import BaseEstimator
from sklearn.utils import get_tags

from .generated_pipeline import classifier_inventory


ENTRY_POINT_GROUP = "superagent.classifiers.v1"


def _distribution(point):
    name = point.dist.metadata["Name"] or ""
    return re.sub(r"[-_.]+", "-", name).lower()


@lru_cache(maxsize=1)
def installed_extensions():
    """Only package metadata can supply a constructor. Task files never supply imports."""
    points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    aliases = Counter((_distribution(point), point.name) for point in points)
    accepted = {}
    rejected = {}
    for point in sorted(points, key=lambda item: (_distribution(item), item.name, item.value)):
        distribution = _distribution(point)
        alias = (distribution, point.name)
        # A different distribution, version or import target must never resolve an old registry entry.
        key = f"{distribution}@{point.dist.version}/{point.name}={point.value}"
        if not distribution or aliases[alias] != 1:
            rejected[key] = "missing distribution name or duplicate classifier alias"
            continue
        try:
            cls = point.load()
            if not inspect.isclass(cls) or not issubclass(cls, BaseEstimator):
                raise ValueError("entry point must export an estimator class")
            estimator = cls()
            params = estimator.get_params(deep=False)
            tags = get_tags(estimator)
            if tags.classifier_tags is None or not tags.target_tags.single_output:
                raise ValueError("requires a single-output classifier")
            if not tags.input_tags.two_d_array or not callable(getattr(estimator, "predict", None)):
                raise ValueError("requires a two-dimensional feature matrix and predict")
            if not callable(getattr(estimator, "fit", None)) or cls.__name__.endswith("CV"):
                raise ValueError("requires one supervised fit")
            if {"estimator", "estimators", "base_estimator", "cv"}.intersection(params):
                raise ValueError("implicit component fits require separate accounting")
        except Exception as error:
            # A broken optional installation must not hide the other executable methods.
            rejected[key] = type(error).__name__
            continue
        accepted[key] = cls
    return accepted, rejected


@lru_cache(maxsize=1)
def runtime_classifiers():
    extensions, _ = installed_extensions()
    return {**classifier_inventory(), **extensions}


class ProviderRequirement:
    """Verify the same package registration when a saved model is loaded in another process."""

    def __init__(self, key):
        if key not in installed_extensions()[0]:
            raise ValueError("Required installed classifier version or entry point is unavailable: " + key)
        self.key = key

    def __reduce__(self):
        return type(self), (self.key,)
