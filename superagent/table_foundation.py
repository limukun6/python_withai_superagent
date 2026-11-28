"""Optional pinned, offline in-context classifier for numeric or categorical tables."""
from functools import lru_cache, partial
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .models import Candidate

NAME="tabicl_v2_frozen_classifier"
WEIGHT_ENV="SUPERAGENT_TABICL_V2_WEIGHTS"
MODEL="jingang/TabICL"
REVISION="4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
FILENAME="tabicl-classifier-v2-20260212.ckpt"
SIZE=110368038
SHA256="bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
VIEWS=4


@lru_cache(maxsize=8)
def _verified(path, identity):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == SHA256


def checkpoint(environment=None):
    environment = os.environ if environment is None else environment
    if not find_spec("tabicl") or not find_spec("torch"):
        return None
    try:
        if version("tabicl") != "2.0.3":
            return None
        value = environment.get(WEIGHT_ENV)
        if not value:
            # A pinned local cache lookup never downloads or contacts the provider.
            from huggingface_hub import try_to_load_from_cache
            cache = environment.get("HF_HUB_CACHE")
            if not cache and environment.get("HF_HOME"):
                cache = Path(environment["HF_HOME"]).expanduser() / "hub"
            value = try_to_load_from_cache(MODEL, FILENAME, revision=REVISION, cache_dir=cache)
            if not isinstance(value, str):
                return None
        path = Path(value).expanduser().resolve()
        stat = path.stat()
        if not path.is_file() or stat.st_size != SIZE:
            return None
        identity = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns)
        return path if _verified(path, identity) else None
    except (OSError, ImportError, PackageNotFoundError):
        return None


def device():
    from .compute_resources import MIB, device_for
    return device_for(1024*MIB)


def resource():
    return dict(name="tabicl_v2",environment_variable=WEIGHT_ENV,path_kind="file",model=MODEL,
                revision=REVISION,file=dict(bytes=SIZE,sha256=SHA256))


class FrozenTableClassifier:
    def __init__(self,kind):
        self.kind=kind
        self.classifier=None
        self._saved=None

    def _data(self,x):
        if self.kind=="categorical_tabular":
            return pd.DataFrame(np.asarray(x),dtype=object)
        return pd.DataFrame(np.asarray(x,dtype=float))

    def fit(self,x,y):
        from tabicl import TabICLClassifier
        path=checkpoint()
        if path is None:
            raise ValueError("Verified TabICLv2 weights are unavailable")
        from .compute_resources import note
        hardware=device()
        note("tabular in-context fit",hardware)
        self.classifier=TabICLClassifier(n_estimators=VIEWS,batch_size=1,model_path=path,allow_auto_download=False,
                device=hardware,use_amp=False,use_fa3=False,kv_cache=True,n_jobs=1,random_state=173)
        self.classifier.fit(self._data(x),y)
        self.classes_=self.classifier.classes_
        self.training_metadata=dict(physical_classifier_calls=1,preprocessing_ensemble_members=VIEWS,
            pretrained_weight_updates=0,context_rows=len(y),device=hardware,weight_sha256=SHA256)
        return self

    def __getstate__(self):
        if self.classifier is None:
            if self._saved is None:
                return self.__dict__.copy()
            state = self._saved.copy()
        else:
            # Ask the pinned estimator for its portable state without transferring
            # the device-specific attention cache to CPU during serialization.
            estimator = type(self.classifier).__new__(type(self.classifier))
            estimator.__dict__ = self.classifier.__dict__.copy()
            estimator._save_kv_cache = False
            state = estimator.__getstate__()
        metadata = state["_persistence_metadata"].copy()
        if metadata["saved_training_data"]:
            state["model_kv_cache_"] = None
            metadata["saved_kv_cache"] = False
        state.update(_persistence_metadata=metadata, model_path=None, device=None)
        return dict(kind=self.kind, classes_=self.classes_, training_metadata=self.training_metadata,
                    classifier=None, _saved=state)

    def _restore(self, device_name):
        if self.classifier is None:
            from tabicl import TabICLClassifier
            path=checkpoint()
            if path is None:
                raise ValueError("Verified TabICLv2 weights are required for this fitted solution")
            state=dict(self._saved,model_path=path,device=device_name,allow_auto_download=False)
            restored=TabICLClassifier.__new__(TabICLClassifier)
            restored.__setstate__(state)
            if restored.kv_cache and restored.model_kv_cache_ is None:
                # Recreate cached projections from the saved fitted context, without fitting.
                restored._build_kv_cache()
            self.classifier=restored
            self._saved=None
    def predict_proba(self,x):
        from .compute_resources import device_error, disable, note
        preferred = str(self.classifier.device) if self.classifier is not None else device()
        try:
            self._restore(preferred)
            out=self.classifier.predict_proba(self._data(x))
        except (RuntimeError,NotImplementedError) as error:
            if preferred=='cpu' or not device_error(error):
                raise
            disable(preferred,'tabular in-context inference',error)
            state=self.__getstate__()
            self.classifier,self._saved=None,state['_saved']
            self._restore('cpu')
            preferred='cpu'
            out=self.classifier.predict_proba(self._data(x))
        note('tabular in-context inference',preferred,len(x))
        return out

    def predict(self,x):
        return self.classes_[np.asarray(self.predict_proba(x)).argmax(axis=1)]


def fit_predict(x,y,query,*,kind):
    model=FrozenTableClassifier(kind).fit(x,y)
    return model,model.predict(query)


def candidate(kind):
    return Candidate(NAME,dict(encoder=MODEL,revision=REVISION,weight_sha256=SHA256,ensemble_views=VIEWS,
                     representation="frozen tabular in-context column and row embeddings"),
                     partial(fit_predict,kind=kind))
