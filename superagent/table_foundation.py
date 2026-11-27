"""Optional pinned, offline in-context classifier for numeric or categorical tables."""
from functools import lru_cache,partial
import hashlib
from importlib.metadata import PackageNotFoundError,version
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
def _verified(path,identity):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()==SHA256


def checkpoint():
    value=os.environ.get(WEIGHT_ENV)
    if not value or not find_spec("tabicl") or not find_spec("torch"):
        return None
    path=Path(value).expanduser().resolve()
    try:
        stat=path.stat()
        if version("tabicl")!="2.0.3" or not path.is_file() or stat.st_size!=SIZE:
            return None
        return path if _verified(path,(stat.st_ino,stat.st_mtime_ns,stat.st_ctime_ns)) else None
    except (OSError,PackageNotFoundError):
        return None


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        hardware=device()
        self.classifier=TabICLClassifier(n_estimators=VIEWS,batch_size=1,model_path=path,allow_auto_download=False,
                device=hardware,use_amp=False,use_fa3=False,kv_cache=True,n_jobs=1,random_state=173)
        self.classifier.fit(self._data(x),y)
        self.classes_=self.classifier.classes_
        self.training_metadata=dict(physical_classifier_calls=1,preprocessing_ensemble_members=VIEWS,
            pretrained_weight_updates=0,context_rows=len(y),device=hardware,weight_sha256=SHA256)
        return self

    def __getstate__(self):
        # TabICL supplies CPU-serializable caches. Replace the absolute checkpoint
        # path and choose hardware on the receiving computer during the first prediction.
        state=self.classifier.__getstate__() if self.classifier is not None else dict(self._saved)
        state["model_path"]=None
        state["device"]=None
        return dict(kind=self.kind,classes_=self.classes_,training_metadata=self.training_metadata,classifier=None,_saved=state)

    def predict_proba(self,x):
        if self.classifier is None:
            from tabicl import TabICLClassifier
            path=checkpoint()
            if path is None:
                raise ValueError("Verified TabICLv2 weights are required for this fitted solution")
            state=dict(self._saved,model_path=path,device=device(),allow_auto_download=False)
            restored=TabICLClassifier.__new__(TabICLClassifier)
            restored.__setstate__(state)
            self.classifier=restored
            self._saved=None
        return self.classifier.predict_proba(self._data(x))

    def predict(self,x):
        return self.classes_[np.asarray(self.predict_proba(x)).argmax(axis=1)]


def fit_predict(x,y,query,*,kind):
    model=FrozenTableClassifier(kind).fit(x,y)
    return model,model.predict(query)


def candidate(kind):
    return Candidate(NAME,dict(encoder=MODEL,revision=REVISION,weight_sha256=SHA256,ensemble_views=VIEWS,
                     representation="frozen tabular in-context column and row embeddings"),
                     partial(fit_predict,kind=kind))
