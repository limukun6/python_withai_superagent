"""A local frozen MobileNetV3-Small feature extractor with one supervised linear head."""
from functools import lru_cache
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import os
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


NAME = 'image_mobilenet_v3_small_linear'
WEIGHT_ID = 'MobileNet_V3_Small_Weights.IMAGENET1K_V1'
WEIGHT_BYTES = 10306551
WEIGHT_SHA256 = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'
WEIGHT_ENV = 'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS'


@lru_cache(maxsize=8)
def _weight_digest(path, identity):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def weights_file():
    supplied = os.environ.get(WEIGHT_ENV)
    if not supplied or any(find_spec(module) is None for module in ('torch', 'torchvision')):
        return None
    path = Path(supplied).expanduser().resolve()
    if not path.is_file():
        return None
    meta = path.stat()
    identity = (meta.st_size, meta.st_ino, meta.st_mtime_ns, meta.st_ctime_ns)
    return path if meta.st_size == WEIGHT_BYTES and _weight_digest(path, identity) == WEIGHT_SHA256 else None


@lru_cache(maxsize=1)
def _network(path):
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    torch.set_num_threads(min(3, torch.get_num_threads()))
    network = mobilenet_v3_small(weights=None)
    state = torch.load(path, map_location='cpu', weights_only=True)
    network.load_state_dict(state, strict=True)
    network.classifier = torch.nn.Identity()
    return network.eval().requires_grad_(False), MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()


def decoded_images(rows):
    data = np.asarray(rows)
    if data.ndim == 3:
        data = data[..., None]
    if (data.dtype != np.uint8 or data.ndim != 4 or data.shape[3] not in (1, 3)
            or min(data.shape[:3]) < 1):
        raise ValueError('Frozen vision features require decoded 8-bit grayscale or RGB images')
    return data


class FrozenVisionFeatures(TransformerMixin, BaseEstimator):
    def fit(self, rows, y=None):
        self.shape_ = decoded_images(rows).shape[1:]
        self.versions_ = {p:version(p) for p in ('torch','torchvision')}
        return self

    def transform(self, rows):
        import torch

        path = weights_file()
        if path is None:
            raise ValueError(f'Set {WEIGHT_ENV} to the verified {WEIGHT_ID} checkpoint')
        data = decoded_images(rows)
        if data.shape[1:] != self.shape_ or any(version(p) != v for p,v in self.versions_.items()):
            raise ValueError('Image dimensions or frozen vision runtime versions differ from training')
        net, preprocess = _network(path)
        output = np.empty((len(data), 576), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(data), 24):
                raw = np.ascontiguousarray(data[start:start+24].transpose(0,3,1,2))
                tensor = torch.from_numpy(raw)
                if tensor.shape[1] == 1:
                    tensor = tensor.expand(-1,3,-1,-1)
                features = net(preprocess(tensor))
                output[start:start+len(raw)] = torch.nn.functional.normalize(features, dim=1).numpy()
        if not np.isfinite(output).all():
            raise ValueError('Frozen vision extractor produced nonfinite features')
        return output


def fit_vision(rows, labels, query):
    features = FrozenVisionFeatures()
    classifier = LogisticRegression(C=10., max_iter=2000, random_state=42)
    model = Pipeline([('features',features), ('classifier',classifier)])
    model.fit(rows, labels)
    model.training_metadata = dict(frozen_downstream_encoder=WEIGHT_ID, weight_sha256=WEIGHT_SHA256,
        backend='cpu', versions=features.versions_, trainable_encoder_parameters=0, embedding_dimensions=576)
    return model, model.predict(query)
