"""Encode categorical levels on fitting rows and ship the learned lookup with the classifier."""
from functools import partial
from importlib.util import find_spec
import os
from pathlib import Path

from .compute_resources import cpu_threads

import numpy as np
from scipy.sparse import csr_matrix

from . import ab_sentence as sentence
from . import local_sentence_backbone as local
from .models import Candidate, _clean_categories, _tabular_catboost

LIBRARY = 'SUPERAGENT_CELL_ENCODER_LIBRARY'
PREFIX = 'catboost_cell_'
SCRATCH = 'cooccurrence'


def resources():
    found = {SCRATCH: {'name': 'Across-field co-occurrence', 'bytes': 0}}
    location = os.environ.get(LIBRARY)
    if not location or any(find_spec(p) is None for p in ('torch', 'transformers', 'tokenizers', 'safetensors')):
        return found
    root = Path(location)
    known = [dict(model=sentence.MODEL, files=sentence.FILES, pool='mean', prefix='', limit=256),
             dict(model=sentence.DEEP_MODEL, files=sentence.DEEP_FILES, pool='mean', prefix='', limit=128),
             *(dict(spec, limit=512) for spec in sentence.ALTERNATIVES.values())]
    for spec in known:
        path = root / spec['model'].split('/')[-1]
        if all((path / n).is_file() and (path / n).stat().st_size == size for n, (size, _) in spec['files'].items()):
            key = spec['files']['model.safetensors'][1]
            found[key] = dict(name=spec['model'], path=path, pinned=spec, bytes=spec['files']['model.safetensors'][0])
    for package in sorted(root.glob('*/modules.json')):
        try:
            definition = local.describe(package.parent)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError):
            continue
        weights = definition['files']['model.safetensors']
        if weights['sha256'] not in found:
            found[definition['signature']] = dict(name=package.parent.name, path=package.parent,
                                                   definition=definition, bytes=weights['bytes'])
    return found


def cell_vectors(asset, values):
    if 'definition' in asset:
        return local.encode(asset['path'], asset['definition'], values)
    spec, root = asset['pinned'], asset['path']
    for name, (size, digest) in spec['files'].items():
        path = root / name
        st = path.stat()
        if st.st_size != size or sentence._checksum(path, (size, st.st_ino, st.st_mtime_ns, st.st_ctime_ns)) != digest:
            raise ValueError('Changed pinned cell encoder ' + name)
    tokenizer, model = sentence._encoder(root, spec['limit'])
    return sentence.encode_frozen(values, tokenizer, model, prefix=spec['prefix'], pool=spec['pool'])[0]


def cooccurrences(codes, sizes):
    n, width = codes.shape
    offsets = np.r_[0, np.cumsum(sizes)]
    row = np.repeat(np.arange(n), width)
    col = (codes + offsets[:-1]).ravel()
    matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, offsets[-1]))
    counts = (matrix.T @ matrix).toarray()
    totals = counts.diagonal().copy()
    lift = np.maximum(0, np.log(np.maximum(1e-12, counts*n / (totals[:, None]*totals))))
    for a, b in zip(offsets[:-1], offsets[1:]):
        lift[a:b, a:b] = 0.
    return np.split(lift, offsets[1:-1])


class CellGeometryTree:
    def fit(self, rows, labels, key, asset):
        x = _clean_categories(rows)
        levels, codes = zip(*(np.unique(col, return_inverse=True) for col in x.T))
        if sum(map(len, levels)) > 2048:
            raise ValueError('Cell geometry exceeds its fit-time vocabulary bound')
        if key == SCRATCH:
            vectors = cooccurrences(np.column_stack(codes), [len(v) for v in levels])
        else:
            vocab = sorted(set(np.concatenate(levels)))
            lookup = dict(zip(vocab, cell_vectors(asset, vocab), strict=True))
            vectors = [np.stack([lookup[v] for v in field]) for field in levels]
        self.encoded = []
        for field, values in zip(levels, vectors, strict=True):
            centered = values - values.mean(0)
            rank = min(4, len(field)-1, centered.shape[1])
            if not rank:
                self.encoded.append(({}, 0))
                continue
            u, s, _ = np.linalg.svd(centered, full_matrices=False)
            coords = (u[:, :rank] * s[:rank]).astype(np.float32)
            self.encoded.append((dict(zip(field.tolist(), coords)), rank))
        self.classifier = _tabular_catboost(labels, depth=6, l2_leaf_reg=3.)
        self.classifier.set_params(thread_count=cpu_threads(4))
        self.classifier.fit(self.matrix(x), labels, cat_features=list(range(x.shape[1])))
        self.classes_ = self.classifier.classes_
        self.training_metadata = dict(cell_representation=asset['name'], identity=key,
                                      fitted_levels=sum(map(len, levels)), classifier_fits=1,
                                      coordinate_fit='training vocabulary', unknown_levels='training centroid')
        return self

    def matrix(self, rows):
        x = _clean_categories(rows)
        if x.shape[1] != len(self.encoded):
            raise ValueError('Unexpected number of categorical fields')
        columns = [x.astype(object)]
        for j, (coordinates, rank) in enumerate(self.encoded):
            if rank:
                levels, inverse = np.unique(x[:, j], return_inverse=True)
                zero = np.zeros(rank, dtype=np.float32)
                columns.append(np.stack([coordinates.get(v, zero) for v in levels])[inverse])
        return np.column_stack(columns)

    def predict_proba(self, rows):
        return self.classifier.predict_proba(self.matrix(rows))

    def predict(self, rows):
        return self.classifier.predict(self.matrix(rows)).reshape(-1)


def fit_cell(key, rows, labels, query):
    model = CellGeometryTree().fit(rows, labels, key, resources()[key])
    return model, model.predict(query)


def candidate(key):
    params = dict(depth=6, iterations=600, l2_leaf_reg=3., components_per_field=4,
                  representation='native categories and fitting-only categorical level coordinates')
    if key != SCRATCH:
        params['encoder'] = key
    return Candidate(PREFIX + key, params, partial(fit_cell, key))


def capability(key):
    method = candidate(key)
    return dict(id=method.family, params=method.params, representation=method.params['representation'],
                estimator='catboost', origin='b_port', resource='cpu' if key == SCRATCH else 'local_frozen_encoder_cpu',
                classifier_fits=1, refit_classifier_fits=1)
