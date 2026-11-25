"""Run-local, train-input-keyed reuse of unsupervised paired TF-IDF features."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

import joblib

_ACTIVE=ContextVar('superagent_pair_feature_cache',default=None)


class FeatureCacheError(ValueError):pass


def _json(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def _digest(value):return hashlib.sha256(value).hexdigest()
def training_fingerprint(pairs):return _digest(_json(pairs).encode())


def _parameter(value):
    if value is None or type(value) in (str,int,float,bool):return value
    if isinstance(value,(tuple,list)):return [_parameter(v) for v in value]
    if isinstance(value,dict):return {str(k):_parameter(v) for k,v in value.items()}
    if isinstance(value,type):return {'python_type':value.__module__+'.'+value.__qualname__}
    raise FeatureCacheError('Unsupported representation parameter in the explicit cache key')


class PairFeatureCache:
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        source=Path(__file__).resolve().parent
        self.source_sha256=_digest(_json({p.name:_digest(p.read_bytes()) for p in sorted(source.glob('*.py'))}).encode())
        self.versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','joblib')}
        self.events=[];self.memory={}

    def identity(self,vectorizer,pairs):
        return {'format_version':1,'operation':'fit_training_pairs_then_left_minus_right',
            'representation':type(vectorizer).__module__+'.'+type(vectorizer).__qualname__,
            'parameters':_parameter(vectorizer.get_params(deep=False)),
            'training_input_sha256':training_fingerprint(pairs),'training_rows':len(pairs),
            'source_sha256':self.source_sha256,'python':platform.python_version(),'dependencies':self.versions,
            'labels_used_by_representation':False}

    def _read(self,entry,identity):
        try:
            metadata=json.loads((entry/'metadata.json').read_text())
            artifact=entry/'features.joblib'
            if metadata['identity']!=identity or _digest(artifact.read_bytes())!=metadata['artifact_sha256']:
                raise FeatureCacheError('Feature cache identity or artifact checksum mismatch')
            bundle=joblib.load(artifact)
            if bundle['train_matrix'].shape[0]!=identity['training_rows']:
                raise FeatureCacheError('Feature cache row count mismatch')
            if _parameter(bundle['vectorizer'].get_params(deep=False))!=identity['parameters']:
                raise FeatureCacheError('Cached representation parameters changed')
            return bundle,metadata
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise FeatureCacheError('Invalid published feature cache entry: '+str(error)) from error

    def fit(self,vectorizer,pairs,compute):
        started=time.monotonic();identity=self.identity(vectorizer,pairs)
        key=_digest(_json(identity).encode());entry=self.directory/key
        if entry.exists():
            bundle,metadata=self._read(entry,identity);status='disk_hit'
            vectorizer.__dict__.clear();vectorizer.__dict__.update(copy.deepcopy(bundle['vectorizer'].__dict__))
            matrix=bundle['train_matrix']
        else:
            matrix=compute();status='built'
            # Publish the complete directory atomically. Interrupted temporary
            # directories never count as hits; racing builders are both counted.
            temporary=Path(tempfile.mkdtemp(prefix='.'+key+'-',dir=self.directory))
            try:
                artifact=temporary/'features.joblib'
                joblib.dump({'vectorizer':vectorizer,'train_matrix':matrix},artifact)
                metadata={'identity':identity,'artifact_sha256':_digest(artifact.read_bytes())}
                (temporary/'metadata.json').write_text(_json(metadata)+'\n')
                try:os.rename(temporary,entry)
                except OSError as error:
                    if error.errno not in (errno.EEXIST,errno.ENOTEMPTY):raise
                    # The winning publisher must contain the same keyed representation.
                    _,metadata=self._read(entry,identity);status='built_with_concurrent_publisher'
            finally:
                if temporary.exists():shutil.rmtree(temporary)
        self.memory[id(vectorizer)]=(vectorizer,identity['training_input_sha256'],matrix)
        self.events.append({'event':status,'key':key,'training_input_sha256':identity['training_input_sha256'],
            'training_rows':len(pairs),'matrix_shape':list(matrix.shape),'elapsed_seconds':time.monotonic()-started,
            'artifact_sha256':metadata['artifact_sha256']})
        return matrix.copy()

    def transform(self,vectorizer,pairs,compute):
        saved=self.memory.get(id(vectorizer))
        if saved is not None and saved[0] is vectorizer and training_fingerprint(pairs)==saved[1]:
            self.events.append({'event':'in_process_training_reuse','training_input_sha256':saved[1]})
            return saved[2].copy()
        return compute()

    def report(self):
        return {'enabled':True,'scope':'same run, unsupervised paired TF-IDF only','events':list(self.events),
            'builds':sum(r['event'].startswith('built') for r in self.events),
            'disk_hits':sum(r['event']=='disk_hit' for r in self.events),
            'in_process_training_reuses':sum(r['event']=='in_process_training_reuse' for r in self.events)}


@contextmanager
def using_feature_cache(directory=None):
    cache=PairFeatureCache(directory) if directory else None
    token=_ACTIVE.set(cache)
    try:yield cache
    finally:_ACTIVE.reset(token)


def paired_features(vectorizer,pairs,fit,compute):
    cache=_ACTIVE.get()
    if cache is None:return compute()
    return cache.fit(vectorizer,pairs,compute) if fit else cache.transform(vectorizer,pairs,compute)
