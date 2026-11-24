"""Conventional split/class/image ZIP layouts, without test-label reads."""
from pathlib import PurePosixPath

import numpy as np

from .image_files import load_image_rows, DECODER_POLICY


EXTENSIONS={'.png','.jpg','.jpeg'}
DEFAULT_CONTRACT={'width':64,'height':64,'mode':'RGB','resize':'stretch_bilinear'}


def _images(archive):
    names=[]
    for name in archive.namelist():
        path=PurePosixPath(name)
        if name.startswith('__MACOSX/') or path.suffix.casefold() not in EXTENSIONS:continue
        if path.is_absolute() or '..' in path.parts or '\\' in name:
            raise ValueError('Image archive path must remain inside its package')
        names.append(name)
    if len(set(names))!=len(names):raise ValueError('Ambiguous duplicate image paths')
    return sorted(names)


def _prefix(archive,split,required=False):
    aliases={'train':{'train'},'validation':{'validation','valid','val'},'test':{'test'}}[split]
    candidates=set()
    for name in _images(archive):
        parts=PurePosixPath(name).parts
        for i,part in enumerate(parts[:-1]):
            if part.casefold() in aliases and (split=='test' or len(parts)-i>=3):
                candidates.add('/'.join(parts[:i+1])+'/')
    if len(candidates)>1:raise ValueError('Ambiguous image split roots; declare a manifest')
    if not candidates:
        if required:raise ValueError('Missing image folder split: '+split)
        return None
    return next(iter(candidates))


def infer_folder_manifest(archive):
    conventional={'train.csv','train.npz','X_train.npy','C_train.npy'}
    if any(PurePosixPath(n).name in conventional for n in archive.namelist()):return None
    train=_prefix(archive,'train')
    if train is None:return None
    manifest={'schema_version':1,'objective':'classification','metric':'accuracy','modality':'image',
        '_manifest_parent':'.','_schema_origin':'folder_v1','image_preprocessing':dict(DEFAULT_CONTRACT),
        'splits':{'train':{'format':'image_folder','prefix':train}}}
    validation=_prefix(archive,'validation')
    if validation:manifest['splits']['validation']={'format':'image_folder','prefix':validation}
    return manifest


def load_folder_split(archive,manifest,split,include_labels=True):
    descriptor=manifest.get('splits',{}).get(split,{})
    prefix=descriptor.get('prefix')
    if not isinstance(prefix,str) or not prefix.endswith('/') or '..' in PurePosixPath(prefix).parts or prefix.startswith('/'):
        raise ValueError('Invalid image-folder prefix')
    if split!='test' and any(p.casefold()=='test' for p in PurePosixPath(prefix).parts):
        raise ValueError('Development images cannot come from a test folder')
    names=[n for n in _images(archive) if n.startswith(prefix)]
    if not names:raise ValueError('Empty image split')
    labels=[]
    if include_labels:
        for name in names:
            remaining=PurePosixPath(name[len(prefix):]).parts
            if len(remaining)<2:raise ValueError('Training image folder requires one class directory per image')
            labels.append(remaining[0])
    x,contract=load_image_rows(archive,manifest,split,names)
    from importlib.metadata import version
    spec={'manifest_schema_version':1,'modality':'image','metric':manifest.get('metric','accuracy'),
        'goal_sources':{'primary_metric':'default_accuracy' if manifest.get('_schema_origin')=='folder_v1' or 'metric' not in manifest else 'declared_manifest'},
        'test_accessed':split=='test','schema_origin':'folder_v1','input_format':'image_folder',
        'feature_shape':list(x.shape[1:]),'image_decoder_policy':DECODER_POLICY,
        'image_decoder_version':version('Pillow'),'kind':'image',
        'input_schema':{'modality':'image','image_preprocessing':contract}}
    return 'image',x,np.asarray(labels) if include_labels else None,spec


def prediction_folder_manifest(archive,frozen_spec):
    return {'schema_version':1,'modality':'image','metric':frozen_spec['metric'],
        '_manifest_parent':'.','image_preprocessing':frozen_spec['input_schema']['image_preprocessing'],
        'splits':{'test':{'format':'image_folder','prefix':_prefix(archive,'test',required=True)}}}
