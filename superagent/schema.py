"""Versioned data-only task manifests; no embedded code or task-name routing."""
from dataclasses import dataclass
import csv
import io
import json
from pathlib import PurePosixPath,PureWindowsPath

import numpy as np

KINDS={'numeric_tabular','mixed_tabular','text_tabular','single_text','image'}
MISSING={'','na','nan','none','null','?'}


@dataclass
class MixedRows:
    values: np.ndarray
    numeric_columns: tuple[int,...]
    columns: tuple[str,...]

    def __len__(self):return len(self.values)
    @property
    def shape(self):return self.values.shape
    def __getitem__(self,index):
        values=self.values[index]
        return MixedRows(values,self.numeric_columns,self.columns) if values.ndim==2 else values
    def take(self,indices):return self[indices]


@dataclass
class TextTableRows(MixedRows):
    text_columns: tuple[int,...] = ()

    def __getitem__(self,index):
        values=self.values[index]
        return TextTableRows(values,self.numeric_columns,self.columns,self.text_columns) if values.ndim==2 else values


def read_manifest(archive):
    names=[n for n in archive.namelist() if not n.startswith('__MACOSX/') and PurePosixPath(n).name=='manifest.json']
    if not names:return None
    if len(names)!=1:raise ValueError('Expected one task manifest')
    value=json.loads(archive.read(names[0]).decode('utf-8'))
    if value.get('schema_version')!=1:raise ValueError('Unsupported manifest schema version')
    if value.get('objective','classification')!='classification':raise ValueError('Only classification is supported')
    from .metrics import validate_metric
    validate_metric(value.get('metric','accuracy'))
    if value.get('modality') not in ('tabular','text','pair_text','image'):
        raise ValueError('Unsupported declared modality')
    value['_manifest_parent']=str(PurePosixPath(names[0]).parent)
    return value


def _member(archive,manifest,split,field='path'):
    descriptor=manifest.get('splits',{}).get(split)
    if not isinstance(descriptor,dict):raise ValueError(f'Manifest has no {split} split')
    raw=descriptor.get(field)
    if not isinstance(raw,str) or not raw:raise ValueError('Split path must be a nonempty string')
    relative=PurePosixPath(raw.replace('\\','/'))
    if relative.is_absolute() or PureWindowsPath(raw).drive or '..' in relative.parts:raise ValueError('Split path must stay inside its archive')
    from .task_files import evaluation_path
    if split!='test' and evaluation_path(relative,split):raise ValueError('Development split cannot point at an evaluation member')
    path=(PurePosixPath(manifest['_manifest_parent'])/relative).as_posix()
    if split!='test' and evaluation_path(path,split):
        raise ValueError('Development split cannot point inside a test path')
    if archive.namelist().count(path)!=1:raise ValueError('Missing or ambiguous split archive member')
    return descriptor,path


def _validate_labels(labels,n):
    labels=np.asarray(labels)
    if labels.ndim!=1 or len(labels)!=n:raise ValueError('Expected one target per feature row')
    if labels.dtype.kind in 'fc' and not np.isfinite(labels).all():raise ValueError('Labels must be finite')
    if labels.dtype.kind not in 'biufUS':raise ValueError('Labels must be numeric or strings, not pickled objects')
    if labels.dtype.kind in 'US' and any(not str(v).strip() for v in labels):raise ValueError('Empty target values are unsupported')
    return labels


def load_manifest_split(archive,manifest,split,include_labels=True):
    if manifest.get('splits',{}).get(split,{}).get('format')=='image_folder':
        from .folder_images import load_folder_split
        return load_folder_split(archive,manifest,split,include_labels)
    descriptor,path=_member(archive,manifest,split)
    modality=manifest['modality'];fmt=descriptor.get('format')
    spec={'manifest_schema_version':1,'modality':modality,'metric':manifest.get('metric','accuracy'),'test_accessed':split=='test',
          'schema_origin':manifest.get('_schema_origin','manifest_v1'),'input_format':fmt,
          'input_schema':{key:manifest[key] for key in ('modality','columns','target','pair_label_one','pair_semantics','image_preprocessing') if key in manifest}}
    if 'positive_label' in manifest:spec['positive_label']=manifest['positive_label']
    if '_prediction_member' in manifest:spec['prediction_member']=manifest['_prediction_member']
    if 'evaluation_protocol' in manifest:spec['evaluation_protocol']=manifest['evaluation_protocol']
    if fmt in ('npz','npy'):
        if modality not in ('tabular','image'):raise ValueError('NPZ adapter is for numeric tables or images')
        if fmt=='npz':
            with np.load(io.BytesIO(archive.read(path)),allow_pickle=False) as arrays:
                x=np.array(arrays[descriptor.get('features_key','X')],copy=True)
                y=np.array(arrays[descriptor.get('target_key','y')],copy=True) if include_labels else None
        else:
            x=np.load(io.BytesIO(archive.read(path)),allow_pickle=False)
            y=np.load(io.BytesIO(archive.read(_member(archive,manifest,split,'target_path')[1])),allow_pickle=False) if include_labels else None
        if x.dtype.kind not in 'biuf' or x.ndim not in ((2,) if modality=='tabular' else (3,4)):
            raise ValueError('Expected numeric rows or N,H,W[,C] images')
        if np.isinf(x).any():raise ValueError('Infinite features are unsupported')
        if modality=='image' and not np.isfinite(x).all():raise ValueError('Images must have finite pixel values')
        kind='numeric_tabular' if modality=='tabular' else 'image'
        spec['feature_shape']=list(x.shape[1:])
    elif fmt in ('csv','tsv','parquet'):
        from .table_input import table
        columns=manifest.get('columns')
        if not isinstance(columns,list) or not columns or any(not isinstance(c,dict) for c in columns):
            raise ValueError('Declare input columns and their types')
        names=[c.get('name') for c in columns];types=[c.get('type') for c in columns]
        target=manifest.get('target')
        if len(set(names))!=len(names) or target in names:
            raise ValueError('Input columns must be unique, present, and exclude target')
        requested=names+[target] if include_labels else names
        _,reader=table(archive,path,fmt,requested)
        feature_rows=[];labels=[]
        for row in reader:
            feature_rows.append(row[:len(names)])
            if include_labels:labels.append(row[-1])
        if not feature_rows:raise ValueError('Empty split')
        if modality=='image':
            if len(names)!=1 or types!=['image_path']:raise ValueError('Image CSV requires exactly one image_path input column')
            from .image_files import load_image_rows,DECODER_POLICY
            from importlib.metadata import version
            x,contract=load_image_rows(archive,manifest,split,[row[0] for row in feature_rows]);kind='image'
            spec['feature_shape']=list(x.shape[1:]);spec['image_decoder_policy']=DECODER_POLICY
            spec['image_decoder_version']=version('Pillow')
            spec['input_schema']['image_preprocessing']=contract
        elif modality=='tabular':
            if any(t not in ('numeric','categorical','text') for t in types):raise ValueError('Declare numeric/categorical/text table types')
            numeric=tuple(i for i,t in enumerate(types) if t=='numeric')
            text_columns=tuple(i for i,t in enumerate(types) if t=='text')
            values=np.asarray(feature_rows,dtype=object)
            for i in numeric:
                values[:,i]=[float(v) if str(v).strip().casefold() not in MISSING else np.nan for v in values[:,i]]
                if np.isinf(np.asarray(values[:,i],dtype=float)).any():raise ValueError('Infinite features are unsupported')
            if text_columns:x=TextTableRows(values,numeric,tuple(names),text_columns);kind='text_tabular'
            elif len(numeric)==len(names):x=values.astype(float);kind='numeric_tabular'
            elif not numeric:x=values;kind='categorical_tabular'
            else:x=MixedRows(values,numeric,tuple(names));kind='mixed_tabular'
            spec['feature_shape']=[len(names)]
        else:
            if any(t!='text' for t in types) or modality=='pair_text' and len(names)!=2:
                raise ValueError('Text modality has incompatible input columns')
            relation=modality=='pair_text' and manifest.get('pair_semantics')=='relation'
            if modality=='pair_text' and not relation:
                kind='pair_text';x=[tuple(row) for row in feature_rows]
            elif len(names)==1:
                kind='single_text';x=[row[0] for row in feature_rows]
            else:
                kind='text_tabular';x=TextTableRows(np.asarray(feature_rows,dtype=object),(),tuple(names),tuple(range(len(names))))
                spec['feature_shape']=[len(names)]
        y=np.asarray(labels) if include_labels else None
        if kind=='pair_text' and include_labels:
            if not set(labels)<={'0','1'}:raise ValueError('Pair targets must be 0/1 with explicit label meaning')
            if manifest.get('pair_semantics')=='symmetric':
                if manifest.get('pair_label_one')!='equivalent':
                    raise ValueError('Symmetric pair tasks require pair_label_one=equivalent, not left/right preference')
            elif manifest.get('pair_label_one')!='left':
                raise ValueError('Declare pair_label_one=left explicitly')
            y=np.asarray(labels,dtype=int)
        spec['input_columns']=[{'name':name,'type':t} for name,t in zip(names,types)]
    else:raise ValueError('Supported split formats are csv, npz and numeric npy pairs')
    if include_labels:
        y=_validate_labels(y,len(x))
        if 'positive_label' in spec and y.dtype.kind in 'US':spec['positive_label']=str(spec['positive_label'])
    if len(x)==0:raise ValueError('Empty split')
    spec['goal_sources']={'primary_metric':('default_accuracy' if manifest.get('_schema_origin')=='native_v1' or 'metric' not in manifest
                                           else 'declared_manifest')}
    if manifest.get('_metric_source'):
        spec['goal_sources']['primary_metric']=manifest['_metric_source']
    spec['kind']=kind
    return kind,x,y,spec
