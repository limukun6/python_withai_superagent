"""Conservative schema discovery from standard training-file layouts only.

The task/dataset name is never used. Ambiguous layouts or label semantics need
an explicit manifest. All inferred feature decisions use training contents.
"""
import io
from pathlib import PurePosixPath
import re
import statistics

import numpy as np

from .schema import MISSING
from .task_files import TABLE_FORMATS, evaluation_path
from .table_input import table, target_column, declared_metric

def find_member(archive,basenames,required=False):
    accepted={s.casefold() for s in basenames}
    found=[name for name in archive.namelist() if not name.startswith('__MACOSX/')
           and PurePosixPath(name).name.casefold() in accepted]
    if len(found)>1:raise ValueError('Ambiguous standard split files; supply one explicit manifest')
    if not found:
        if required:raise ValueError('Missing standard split file: '+', '.join(sorted(basenames)))
        return None
    path=PurePosixPath(found[0])
    if path.is_absolute() or '..' in path.parts or '\\' in found[0]:
        raise ValueError('Unsupported archive member path')
    return found[0]


def descriptor(archive,fmt,split):
    aliases={'train':['train'],'validation':['validation','val','valid'],'test':['test']}[split]
    if fmt in (*TABLE_FORMATS,'npz'):
        return {'format':fmt,'path':find_member(archive,[s+'.'+fmt for s in aliases],required=True)}
    return {'format':'npy','path':find_member(archive,['X_'+s+'.npy' for s in aliases],required=True),
            **({'target_path':find_member(archive,['y_'+s+'.npy' for s in aliases],required=True)} if split!='test' else {})}


def _column_type(values):
    present=[v for v in values if v.strip().casefold() not in MISSING]
    if not present:raise ValueError('An all-missing column has no identifiable type; declare it in a manifest')
    try:
        numeric=np.asarray([float(v) for v in present])
    except ValueError:pass
    else:
        if not np.isfinite(numeric).all():raise ValueError('Non-finite numeric column')
        return 'numeric'
    typical_length=statistics.median(map(len,present))
    typical_words=statistics.median(len(v.split()) for v in present)
    if (typical_length>=16 and typical_words>=3) or typical_length>=40:return 'text'
    return 'categorical'


def description(archive):
    from .format_sources import decode_text,task_documents
    names = task_documents(archive.namelist())
    if not names:
        return ''
    if len(names)!=1 or archive.getinfo(names[0]).file_size > 256*1024:
        raise ValueError('Task instructions exceed the reading limit or have multiple possible sources')
    return decode_text(archive.read(names[0])).replace(chr(96), '').replace('*', '')


def named_table(text,split):
    words={'train':r'(?:training|train|训练)','test':r'(?:prediction|testing|test|预测|测试)'}[split]
    field=r'(?:\s*(?:data\s*)?(?:file|table)|(?:文件|数据文件))'
    found=set(re.findall(words+field+r'\s*(?:is\s+|为|是|[:：=]\s*)?[\"\']?([\w./-]+\.(?:csv|tsv|parquet))\b',text,re.I))
    if len(found)>1:
        raise ValueError('Conflicting file declarations for '+split)
    if not found:
        return None
    path=PurePosixPath(found.pop())
    if path.is_absolute() or '..' in path.parts or (split=='train' and evaluation_path(path,split)):
        raise ValueError('Training and prediction input files must be distinct safe members')
    return path.as_posix()


def declared_member(archive,path):
    single=getattr(archive,'standalone',None)
    if single and PurePosixPath(path).name==single[0]:
        return single[1]
    matches=[n for n in archive.namelist() if '__MACOSX' not in PurePosixPath(n).parts and
             (n==path or n.endswith('/'+path))]
    if len(matches)!=1:
        raise ValueError('Declared table member is absent or ambiguous: '+path)
    return matches[0]


def declared_preference(text, target, names):
    """Return canonical column order only with an explicit, uncontradicted 0/1 direction."""
    if len(names) != 2:
        return None
    aliases = []
    for i, name in enumerate(names):
        words = re.split(r'[_\W]+', name.lower())
        sides = {'a','b','left','right'} & set(words)
        aliases.append({name.lower(), ('first','second')[i], *sides})
    for i,side in enumerate(('left','right')):
        if not any(side in group for group in aliases):
            aliases[i].add(side)
    aliases = [sorted(a - (aliases[1-i] & a), key=lambda s: (-len(s), s)) for i, a in enumerate(aliases)]
    votes = set()
    for sentence in re.split(r'[.!?\n]+', text.lower()):
        if not re.search(r'\b(?:win(?:s|ner)?|won|prefer(?:red|ence)?|higher|better|chosen|selected|outperform\w*)\b', sentence):
            continue
        if re.search(r'\b(?:not|never|hypothetical|example|unless|could|might|ignore)\b', sentence):
            continue
        if not re.search(r'\b'+re.escape(target.lower())+r'\b|\blabel\b|\btarget\b|\b(?:0|1)\s+(?:if|means|when|indicates)\b', sentence):
            continue
        for i, group in enumerate(aliases):
            side = r'\b(?:'+'|'.join(map(re.escape,group))+r')\b'
            for match in re.finditer(r'\b([01])\b\s*(?:=|:|if|when|means|denotes|indicates|for)\s*(?:that\s+)?(?:the\s+)?'+side, sentence):
                votes.add(i if match[1]=='1' else 1-i)
    if len(votes) != 1:
        return None
    first = votes.pop()
    return [names[first], names[1-first]]


def record_schema(header, rows, target, prose):
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError('Empty or malformed training table')
    columns = [{'name': name, 'type': _column_type([row[i] for row in rows])}
               for i, name in enumerate(header) if name != target]
    if not columns:
        raise ValueError('Training table has no input columns')
    text = all(col['type'] == 'text' for col in columns)
    result = dict(modality='text' if text else 'tabular', columns=columns, target=target)
    if text and len(columns) == 2 and {r[header.index(target)] for r in rows} == {'0', '1'}:
        order = declared_preference(prose, target, [col['name'] for col in columns])
        if order:
            result.update(modality='pair_text', pair_semantics='preference', pair_label_one='left',
                          columns=[dict(name=n, type='text') for n in order])
    return result


def infer_manifest(archive, *, read_declarations=False):
    from .folder_images import infer_folder_manifest
    folder=infer_folder_manifest(archive)
    if folder is not None:
        metric=declared_metric(description(archive))
        if metric:
            folder.update(metric=metric,_metric_source='declared_task_readme')
        return folder
    # This exact legacy categorical array convention keeps its existing adapter.
    if find_member(archive,['C_train.npy']) is not None:return None
    prose = description(archive) if read_declarations else ''
    resolved=getattr(archive,'shared_table',None)
    named_train=None if resolved else named_table(prose,'train')
    named_test=None if resolved else named_table(prose,'test')
    if named_train and named_train==named_test:
        raise ValueError('One file cannot serve as both training and prediction input')
    choices={fmt:find_member(archive,[name]) for fmt,name in
             [*((fmt,'train.'+fmt) for fmt in TABLE_FORMATS),('npz','train.npz'),('npy','X_train.npy')]}
    choices={fmt:path for fmt,path in choices.items() if path is not None}
    if named_train:
        choices={PurePosixPath(named_train).suffix[1:].lower():declared_member(archive,named_train)}
    if not choices:return None
    if len(choices)!=1:raise ValueError('Multiple possible training layouts; supply an explicit manifest')
    fmt,path=next(iter(choices.items()))
    manifest={'schema_version':1,'objective':'classification','metric':'accuracy',
              '_manifest_parent':'.','_schema_origin':'native_v1','_native_format':fmt,
              'splits':{'train':{'format':fmt,'path':path} if fmt in TABLE_FORMATS else descriptor(archive,fmt,'train')}}
    if named_test:
        manifest['_prediction_member']=named_test
    if evaluation_path(path,'train'):
        raise ValueError('Native training file cannot reside in a test path')
    if fmt in TABLE_FORMATS:
        header, reader=table(archive,path,fmt)
        if not header or len(set(header))!=len(header):raise ValueError('CSV needs unique headers')
        if not read_declarations and {'headline_A','headline_B','label_pairwise'}<=set(header):return None
        target=target_column(prose,header)
        manifest.update(record_schema(header, list(reader), target, prose))
    else:
        if fmt=='npz':
            with np.load(io.BytesIO(archive.read(path)),allow_pickle=False) as arrays:
                if not {'X','y'}<=set(arrays.files):raise ValueError('Native train.npz requires X and y arrays')
                x=np.asarray(arrays['X'])
        else:x=np.load(io.BytesIO(archive.read(path)),allow_pickle=False)
        if x.dtype.kind not in 'biuf' or x.ndim not in (2,3,4):
            raise ValueError('Native arrays require numeric N,D or N,H,W[,C]; otherwise declare a supported manifest')
        manifest['modality']='tabular' if x.ndim==2 else 'image'
    metric=declared_metric(prose if read_declarations else description(archive))
    if metric:
        manifest.update(metric=metric,_metric_source='declared_task_readme')
    return manifest


def add_native_split(archive,manifest,split):
    if manifest.get('_schema_origin')=='native_v1' and split not in manifest['splits']:
        aliases={'validation':['validation','val','valid'],'test':['test']}[split]
        fmt=manifest['_native_format']
        existing=find_member(archive,[s+'.'+fmt for s in aliases] if fmt in (*TABLE_FORMATS,'npz') else ['X_'+s+'.npy' for s in aliases])
        if existing is not None:manifest['splits'][split]=descriptor(archive,fmt,split)
    return manifest


def prediction_manifest(archive,frozen_spec):
    """No re-inference and no training/target reads at prediction time."""
    inputs=frozen_spec['input_schema']
    fmt=frozen_spec['input_format']
    if fmt=='image_folder':
        from .folder_images import prediction_folder_manifest
        return prediction_folder_manifest(archive,frozen_spec)
    if frozen_spec.get('prediction_member'):
        member=declared_member(archive,frozen_spec['prediction_member'])
        return {'schema_version':1,**inputs,'_manifest_parent':'.',
                'splits':{'test':{'format':PurePosixPath(member).suffix[1:].lower(),'path':member}}}
    return {'schema_version':1,'metric':'accuracy',**inputs,'_manifest_parent':'.',
            'splits':{'test':descriptor(archive,fmt,'test')}}
