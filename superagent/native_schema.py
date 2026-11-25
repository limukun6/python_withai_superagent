"""Conservative schema discovery from standard training-file layouts only.

The task/dataset name is never used. Ambiguous layouts or label semantics need
an explicit manifest. All inferred feature decisions use training contents.
"""
import csv
import io
from pathlib import PurePosixPath
import re
import statistics

import numpy as np

from .schema import MISSING

TARGET_NAMES={'target','label','y','class'}


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
    if fmt in ('csv','npz'):
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
    member = find_member(archive, ['readme.md', 'readme.txt'])
    if member is None or archive.getinfo(member).file_size > 256 * 1024:
        return ''
    return archive.read(member).decode('utf-8-sig', errors='replace').replace(chr(96), '').replace('*', '')


def declared_preference(text, target, names):
    """Return canonical column order only with an explicit, uncontradicted 0/1 direction."""
    if len(names) != 2:
        return None
    aliases = []
    for i, name in enumerate(names):
        words = re.split(r'[_\W]+', name.lower())
        sides = ({'a', 'first', 'left'}, {'b', 'second', 'right'})[i]
        # Letter aliases require a matching header. Positional words denote CSV order.
        aliases.append({name.lower(), *({'first','left'} if i == 0 else {'second','right'}), *(sides & set(words))})
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


def infer_manifest(archive, *, read_declarations=False):
    from .folder_images import infer_folder_manifest
    folder=infer_folder_manifest(archive)
    if folder is not None:return folder
    # This exact legacy categorical array convention keeps its existing adapter.
    if find_member(archive,['C_train.npy']) is not None:return None
    choices={fmt:find_member(archive,[name]) for fmt,name in
             [('csv','train.csv'),('npz','train.npz'),('npy','X_train.npy')]}
    choices={fmt:path for fmt,path in choices.items() if path is not None}
    if not choices:return None
    if len(choices)!=1:raise ValueError('Multiple possible training layouts; supply an explicit manifest')
    fmt,path=next(iter(choices.items()))
    manifest={'schema_version':1,'objective':'classification','metric':'accuracy',
              '_manifest_parent':'.','_schema_origin':'native_v1','_native_format':fmt,
              'splits':{'train':descriptor(archive,fmt,'train')}}
    if any('test' in part.casefold() for part in PurePosixPath(path).parts):
        raise ValueError('Native training file cannot reside in a test path')
    if fmt=='csv':
        reader=csv.reader(io.StringIO(archive.read(path).decode('utf-8-sig')))
        header=next(reader,None)
        if not header or len(set(header))!=len(header):raise ValueError('CSV needs unique headers')
        if not read_declarations and {'headline_A','headline_B','label_pairwise'}<=set(header):return None
        targets=[name for name in header if name.strip().casefold() in TARGET_NAMES]
        prose = description(archive) if read_declarations else ''
        if read_declarations:
            declared = [name for name in header if re.search(
                r'\b(?:target|label)(?:\s+column)?(?:\s+is|\s*[:=])?\s+'+re.escape(name)+r'(?!\w)', prose,re.I)]
            if declared:
                targets = list(dict.fromkeys([*declared, *targets]))
        if len(targets)!=1:raise ValueError('Target column is absent or ambiguous; declare target in manifest.json')
        rows=list(reader)
        if not rows or any(len(row)!=len(header) for row in rows):raise ValueError('Empty or malformed training CSV')
        target=targets[0];columns=[]
        for i,name in enumerate(header):
            if name==target:continue
            columns.append({'name':name,'type':_column_type([row[i] for row in rows])})
        if not columns:raise ValueError('Training CSV has no input columns')
        text_columns=[c for c in columns if c['type']=='text']
        manifest.update(modality='text' if len(text_columns)==len(columns) else 'tabular',columns=columns,target=target)
        if len(columns)==len(text_columns)==2 and {row[header.index(target)] for row in rows}=={'0','1'}:
            order = declared_preference(prose, target, [col['name'] for col in columns])
            if order:
                manifest.update(modality='pair_text', pair_semantics='preference', pair_label_one='left',
                                columns=[dict(name=n,type='text') for n in order])
        if read_declarations:
            from .metrics import SUPPORTED
            named = {m for m in SUPPORTED if re.search(r'\b(?:report|primary metric\s*(?:is|:)?)\s+(?:test\s+)?'
                        + re.escape(m).replace('_',r'[_ -]') + r'\b',prose,re.I)}
            if len(named)>1:
                raise ValueError('Conflicting declared primary metrics')
            if named:
                manifest['metric']=named.pop()
    else:
        if fmt=='npz':
            with np.load(io.BytesIO(archive.read(path)),allow_pickle=False) as arrays:
                if not {'X','y'}<=set(arrays.files):raise ValueError('Native train.npz requires X and y arrays')
                x=np.asarray(arrays['X'])
        else:x=np.load(io.BytesIO(archive.read(path)),allow_pickle=False)
        if x.dtype.kind not in 'biuf' or x.ndim not in (2,3,4):
            raise ValueError('Native arrays require numeric N,D or N,H,W[,C]; otherwise declare a supported manifest')
        manifest['modality']='tabular' if x.ndim==2 else 'image'
    return manifest


def add_native_split(archive,manifest,split):
    if manifest.get('_schema_origin')=='native_v1' and split not in manifest['splits']:
        aliases={'validation':['validation','val','valid'],'test':['test']}[split]
        fmt=manifest['_native_format']
        existing=find_member(archive,[s+'.'+fmt for s in aliases] if fmt in ('csv','npz') else ['X_'+s+'.npy' for s in aliases])
        if existing is not None:manifest['splits'][split]=descriptor(archive,fmt,split)
    return manifest


def prediction_manifest(archive,frozen_spec):
    """No re-inference and no training/target reads at prediction time."""
    inputs=frozen_spec['input_schema']
    fmt=frozen_spec['input_format']
    if fmt=='image_folder':
        from .folder_images import prediction_folder_manifest
        return prediction_folder_manifest(archive,frozen_spec)
    return {'schema_version':1,'metric':'accuracy',**inputs,'_manifest_parent':'.',
            'splits':{'test':descriptor(archive,fmt,'test')}}
