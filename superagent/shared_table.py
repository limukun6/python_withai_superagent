"""Expose only the declared rows of a table that physically contains multiple evaluation splits."""
from contextlib import contextmanager
import csv
import io
import json
from pathlib import PurePosixPath
import re
import zipfile

from .native_schema import declared_member, description, find_member
from .schema import read_manifest
from .table_input import heading, table, target_column
from .task_files import TABLE_FORMATS, evaluation_path


ROLES={'train':{'train','training'},'validation':{'val','valid','validation','dev'},
       'test':{'test','testing','prediction','predict','inference'}}


def grouping_column(header,prose,explicit=None):
    declarations=[n for n in header if re.search(r'(?:\b(?:split|partition)\s+(?:column|field)\s*(?:is|[:=])?\s*|(?:划分列|分区列)\s*(?:为|是|[:：])?\s*)'
                                        +re.escape(n)+r'(?!\w)',prose,re.I)]
    if explicit is not None:
        declarations.append(explicit)
    if len(set(declarations))>1 or declarations and declarations[0] not in header:
        raise ValueError('Missing or conflicting declared split column')
    if declarations:
        return declarations[0]
    ordinary=[n for n in header if n.casefold() in ('split','partition','subset')]
    if len(ordinary)>1:
        raise ValueError('Several possible split columns')
    return next(iter(ordinary),None)


def shared_source(archive,split,manifest,contract):
    if contract:
        matches=[n for n in archive.namelist() if n==contract['member'] or n.endswith('/'+contract['member'])]
        single=getattr(archive,'standalone',None)
        if not matches and single and PurePosixPath(contract['member']).name==single[0]:
            matches=[single[1]]
        if len(matches)>1:
            raise ValueError('Shared table is ambiguous')
        return next(iter(matches),None),bool(matches)
    if manifest:
        descriptor=manifest.get('splits',{}).get(split,{})
        member=descriptor.get('path')
        if not isinstance(member,str) or descriptor.get('format') not in TABLE_FORMATS:
            return None,False
        shared=sum(d.get('path')==member for d in manifest['splits'].values() if isinstance(d,dict))>1
        if shared or manifest.get('split_column'):
            full=(PurePosixPath(manifest['_manifest_parent'])/member).as_posix()
            return declared_member(archive,full),True
        return None,False
    if split!='train':
        return None,False
    choices=[n for n in archive.namelist() if PurePosixPath(n).suffix[1:].lower() in TABLE_FORMATS
             and '__MACOSX' not in PurePosixPath(n).parts and not evaluation_path(n,'train')]
    named = getattr(archive,'documented_roles',{})
    if 'train' in named:
        member=declared_member(archive,named['train'])
        if PurePosixPath(member).suffix[1:].lower() in TABLE_FORMATS:
            return member,list(named.values()).count(named['train'])>1
    primary=[n for n in choices if PurePosixPath(n).stem.lower()=='train']
    if len(primary)==1:
        return primary[0],False
    return (choices[0],False) if len(choices)==1 else (None,False)


@contextmanager
def stage(archive,split,contract=None,*,for_partial_labels=False):
    manifest=read_manifest(archive,for_partial_labels=for_partial_labels)
    member,required=shared_source(archive,split,manifest,contract)
    if not member:
        yield archive
        return
    if split=='train' and evaluation_path(member,split):
        raise ValueError('A training source names an evaluation split')
    path=PurePosixPath(member)
    if path.is_absolute() or '..' in path.parts or chr(92) in member:
        raise ValueError('Unsafe shared table path')
    fmt=path.suffix[1:].lower()
    header=heading(archive,member,fmt)
    prose=description(archive)
    grouping=grouping_column(header,prose,(contract or manifest or {}).get('split_column'))
    if not grouping:
        if required or re.search(r'(?:split|partition)\s+(?:column|field)\s*(?:is|[:=])|(?:划分列|分组列)[：为是]',prose,re.I):
            raise ValueError('A table reused for train and evaluation needs a declared split column')
        yield archive
        return
    labels=(contract or manifest or {}).get('split_values') or {s:sorted(v) for s,v in ROLES.items()}
    if not isinstance(labels,dict) or not {'train','test'}<=set(labels) or set(labels)-set(ROLES):
        raise ValueError('Invalid split values')
    buckets={}
    for role,values in labels.items():
        values=[values] if isinstance(values,str) else values
        if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values):
            raise ValueError('Invalid split value')
        for value in values:
            token=value.strip().lower()
            if token in buckets:
                raise ValueError('A row marker maps to two splits')
            buckets[token]=role
    _,markers=table(archive,member,fmt,[grouping])
    markers=[r[0].strip().lower() for r in markers]
    seen=set(markers)
    if not required and not seen.intersection(buckets):
        raise ValueError('Split values were not recognized; row membership cannot be established')
    if not seen<=buckets.keys() or split not in {buckets[v] for v in seen}:
        raise ValueError('Missing or ambiguous row membership for '+split)
    target=(contract or manifest or {}).get('target') or target_column(prose,header)
    columns=[n for n in header if n!=grouping and (n!=target or split=='train')]
    if not columns or grouping==target:
        raise ValueError('Split and target columns must be distinct')
    _,rows=table(archive,member,fmt,columns,row_mask=[buckets[m]==split for m in markers])
    buf=io.StringIO(newline='')
    writer=csv.writer(buf)
    writer.writerow(columns)
    writer.writerows(rows)
    raw=io.BytesIO()
    name=split+'.csv'
    with zipfile.ZipFile(raw,'w') as out:
        out.writestr(name,buf.getvalue())
        if manifest:
            value={k:v for k,v in manifest.items() if not k.startswith('_') and k not in ('splits','split_column','split_values')}
            if 'columns' in value:
                value['columns']=[c for c in value['columns'] if c['name']!=grouping]
            value['splits']={split:{'format':'csv','path':name}}
            out.writestr('manifest.json',json.dumps(value))
        if split=='train':
            from .format_sources import task_documents
            docs=task_documents(archive.namelist())
            if docs:
                # Preserve the source bytes, including ordering punctuation, alongside the selected rows.
                out.writestr(PurePosixPath(docs[0]).name,archive.read(docs[0]))
    raw.seek(0)
    with zipfile.ZipFile(raw) as view:
        original=getattr(archive,'standalone',None)
        view.role_names=getattr(archive,'role_names',archive.namelist())
        view.documented_roles=getattr(archive,'documented_roles',{})
        view.observed_roles={buckets[value] for value in seen}
        view.shared_table=dict(member=original[0] if original else member,split_column=grouping,
                               target=target,split_values=labels)
        yield view
