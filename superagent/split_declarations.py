"""Validate file-role metadata without opening evaluation members."""
from pathlib import PurePosixPath
import re

from .format_sources import checked_name, roles


FILE = r'(?P<path>[\w./-]+\.(?:csv|tsv|parquet|jsonl?|ndjson|npz|npy|xlsx|db|sqlite|arff|xml|txt|arrow|feather))\b'
ROLE = {'train':r'(?:training|train|训练(?:集)?)','validation':r'(?:validation|development|held[- ]out validation|验证(?:集)?)',
        'test':r'(?:prediction|inference|test(?:ing)?|final evaluation|预测(?:集)?|测试(?:集)?)'}


def declared_paths(text):
    result={}
    for raw in re.split(r'[\n。；;！？]+|[.!?](?=\s|$)',text.replace(chr(96),'').replace('**','')):
        if re.search(r'\b(?:example|hypothetical|ignore|maybe|not|never)\b|例如|假设|忽略',raw,re.I):
            continue
        for role,term in ROLE.items():
            forward = term+r'\s*(?:(?:data\s*)?(?:file|table)|文件|数据文件)\s*(?:is\s+|为|是|[:：=]\s*)?["\x27]?'+FILE
            reverse = FILE+r'["\x27]?\s+(?:is (?:reserved |used |intended )?(?:only )?for|contains(?: only)?|holds)\s+(?:the\s+)?'+term+r'(?:\s+(?:data|rows|examples|samples|records|features))?\b'
            for pattern in (forward,reverse):
                for m in re.finditer(pattern,raw,re.I):
                    path=checked_name(m['path'])
                    if roles(path)-{role} or role in result and result[role]!=path:
                        raise ValueError('Conflicting documented '+role+' file declaration')
                    result[role]=path
    return result


def matches(names, path):
    return [n for n in names if n == path or n.endswith('/'+path)]


def inventory(names, declared, manifest=None):
    """Only paths that exist and agree with their declared role are reported as verified."""
    listed={}
    nameset=set(names)
    for role in ROLE:
        paths = []
        source = 'split_filename'
        info = (manifest or {}).get('splits',{}).get(role) or {}
        if info:
            parent = PurePosixPath(manifest.get('_manifest_parent','.'))
            paths = [str(parent/checked_name(p)) for p in (info.get('path'),info.get('target_path')) if isinstance(p,str)]
            source=('declared_task_readme' if role in declared else 'native_layout') if manifest.get('_schema_origin')=='native_v1' else 'declared_manifest'
        elif role in declared:
            paths=[declared[role]]
            source='declared_task_readme'
        else:
            paths=sorted(n for n in names if roles(n)=={role} and PurePosixPath(n).suffix.lower() in ('.csv','.npy','.npz','.tsv','.parquet','.jsonl','.ndjson'))
        if not paths:
            continue
        verified=[]
        for path in paths:
            found=[path] if path in nameset else matches(names,path)
            if len(found)>1 or (found and roles(found[0])-{role}):
                raise ValueError('Ambiguous or contradictory split member: '+path)
            if found:
                verified.append(found[0])
        listed[role]=dict(members=verified[:16],member_count=len(verified),source=source,
                          status='confirmed' if len(verified)==len(paths) else 'unknown')
    return listed
