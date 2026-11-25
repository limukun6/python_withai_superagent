"""Conservative extraction of explicit legacy task declarations, never executable instructions.

Only a small documented declaration grammar is recognized. Unrecognized prose
stays unknown and does not authorize an algorithm or change a manifest schema.
"""
from copy import deepcopy
import hashlib
import re
from .format_sources import decode_text,task_documents


def legacy_declarations(archive, kind, spec):
    result=deepcopy(spec)
    names=task_documents(archive.namelist())
    if len(names)!=1:return result
    if archive.getinfo(names[0]).file_size>256*1024:
        raise ValueError('Task instruction file exceeds the reading limit')
    payload=archive.read(names[0])
    raw=decode_text(payload)
    text=re.sub(r'\s+',' ',raw.replace('`','').replace('*','')).strip()
    sentences=re.split(r'(?<=[.!?])\s+',text)
    negation=re.compile(r'\b(?:not|no|never|avoid|instead|ignore|unless|optional|might|could|hypothetical|example)\b',re.I)
    from .metric_declarations import declared_metric
    recognised={}
    metric=declared_metric(raw.replace(chr(96),'').replace('*',''))
    if metric:
        result['metric']=metric
        result.setdefault('goal_sources',{})['primary_metric']='declared_task_readme'
        recognised['primary_metric']=dict(value=metric)
    pair_pattern=(r'\b(?:target\s+)?label_pairwise\s+is\s+1\s+if\s+A\s+had\s+the\s+higher\s+rate'
                  r'\s+and\s+0\s+if\s+B\s+did\b')
    pair_lines=[s for s in sentences if re.search(pair_pattern,s,re.I)]
    # This grammar is tied to the already-recognized column contract, not a dataset identity.
    if kind=='pair_text' and pair_lines and not any(negation.search(s) or s.endswith('?') for s in pair_lines):
        if not re.search(r'\b(?:label_pairwise\s+is\s+0\s+if\s+A|1\s+if\s+B)\b',text,re.I):
            result['input_schema']={'modality':'pair_text','pair_semantics':'preference','pair_label_one':'left',
                'columns':[{'name':name,'type':'text'} for name in ('headline_A','headline_B')],
                'target':'label_pairwise'}
            result['input_format']='csv'
            recognised['pair_semantics']=dict(value='preference',pair_label_one='left',excerpt=pair_lines[0][:500])
    result['task_description_source']=dict(member=names[0],sha256=hashlib.sha256(payload).hexdigest(),
        parser_version='shared-metric-declarations-v2',recognized=recognised,
        scope='A limited declaration grammar; unrecognized prose remains unknown, never executed')
    return result
