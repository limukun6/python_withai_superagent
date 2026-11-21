"""Conservative extraction of explicit legacy task declarations, never executable instructions.

Only a small documented declaration grammar is recognized. Unrecognized prose
stays unknown and does not authorize an algorithm or change a manifest schema.
"""
from copy import deepcopy
import hashlib
from pathlib import PurePosixPath
import re


def legacy_declarations(archive, kind, spec):
    result=deepcopy(spec)
    names=[n for n in archive.namelist() if '__MACOSX' not in PurePosixPath(n).parts
           and PurePosixPath(n).name.casefold() in ('readme.md','readme.txt')]
    if len(names)!=1:return result
    payload=archive.read(names[0])
    if len(payload)>256*1024:return result
    try:raw=payload.decode('utf-8-sig')
    except UnicodeDecodeError:return result
    text=re.sub(r'\s+',' ',raw.replace('`','').replace('*','')).strip()
    sentences=re.split(r'(?<=[.!?])\s+',text)
    negation=re.compile(r'\b(?:not|no|never|avoid|instead|ignore|unless|optional|might|could|hypothetical|example)\b',re.I)
    metric_lines=[s for s in sentences if re.search(r'\breport\s+(?:test\s+)?accuracy\b',s,re.I)]
    recognised={}
    # Ambiguous/conflicting report requests in the same document do not establish one objective.
    other_metric=re.search(r'\breport\b[^.!?]*\b(?:f1|auc|precision|recall)\b',text,re.I)
    if metric_lines and not other_metric and not any(negation.search(s) or s.endswith('?') for s in metric_lines):
        result['metric']='accuracy'
        result.setdefault('goal_sources',{})['primary_metric']='declared_task_readme'
        recognised['primary_metric']=dict(value='accuracy',excerpt=metric_lines[0][:500])
    pair_pattern=(r'\b(?:target\s+)?label_pairwise\s+is\s+1\s+if\s+A\s+had\s+the\s+higher\s+rate'
                  r'\s+and\s+0\s+if\s+B\s+did\b')
    pair_lines=[s for s in sentences if re.search(pair_pattern,s,re.I)]
    # This grammar is tied to the already-recognized column contract, not a dataset identity.
    if kind=='pair_text' and pair_lines and not any(negation.search(s) or s.endswith('?') for s in pair_lines):
        if not re.search(r'\b(?:label_pairwise\s+is\s+0\s+if\s+A|1\s+if\s+B)\b',text,re.I):
            result['input_schema']={'modality':'pair_text','pair_semantics':'preference','pair_label_one':'left'}
            recognised['pair_semantics']=dict(value='preference',pair_label_one='left',excerpt=pair_lines[0][:500])
    result['task_description_source']=dict(member=names[0],sha256=hashlib.sha256(payload).hexdigest(),
        parser_version='legacy-explicit-declarations-v1',recognized=recognised,
        scope='A limited declaration grammar; unrecognized prose remains unknown, never executed')
    return result
