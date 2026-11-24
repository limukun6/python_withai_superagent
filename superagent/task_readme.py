"""Conservative extraction of explicit legacy task declarations, never executable instructions.

Only a small documented declaration grammar is recognized. Unrecognized prose
stays unknown and does not authorize an algorithm or change a manifest schema.
"""
from copy import deepcopy
import hashlib
from pathlib import PurePosixPath
import re

from .metrics import SUPPORTED

METRIC_NAMES = {name.replace('_',' '): name for name in SUPPORTED}
PRIMARY = re.compile(r'^(?:the\s+)?primary\s+(?:evaluation\s+)?metric\s*(?:is\s+|[:=]\s*)(.+)$', re.I)
REPORT = re.compile(r'\breport\s+(?:test\s+)?(.+)$', re.I)
POSITIVE = re.compile(r'^(?:the\s+)?positive[ _](?:label|class)\s*(?:is\s+|[:=]\s*)(.+)$', re.I)
NEGATED = re.compile(r'\b(?:not|no|never|avoid|instead|ignore|unless|optional|might|could|hypothetical|example)\b', re.I)


def declared_metric(sentences):
    strong, requests = [], []
    for sentence in sentences:
        if sentence.endswith('?') or NEGATED.search(sentence):
            continue
        claim = PRIMARY.match(sentence)
        report = REPORT.search(sentence)
        if not (claim or report):
            continue
        name = (claim or report).group(1).casefold().strip(' .!')
        name = re.sub(r'[\s_–-]+', ' ', name).removesuffix(' score')
        metric = METRIC_NAMES.get(name)
        if not claim and metric is None and re.fullmatch(r'accuracy (?:on|for|over) .+', name):
            # Retain legacy "report accuracy on the held-out set" without absorbing a second metric.
            if not re.search(r'\b(?:f1|auc|precision|recall|loss|accuracy)\b', name.removeprefix('accuracy ')):
                metric = 'accuracy'
        item = (metric, sentence)
        if claim:
            strong.append(item)
        else:
            requests.append(item)
    matches = strong or requests
    values = {m[0] for m in matches}
    if len(values) == 1 and None not in values:
        return matches[0]
    if strong:
        raise ValueError('Unsupported or conflicting primary metric declaration; supply a supported metric in manifest.json')
    return None


def legacy_declarations(archive, kind, spec, labels=(), *, legacy_pair=True):
    result=deepcopy(spec)
    excluded={'__macosx','test','val','valid','validation'}
    names=[n for n in archive.namelist() if not excluded.intersection(p.casefold() for p in PurePosixPath(n).parts[:-1])
           and PurePosixPath(n).name.casefold() in ('readme.md','readme.txt')]
    if len(names)!=1:return result
    if archive.getinfo(names[0]).file_size>256*1024:return result
    payload=archive.read(names[0])
    try:raw=payload.decode('utf-8-sig')
    except UnicodeDecodeError:return result
    raw=raw.replace('`','').replace('*','')
    text=re.sub(r'\s+',' ',raw).strip()
    sentences=re.split(r'(?<=[.!?])\s+',text)
    fields=[s.strip().lstrip('#- ').strip() for s in re.split(r'\n|(?<=[.!?])\s+',raw) if s.strip()]
    recognised={}
    metric=declared_metric(fields)
    if metric:
        result['metric'], excerpt=metric
        result.setdefault('goal_sources',{})['primary_metric']='declared_task_readme'
        recognised['primary_metric']=dict(value=result['metric'],excerpt=excerpt[:500])
    if result.get('metric') in ('binary_f1','roc_auc') and 'positive_label' not in result:
        declarations=[(m.group(1).rstrip('.!').strip().strip('"\''), s) for s in fields
                      if not s.endswith('?') and (m := POSITIVE.match(s))]
        if declarations:
            values={name for name,_ in declarations}
            domain={str(value):value for value in set(labels.tolist() if hasattr(labels,'tolist') else labels)}
            if len(values)!=1 or not values.issubset(domain) or len(domain)!=2:
                raise ValueError('Positive class declaration does not uniquely identify one of two training labels')
            result['positive_label']=domain[declarations[0][0]]
            result.setdefault('goal_sources',{})['positive_label']='declared_task_readme'
            recognised['positive_label']=dict(value=result['positive_label'],excerpt=declarations[0][1][:500])
    pair_pattern=(r'\b(?:target\s+)?label_pairwise\s+is\s+1\s+if\s+A\s+had\s+the\s+higher\s+rate'
                  r'\s+and\s+0\s+if\s+B\s+did\b')
    pair_lines=[s for s in sentences if re.search(pair_pattern,s,re.I)]
    # This grammar is tied to the already-recognized column contract, not a dataset identity.
    if legacy_pair and kind=='pair_text' and pair_lines and not any(NEGATED.search(s) or s.endswith('?') for s in pair_lines):
        if not re.search(r'\b(?:label_pairwise\s+is\s+0\s+if\s+A|1\s+if\s+B)\b',text,re.I):
            result['input_schema']={'modality':'pair_text','pair_semantics':'preference','pair_label_one':'left',
                'columns':[{'name':name,'type':'text'} for name in ('headline_A','headline_B')],
                'target':'label_pairwise'}
            result['input_format']='csv'
            recognised['pair_semantics']=dict(value='preference',pair_label_one='left',excerpt=pair_lines[0][:500])
    result['task_description_source']=dict(member=names[0],sha256=hashlib.sha256(payload).hexdigest(),
        parser_version='explicit-declarations-v2',recognized=recognised,
        scope='A limited declaration grammar; unrecognized prose remains unknown, never executed')
    return result
