"""Bind documented facts to actual training columns, labels and package metadata."""
import hashlib
import re
from pathlib import PurePosixPath

import numpy as np

from .format_sources import decode_text, task_documents
from .task_constraints import declared_restrictions


class DeclarationError(ValueError):
    """An explicit declaration disagrees with the protected training schema."""


def read_document(archive):
    names = task_documents(archive.namelist())
    if not names:
        return '', None
    if len(names) != 1 or archive.getinfo(names[0]).file_size > 256 * 1024:
        raise ValueError('Task declaration source is ambiguous or exceeds its reading limit')
    raw = archive.read(names[0])
    return decode_text(raw), hashlib.sha256(raw).hexdigest()


FIELD = r'(?:\x60([^\x60\r\n]{1,80})\x60|"([^"\r\n]{1,80})"|([\w.-]{1,80}))'
ORDERED = re.compile(r'(?:\b(?:columns?|fields?|features?)\b.{0,160}?(?:from left to right|in (?:this |the following )?order|in sequence|ordered)|\bordered\s+(?:columns?|fields?|features?)\b|(?:列名|字段|特征).{0,40}(?:从左到右|依次|顺序)|(?:从左到右|有序)(?:的)?(?:列名|字段|特征))\s*[,，:]?\s*(?:are|is|as follows|为|是|为下列)?\s*[:：]?\s*', re.I)
COLUMN_HEADING = re.compile(r'^\s*(?:#+\s*)?(?:(?:input |feature )?column(?: name)?s|feature names|列名|输入字段|特征名称)\s*[:：]\s*',re.I)
LIST_LINE = re.compile(r'\b(?:columns?|features?|fields?|layout|coordinates|dimensions)\b|列名|列顺序|特征|字段',re.I)
LABEL_LINE = re.compile(r'\b(?:class(?:es)?|labels?|outcomes?|codes?|predicts?|target)\b|类别|标签|代表|预测',re.I)
NON_FACT = re.compile(r'\b(?:example|hypothetical|might|could|ignore|optionally|instead of)\b|例如|假设|忽略|可选',re.I)


def sentences(text):
    return [s.strip().rstrip('.;。；') for s in re.split(r'[\n]+|(?<=[。！？；])|(?<=[.!?;])(?:\s+|$)', text) if s.strip()]


def explicit_names(fragment):
    """The full list must consist of literal field names and separators (no omitted middle fields)."""
    items = re.split(r'\s*(?:[,，、]\s*(?:and\s+)?|\s+and\s+)\s*', fragment.strip().strip(' .:：'))
    names = []
    for item in items:
        match = re.fullmatch(FIELD, item.strip())
        if not match:
            return None
        names.append(next(v for v in match.groups() if v is not None))
    return names


def valid_names(names, width, target):
    return (isinstance(names,list) and 0 < len(names) == width and all(
        isinstance(v,str) and 0 < len(v) <= 80 and not any(ord(c)<32 for c in v) for v in names)
        and len({v.casefold() for v in names}) == width
        and (target is None or str(target).casefold() not in {v.casefold() for v in names}))


def label_definitions(line, domain):
    token = r'(?<![\w.-])(?:' + '|'.join(re.escape(str(v)) for v in sorted(domain,key=lambda v:-len(str(v)))) + r')(?![\w.-])'
    pair = re.compile(r'(?P<label>' + token + r')\s*(?:=|:|：|\bmeans\b|\bdenotes\b|\bstands for\b|表示|代表)\s*'
                      r'(?P<meaning>[^\W\d_][\w -]{0,64}?)(?=\s*(?:[,，、)/]|$|\band\s+' + token + r'))', re.I)
    mapping = {}
    for match in pair.finditer(line.replace(chr(96),'').replace('"','').replace("'",'')):
        key, meaning = match['label'], match['meaning'].strip()
        label = next((v for v in domain if str(v) == key), None)
        if label is None or not meaning:
            continue
        if key in mapping and mapping[key] != meaning:
            raise DeclarationError('Conflicting descriptions for a training label')
        mapping[key] = meaning
    return mapping


def resolve_positive(text, domain, meanings, metric):
    from .metric_declarations import announced_clause, named_recall_labels
    values = set()
    for line in sentences(text):
        if NON_FACT.search(line) or line.endswith(('?','？')):
            continue
        matches = re.findall(r'(?:\bpositive (?:class|label)(?:\s+(?:is|equals)|\s*[:=])\s*|正类(?:标签)?\s*(?:为|是|[:：=])\s*)'
                             r'[\x60"\x27]?([\w -]+?)[\x60"\x27]?(?=[,，;.)]|$)',line,re.I)
        if metric=='binary_recall':
            matches.extend(value for _,value in named_recall_labels(announced_clause(line)))
        for value in matches:
            token = value.strip().casefold()
            options = [v for v in domain if token == str(v).casefold() or token == meanings.get(str(v),'').casefold()]
            if len(domain)!=2 or len(options)!=1:
                raise DeclarationError('Declared positive class does not identify one training label')
            values.add(options[0])
    if len(values)>1:
        raise DeclarationError('Conflicting declared positive classes')
    return next(iter(values),None)


def bind(task, archive, *, client=None, log=None, manifest=None):
    """Only verified schema and restrictive flags can influence execution. No method is selected here."""
    from .contract_extraction import proposed_columns
    spec = task.spec
    text, document_hash = read_document(archive)
    domain = np.unique(task.y_train).tolist()
    schema = spec.get('input_schema',{})
    target = schema.get('target')
    shape = getattr(task.X_train,'shape',())
    width = shape[1] if len(shape)==2 and task.kind.endswith('tabular') else None
    existing = schema.get('columns',[])
    existing_names = [c['name'] for c in existing]
    evidence = {}
    declared = (manifest or {}).get('restrictions')
    restrictions,prose_flags=declared_restrictions(declared,text)
    if restrictions:
        spec['method_restrictions'] = restrictions
        evidence['method_restrictions'] = dict(value=restrictions,status='confirmed',
            source='declared_manifest_and_readme' if declared and prose_flags else 'declared_manifest' if declared else 'declared_task_readme',
            witnesses=prose_flags)

    names = None
    name_source = 'declared_task_readme'
    lines = [line for line in sentences(text) if not NON_FACT.search(line) and not line.endswith(('?','？'))]
    if width is not None:
        if existing and not valid_names(existing_names,width,target):
            raise DeclarationError('Declared column names conflict with the training feature width or target')
        for line in lines:
            declaration = ORDERED.search(line) or COLUMN_HEADING.match(line)
            if declaration:
                proposed = explicit_names(line[declaration.end():])
                if not valid_names(proposed,width,target) or names is not None and proposed != names:
                    raise DeclarationError('Documented ordered fields disagree with the training schema')
                names = proposed
        if names is None and not existing and text and log is not None:
            names = proposed_columns(lines, width, target, domain, document_hash, client, log)
            name_source = 'llm_extracted'
        if names is not None:
            if existing and existing_names != names:
                raise DeclarationError('Documented column order conflicts with the declared input schema')
            if not existing:
                kind = {'numeric_tabular':'numeric','categorical_tabular':'categorical'}.get(task.kind)
                if kind:
                    spec['input_schema'] = dict(schema,modality='tabular',columns=[dict(name=n,type=kind) for n in names])
                    spec['feature_shape'] = [width]
            evidence['columns'] = dict(value=names, source=name_source,
                                       status='proposed' if name_source=='llm_extracted' else 'confirmed')

    meanings = {}
    dictionary = False
    for line in lines:
        if re.fullmatch(r'\s*(?:#+\s*)?(?:label(?:s| meanings| dictionary| codes)?|class codes|标签(?:含义|字典)?|类别编码)\s*[:：]?\s*',line,re.I):
            dictionary=True
            continue
        entry = re.match(r'^\s*(?:[-*]\s*)?[\x60"\x27]?[\w-]+[\x60"\x27]?\s*(?:[=:：]|means\b|denotes\b|表示)',line,re.I)
        if not entry:
            dictionary=False
        if dictionary or re.match(r'^\s*(?:labels?|class codes|标签(?:含义|字典)?)\s*[:：]',line,re.I):
            numbers=re.findall(r'(?<![\w.-])([+-]?\d+(?:\.\d+)?)\s*(?:[=:：]|means\b|denotes\b|表示)',line.replace(chr(96),''),re.I)
            if set(numbers)-{str(v) for v in domain}:
                raise DeclarationError('Documented label code is outside the training domain')
        found = label_definitions(line,domain)
        if not found or (not dictionary and not LABEL_LINE.search(line) and set(found) != {str(v) for v in domain}):
            continue
        if any(k in meanings and meanings[k]!=v for k,v in found.items()):
            raise DeclarationError('Contradictory documented label meanings')
        meanings.update(found)
    manifest_meanings = (manifest or {}).get('label_meanings',{})
    if (not isinstance(manifest_meanings,dict) or any(k not in {str(v) for v in domain}
            or not isinstance(v,str) or not v.strip() or len(v)>120 for k,v in manifest_meanings.items())
            or any(k in meanings and meanings[k]!=v for k,v in manifest_meanings.items())):
        raise DeclarationError('Declared label dictionary conflicts with the training domain or documentation')
    meanings.update(manifest_meanings)
    if meanings:
        evidence['label_meanings'] = dict(value=meanings,source='declared_manifest' if manifest_meanings else 'declared_task_readme',status='confirmed')
    positive = resolve_positive(text,domain,meanings,spec.get('metric'))
    if positive is not None:
        if spec.get('positive_label', positive)!=positive:
            raise DeclarationError('Conflicting positive class in manifest and task documentation')
        if 'positive_label' not in spec:
            spec['positive_label'] = positive
            spec.setdefault('goal_sources',{})['positive_label'] = 'declared_task_readme'
    if 'columns' not in evidence:
        evidence['columns']=dict(value=existing_names or None,source=spec.get('schema_origin','input_schema') if existing_names else 'undeclared',
                                 status='confirmed' if existing_names else 'unknown')
    evidence.setdefault('label_meanings',dict(value=None,source='undeclared',status='unknown'))
    evidence.setdefault('method_restrictions',dict(value=None,source='undeclared',status='unknown'))
    spec['declaration_evidence'] = dict(document_sha256=document_hash,fields=evidence)
    from .split_declarations import declared_paths, inventory
    role_names=getattr(archive,'role_names',archive.namelist())
    if spec.get('shared_table'):
        shared=spec['shared_table']
        # The training view has a synthetic train.csv. Preserve the original row-based declaration.
        split_roles={k:dict(members=[shared['member']],source='verified_split_column',
                            status='confirmed' if k in getattr(archive,'observed_roles',()) else 'unknown',
                            split_column=shared['split_column'],values=v) for k,v in shared['split_values'].items()}
    else:
        paths=getattr(archive,'documented_roles',None) or declared_paths(text)
        split_roles=inventory(role_names,paths,manifest)
    if split_roles:
        spec['split_roles']=split_roles
    return task
