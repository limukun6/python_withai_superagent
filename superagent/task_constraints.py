"""Task-owned restrictions on downstream methods, shared by menus, workers and final selection."""
import re


class RestrictionError(ValueError):
    """A binding method restriction could not be enforced."""


SUBJECTS = {
    'no_pretrained_assets': r'(?:\bpre[\s-]?trained\b(?:\s+(?:models?|weights?|assets?|encoders?|embeddings?|networks?|checkpoints?))?|预训练(?:权重|模型|编码器|资产)?)',
    'no_external_data': r'(?:\b(?:external|outside|third[- ]party|additional external)\s+(?:(?:training|labelled|labeled)\s+)?(?:datasets?|data|corpora|resources?)\b|外部(?:训练)?(?:数据集?|语料|资源))',
    'linear_only': r'(?:\blinear\s+(?:models?|classifiers?|estimators?|methods?|algorithms?)\b|线性(?:模型|分类器|方法|算法))',
}
FLAGS = frozenset(SUBJECTS)
DENY = r'(?:\b(?:no|without|never use|(?:must|may|can|should)\s+not\s+(?:be\s+)?(?:use[d]?|include)|do not use|cannot use|avoid|forbid(?:s|den)?|prohibit(?:s|ed)?|disallow(?:s|ed)?|ban(?:s|ned)?)\b|禁止|不得|不允许|不要使用|不使用|不能使用|禁用)'
ONLY = r'(?:\bonly\b|\brestrict(?:ed)?\s+to\b|\blimited\s+to\b|\bexclusively\b|只能|只(?:可|准|允许|能用|使用)|仅(?:可|准|允许|能用|使用|限))'
BARRED = r'(?:are|is|shall be|must be)?\s*(?:strictly\s+)?(?:forbidden|prohibited|disallowed|banned|not (?:permitted|allowed))\b|(?:不得使用|不可用|不允许|被禁止)'
MODEL_WORD = re.compile(r'\b(?:classifiers?|encoders?|models?|estimators?|algorithms?|networks?|trees?|forests?|kernels?|svms?|boosting|regression|ensembles?|neural|knn|xgboost|catboost|lightgbm|training data|external data|features?)\b|模型|算法|分类器|神经网络|预训练|外部数据',re.I)
NON_BINDING = re.compile(r'\b(?:for example|hypothetical|if|unless|might|perhaps|not required|no restriction on|not (?:forbidden|prohibited|disallowed|restricted|limited)|not only|ignore (?:all|previous|these))\b|例如|假设|无需|并非只|没有限制|忽略.{0,6}指令',re.I)


def manifest_flags(value):
    if value is None:
        return []
    if isinstance(value, list) and all(isinstance(k, str) and k in FLAGS for k in value):
        return sorted(set(value))
    if isinstance(value, dict) and set(value) <= FLAGS and all(type(v) is bool for v in value.values()):
        return sorted(k for k, enabled in value.items() if enabled)
    raise RestrictionError('Unrecognized mandatory manifest method restriction')


def documented_flags(text):
    """Only binding clauses count. Unsupported mandatory rules stop the run before a fit."""
    found = {}
    for part in re.finditer(r'([^.!?。！？;\n；]+)([.!?。！？;\n；]|$)',text):
        raw=part[1]
        if part[2] in ('?','？') or re.match(r'\s*(?:for (?:example|instance)|hypothetical|例如|比如|假设)',raw,re.I):
            continue
        for clause in re.split(r'[,，]\s*(?=(?:and\s+)?(?:only|no |do not|must not|只|禁止|不得))|\s+(?:and|but)\s+(?=(?:only|do not|must not|never|no )\b)', raw, flags=re.I):
            line = clause.strip(' *' + chr(96) + '\t\r')
            if (re.search(r'\b(?:if|unless)\b|如果|除非',line,re.I) and MODEL_WORD.search(line)
                    and re.search(DENY+'|'+ONLY+'|'+BARRED,line,re.I)):
                raise RestrictionError('Conditional method restriction requires a supported condition: '+line[:200])
            if not line or NON_BINDING.search(line):
                continue
            flags = []
            for flag, subject in SUBJECTS.items():
                if not re.search(subject, line, re.I):
                    continue
                if flag == 'linear_only':
                    pattern = ONLY + r'.{0,28}?' + subject + '|' + subject + r'\s+(?:are\s+|are\s+the\s+)?(?:the\s+)?only\b|' + subject + r'\s+only\b'
                else:
                    pattern = DENY + r'.{0,65}?' + subject + '|' + subject + r'\s*(?:may\s+)?(?:not\s+be\s+used|' + BARRED + ')'
                if re.search(pattern, line, re.I):
                    flags.append(flag)
            if re.search(r'\b(?:(?:use|using)\s+only|only\s+use)\s+(?:the\s+)?(?:(?:provided|supplied|task)\s+(?:training\s+)?data|training data)\b|仅(?:可)?使用(?:所给|提供的|给定的)(?:训练)?数据', line,re.I):
                flags.append('no_external_data')
            bounded = re.search(DENY + '|' + ONLY + '|' + BARRED, line,re.I)
            subject = re.sub(r'^(?:the |a |your )?(?:classifier|model|agent|superagent)\s+','',line,flags=re.I)
            if bounded and MODEL_WORD.search(subject) and not flags:
                # File roles and metric instructions also use "only", without regulating methods.
                if not re.search(r'\bonly\s+(?:for|after|when|on|if)\b|仅(?:用于|在)',line,re.I):
                    raise RestrictionError('Unsupported mandatory method restriction: ' + line[:200])
            if flags:
                # Do not drop a second, unimplemented rule piggybacking on a supported one.
                rest = line
                for flag in flags:
                    rest = re.sub(SUBJECTS[flag], '', rest, flags=re.I)
                if re.search(r'\b(?:and|or|except|excluding)\b|以及|同时|除外',rest,re.I) and MODEL_WORD.search(rest):
                    raise RestrictionError('Partly understood mandatory method restriction: ' + line[:200])
                for flag in flags:
                    found.setdefault(flag, line[:400])
    return found


def declared_restrictions(manifest_value,text):
    declared=manifest_flags(manifest_value)
    quoted=documented_flags(text.replace(chr(96),'').replace('**',''))
    if isinstance(manifest_value,dict) and any(manifest_value.get(k) is False for k in quoted):
        raise RestrictionError('Manifest and task documentation disagree about a mandatory restriction')
    return sorted(set(declared).union(quoted)),quoted


def package_restrictions(package):
    """Used before a legacy run or a recovery path, where no training schema is available yet."""
    import json
    from pathlib import PurePosixPath
    from .format_sources import package_files, roles
    from .task_declarations import read_document
    with package_files(package) as files:
        names=[n for n in files.namelist() if PurePosixPath(n).name=='manifest.json'
               and not roles(n).intersection({'validation','test'}) and '__MACOSX' not in PurePosixPath(n).parts]
        value=None
        if len(names)==1:
            manifest=json.loads(files.read(names[0],256*1024))
            if isinstance(manifest,dict):
                value=manifest.get('restrictions')
        return declared_restrictions(value,read_document(files)[0])[0]


LINEAR_CLASSES = frozenset({'LogisticRegression','LinearSVC','RidgeClassifier','SGDClassifier','Perceptron',
                            'PassiveAggressiveClassifier','LinearDiscriminantAnalysis'})


def permitted(cap, restrictions):
    if not restrictions:
        return True
    params = cap['params']
    generated = params.get('runtime_composed') is True
    # An extension's name cannot attest to its training-data or weight provenance.
    core = cap.get('origin') in ('shared_baseline','b_port') or (generated and '=' not in params['class'])
    pretrained = bool(params.get('encoder') or params.get('weight_sha256') or
                      cap.get('resource') != 'cpu')
    if not core or (pretrained and set(restrictions) & {'no_external_data','no_pretrained_assets','linear_only'}):
        return False
    if 'linear_only' not in restrictions:
        return True
    if generated:
        return params['class'] in LINEAR_CLASSES and params['projection'] == 'native'
    # These baseline factories use ordinary typed/TF-IDF features and a single linear classifier.
    return (cap.get('origin') == 'shared_baseline' and cap.get('estimator') == 'logistic'
            and 'C' in params and set(params) <= {'C','ngram_range','representation','swap_augmentation'}
            and params.get('representation') in (None,'word','char','word_char'))
