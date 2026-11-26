"""Resolve announced classification metrics without treating an unrecognized request as accuracy."""
import re
import unicodedata


class MetricDeclarationError(ValueError):
    pass


ALIASES = {
    'accuracy': ('accuracy','classification accuracy','accuracy score','准确率'),
    'balanced_accuracy': ('balanced accuracy','balanced accuracy score','balanced classification accuracy','平衡准确率'),
    'binary_recall': ('binary recall','binary recall score','二分类召回率'),
    'roc_auc': ('roc auc','roc auc score','auc roc','area under the roc curve','area under the receiver operating characteristic curve','auroc','auc'),
    'f1': ('f1 score','f1','f 1','f measure','f score','f1值'),
}
for average, translated in [('macro','宏平均'),('micro','微平均'),('weighted','加权'),('binary','二分类')]:
    ALIASES[average+'_f1'] = tuple(f'{prefix} {f1}' for prefix in
        (average,average+' averaged',average+' average') for f1 in ('f1','f1 score')) + (
        'f1 '+average,'f1 score '+average,translated+'f1',translated+' f1')


def normalized(text):
    return re.sub(r'\s+',' ',re.sub(r'[_‐‑‒–—−()/-]+',' ',unicodedata.normalize('NFKC',text).casefold())).strip()


LOOKUP = {alias:name for name,variants in ALIASES.items() for alias in variants}
UNSUPPORTED = ('pr auc','auc pr','aucpr','auprc','average precision','precision recall auc','log loss',
               'cross entropy','matthews correlation coefficient','mcc','cohen kappa','precision','recall',
               'top k accuracy','partial auc','weighted roc auc','micro roc auc','brier score','rmse','mae')
TERMS = sorted(set(LOOKUP) | set(UNSUPPORTED),key=len,reverse=True)
PATTERN = re.compile(r'(?<![a-z0-9])(?:'+'|'.join(map(re.escape,TERMS))+r')(?![a-z0-9])')
PRIMARY = re.compile(r'\b(?:primary|main|official|selection|ranking|evaluation|scoring)\s+(?:metric|criterion|measure)|主指标|主要评价指标|评估指标|评价指标')
DIRECTIVE = re.compile(r'\b(?:metrics?|scor(?:ed|ing) (?:by|on|using|with)|evaluat(?:ed|ion) (?:by|on|using|with)|'
                       r'(?:choose|select|rank|judge|judged|leaderboard|assess|graded|score|evaluate|optimise|optimize|maximi[sz]e|report|measured by))\b|评分|评价|评估|指标|报告|汇报|选模|以.+为准')
NEGATIVE = re.compile(r'\b(?:not|no requirement|never|optional|might|hypothetical|example|instead of)\b|不使用|不要求|例如|可选')
SECONDARY = re.compile(r'\b(?:secondary|diagnostic|for reference|for completeness|optional)\b|辅助指标|仅供参考')
BINDING = re.compile(r'\b(?:(?:ranked|graded|assessed|judged|scored|evaluated|measured)\s+(?:by|using|with|on|according to)|'
                     r'leaderboard (?:uses|is based on)|(?:evaluation|scoring|ranking) (?:uses|is based on)|'
                     r'report (?:the |test )?[^.!?]{0,50}\b(?:score|coefficient|loss|index|f[0-9]+)\b)|(?:按|使用|采用).{1,40}(?:评分|排名|选模)')


def metric_name(value):
    if not isinstance(value,str) or normalized(value) not in LOOKUP:
        raise MetricDeclarationError('Supported primary metrics do not include this declaration: '+str(value)[:100])
    return LOOKUP[normalized(value)]


# A bare "recall" still lacks a class/averaging rule. A named binary-class request
# carries an exact label reference which the intake subsequently binds to training labels.
CLASS_WORD = r'(\x60[^\x60\r\n]{1,64}\x60|"[^"\r\n]{1,64}"|\x27[^\x27\r\n]{1,64}\x27|[\w-]+)'
NAMED_RECALL = (
    re.compile(r'\brecall\s+(?:for|of)\s+(?:the\s+)?(?:class\s+|label\s+)?'+CLASS_WORD+r'(?:\s+class)?(?!\w)',re.I),
    re.compile(r'(?<!\w)'+CLASS_WORD+r'\s+class\s+recall\b',re.I),
    re.compile(r'([\w-]+?)\s*类(?:别)?的?召回(?:率)?'),
)


def named_recall_labels(text):
    result=[]
    for pattern in NAMED_RECALL:
        for match in pattern.finditer(text):
            value=re.sub(r'^(?:请)?(?:报告|计算|评估|返回)','',match[1]).strip(chr(96)+'"\x27')
            if value.casefold() not in ('each','every','all','both','any','the','per','各','所有','每个','每一'):
                result.append((match.group(),value))
    return result


def manifest_metric(manifest):
    """Resolve common primary-field names and reject a list without a unique objective."""
    declarations = []
    supplementary = None
    for key, value in manifest.items():
        field = normalized(key).replace(' ','')
        if field in ('metric','primarymetric','evaluationmetric','scoringmetric','selectionmetric','metricname'):
            declarations.append(value)
        elif field == 'metrics':
            supplementary = value
    if not declarations and supplementary is not None:
        declarations = [supplementary]
    if not declarations:
        return None
    names = set()
    for value in declarations:
        if isinstance(value,dict):
            if set(value) <= {'primary','secondary','diagnostics'} and 'primary' in value:
                value = value['primary']
            elif set(value) == {'name'}:
                value = value['name']
            elif set(value) == {'name','average'} and metric_name(value['name']) == 'f1':
                value = str(value['average']) + ' f1'
        for item in value if isinstance(value,list) else [value]:
            names.add(metric_name(item))
    if len(names) != 1:
        raise MetricDeclarationError('Manifest needs one unambiguous primary metric')
    return names.pop()


def resolved_name(value, labels, positive=None):
    name = metric_name(value)
    if name != 'f1':
        return name
    domain = {str(v).casefold() for v in labels}
    if len(labels) == 2 and (positive in labels if positive is not None else domain in ({'0','1'},{'false','true'},{'0.0','1.0'})):
        return 'binary_f1'
    raise MetricDeclarationError('Unqualified F1 needs an averaging rule or a binary domain with a known positive class')


def announced_clause(raw):
    """Keep the actual scoring instruction when the same sentence mentions a diagnostic."""
    if raw.rstrip().endswith(('?', '？')) or re.match(r'\s*(?:for (?:example|instance)\b|例如|比如)',raw,re.I):
        return ''
    parts = re.split(r'[,，]|\s+(?:but|while|whereas)\s+|\s+and\s+(?=[^,.;]{0,55}(?:secondary|diagnostic|for reference))',raw,flags=re.I)
    kept = []
    for part in parts:
        line = normalized(part)
        # A qualifying suffix can exclude another metric. A leading negation scopes the entire clause.
        contrast = re.search(r'\s+(?:rather than|instead of|not)\s+',line)
        if contrast and PATTERN.search(line[:contrast.start()]) and DIRECTIVE.search(line[:contrast.start()]):
            line = line[:contrast.start()]
        if line and not (NEGATIVE.search(line) or SECONDARY.search(line)):
            kept.append(line)
    return ' '.join(kept)


def declared_metric(text):
    declarations = []
    text = re.sub(r'((?:\b(?:primary |evaluation |scoring )?metric|主指标|评价指标|评估指标)\s*[:：]?)\s*\n\s*(?=\S)',r'\1 ',text,flags=re.I)
    for raw in re.findall(r'[^.!?。！？;\n；]+[.!?。！？]?',text):
        line = announced_clause(raw)
        if not line:
            continue
        for phrase,_ in named_recall_labels(line):
            line=line.replace(phrase,' binary recall ')
        primary=bool(PRIMARY.search(line))
        binding=bool(BINDING.search(line))
        if not primary and not binding and not DIRECTIVE.search(line):
            continue
        found = [m.group() for m in PATTERN.finditer(line)]
        explicit = primary or binding or re.search(r'\bmetric\s*(?:is|[:=])|指标\s*(?:为|是|[:：])',line)
        if not found and not explicit:
            continue
        invalid = not found or any(t in UNSUPPORTED for t in found)
        if re.search(r'\b(?:ovo|one (?:vs|versus) one|partial auc|top\s+\d+)\b',line):
            invalid = True
        names = {LOOKUP[t] for t in found if t in LOOKUP}
        if 'roc_auc' in names and re.search(r'\b(?:micro|weighted)\b',line):
            invalid = True
        declarations.append((primary,names,raw.strip()[:200],invalid))
    if not declarations:
        return None
    active = [r for r in declarations if r[0]] or declarations
    names = set().union(*(r[1] for r in active))
    if any(r[3] for r in active) or len(names) != 1:
        raise MetricDeclarationError('Unresolved or conflicting announced evaluation metric: '+' / '.join(r[2] for r in active)[:300])
    return names.pop()
