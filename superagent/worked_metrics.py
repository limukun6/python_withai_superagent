"""Check task-authored, hard-label scoring examples against a closed list of metrics."""
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import json
import re

from .metric_declarations import MetricDeclarationError, metric_name

SUPPORTED = {'accuracy', 'micro_f1', 'balanced_accuracy', 'macro_f1', 'weighted_f1'}
HEADER = re.compile(r'^\s*(?:#{1,6}\s*)?(?:metric evidence|(?:worked )?scoring examples|评分示例)\s*[:：]?\s*$', re.I | re.M)
NUMBER = re.compile(r'(?:0(?:\.\d{1,8})?|1(?:\.0{1,8})?)\Z')


def extract(text, manifest=None):
    """Read a bounded JSON block immediately after a scoring-evidence heading."""
    blocks = []
    for match in HEADER.finditer(text):
        tail = text[match.end():].lstrip()
        fence = re.match(r'^(?:`{3}(?:json)?|json)\s*\n',tail,re.I)
        if fence:
            tail = tail[fence.end():].lstrip()
        if not tail.startswith('{'):
            continue
        try:
            value, length = json.JSONDecoder().raw_decode(tail[:65536])
        except ValueError as error:
            raise MetricDeclarationError('Malformed worked scoring examples') from error
        blocks.append((match.start(), len(text)-len(tail)+length, value))
    declarations = ([manifest] if manifest is not None else []) + [v for _,_,v in blocks]
    if any(value != declarations[0] for value in declarations):
        raise MetricDeclarationError('Conflicting worked scoring example declarations')
    for start, end, _ in reversed(blocks):
        text = text[:start] + '\n' + text[end:]
    return text, declarations[0] if declarations else None


def scores(truth, prediction, labels):
    support, selected = Counter(truth), Counter(prediction)
    correct = Counter(t for t,p in zip(truth,prediction) if t == p)
    accuracy = Fraction(sum(correct.values()),len(truth))
    f1 = {k:Fraction(2*correct[k], max(1,support[k]+selected[k])) for k in labels}
    present = [k for k in labels if support[k]]
    return {'accuracy':accuracy, 'micro_f1':accuracy,
            'balanced_accuracy':sum(Fraction(correct[k],support[k]) for k in present)/len(present),
            'weighted_f1':sum(f1[k]*support[k] for k in labels)/len(truth),
            'macro_f1':sum(f1.values())/len(labels)}


def verify(value, declared):
    if not isinstance(value,dict) or not {'candidates','labels','examples'} <= set(value) or set(value)-{'candidates','labels','examples','rounding'}:
        raise MetricDeclarationError('Worked scoring evidence requires candidates, labels and examples')
    names, labels, examples = (value[k] for k in ('candidates','labels','examples'))
    if not isinstance(names,list) or not 2<=len(names)<=len(SUPPORTED):
        raise MetricDeclarationError('Worked scores need an explicit closed list of alternatives')
    candidates = {metric_name(n) for n in names}
    if len(candidates)!=len(names) or not candidates<=SUPPORTED:
        raise MetricDeclarationError('Unsupported or duplicate worked scoring alternative')
    if (not isinstance(labels,list) or not 2<=len(labels)<=128 or
            not all(type(k) in (str,int) and type(k)==type(labels[0]) for k in labels) or len(set(labels))!=len(labels)):
        raise MetricDeclarationError('Worked scoring examples need an unambiguous example class domain')
    if not isinstance(examples,list) or not 1<=len(examples)<=16 or value.get('rounding') not in (None,'half_up'):
        raise MetricDeclarationError('Unsupported worked scoring example count or rounding rule')
    matching = candidates.copy()
    for ex in examples:
        if not isinstance(ex,dict) or set(ex)!={'truth','prediction','score'}:
            raise MetricDeclarationError('Each worked score needs illustrative truth, prediction and a displayed score')
        truth, pred, displayed = (ex[k] for k in ('truth','prediction','score'))
        if (not isinstance(truth,list) or not isinstance(pred,list) or not 0<len(truth)==len(pred)<=512
                or any(type(k)!=type(labels[0]) or k not in labels for k in truth+pred)
                or not isinstance(displayed,str) or NUMBER.fullmatch(displayed) is None):
            raise MetricDeclarationError('Invalid worked labels, row count or printed score precision')
        digits = Decimal(displayed)
        scale = 10**(-digits.as_tuple().exponent)
        units = int(digits*scale)
        def matches(score):
            difference = score*scale-units
            return (-Fraction(1,2)<=difference<Fraction(1,2) if value.get('rounding')=='half_up'
                    else abs(difference)<=Fraction(1,2))
        observed = scores(truth,pred,labels)
        matching = {n for n in matching if matches(observed[n])}
    if declared == 'f1':
        matching &= {'macro_f1','weighted_f1','micro_f1'}
        declared = None
    if declared is not None and declared not in matching or not matching:
        raise MetricDeclarationError('Declared objective and worked scoring examples disagree')
    if declared is None and len(matching)!=1:
        raise MetricDeclarationError('Worked scoring examples do not distinguish '+', '.join(sorted(matching)))
    return declared or next(iter(matching))
