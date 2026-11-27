"""Cited normalizations local to one task compilation, never global parser overrides."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import re

VERSION='semantic-rules-v1'
ACTIVE=ContextVar('superagent_task_rule_resolution',default=None)
INSTRUCTION=(
    'If a task declaration fails because its wording is not recognized, you may wrap the otherwise requested '
    'interpretation in {version:"semantic-rules-v1",interpretation:<complete existing task/native/output JSON>,'
    'rules:[{field:"method_restrictions" or "metric",quote:<verbatim complete failing clause from one document block>,'
    'evidence_ref:<that actual doc ID>,value:<supported interpretation>,reason:<why the quote entails this interpretation>} ]}. '
    'For method_restrictions value is a NONEMPTY list using only linear_only,no_pretrained_assets,no_external_data; '
    'these are execution restrictions, not the chosen model. They cannot express an arbitrary algorithm-only rule. '
    'For metric value is one implemented canonical metric from accuracy,balanced_accuracy,macro_f1,micro_f1,'
    'weighted_f1,binary_f1,binary_recall,roc_auc. Each rule must actually resolve an unrecognized clause. '
    'Do not wrap already recognized declarations or override a manifest, a known restriction, conflicting rules, '
    'worked scoring examples, targets, input columns, output meaning, or unsupported conditions. '
    'For an explicit manifest interpretation MUST be semantic-output-v2; an unrecognized prose rule can be '
    'normalized but the manifest input/target/declared metric remain authoritative. '
    'If a rule cannot be faithfully expressed, use report_unresolved with field method_restrictions,columns,or metric. '
    'An empty or invented restriction list does not authorize ignoring the task.'
)


def normalized(text):
    return ' '.join(text.replace('**','').replace('`','').strip(' .;。；\r\n\t').split())


def lookup(field,clause,known=()):
    state=ACTIVE.get()
    if state is None:return None
    for index,entry in enumerate(state['rules']):
        if entry['field']==field and normalized(entry['quote'])==normalized(clause):
            if field=='method_restrictions' and not set(known)<=set(entry['value']):
                raise ValueError('Rule normalization cannot remove an already recognized restriction')
            state['used'].add(index)
            return deepcopy(entry['value'])
    return None


@contextmanager
def scope(response,workspace):
    from .semantic_contract import _require
    from .task_constraints import FLAGS
    from .metrics import SUPPORTED
    _require(isinstance(response,dict) and set(response)=={'version','interpretation','rules'} and response['version']==VERSION,
             'Invalid task rule wrapper')
    inner=response['interpretation']
    versions={'semantic-task-v1','semantic-task-v2','semantic-task-v3','semantic-native-v1','semantic-native-v2','semantic-native-v3','semantic-output-v1','semantic-output-v2'}
    _require(isinstance(inner,dict) and inner.get('version') in versions,'Rules need an existing supported interpretation')
    entries=response['rules'];documents={d['id']:d for d in workspace['documents']};seen=set();rules=[]
    _require(isinstance(entries,list) and 1<=len(entries)<=8,'Rules require 1 to 8 cited normalizations')
    for entry in entries:
        _require(isinstance(entry,dict) and set(entry)=={'field','quote','evidence_ref','value','reason'},'Invalid rule fields')
        field=entry['field'];quote=entry['quote'];ref=entry['evidence_ref'];value=entry['value']
        _require(field in ('method_restrictions','metric'),'Only unknown method/metric wording is normalizable')
        _require(isinstance(quote,str) and 0<len(quote)<=2000 and isinstance(ref,str) and ref in documents and quote in documents[ref]['text'],
                 'Rule needs a verbatim quote in the referenced task document')
        _require(isinstance(entry['reason'],str) and 0<len(entry['reason'].strip())<=1200,'Rule needs a bounded explanation')
        key=(field,normalized(quote));_require(key not in seen,'Duplicate task rule');seen.add(key)
        if field=='method_restrictions':
            _require(isinstance(value,list) and value and all(isinstance(v,str) and v in FLAGS for v in value) and len(set(value))==len(value),
                     'Rule needs nonempty implemented restrictions')
        else:_require(isinstance(value,str) and value in SUPPORTED,'Rule needs an implemented metric')
        rules.append(deepcopy(entry))
    state=dict(rules=rules,used=set());token=ACTIVE.set(state)
    try:yield inner,state
    finally:ACTIVE.reset(token)


def compile_response(package,files,workspace,response,*,prediction_files=None,base_task=None):
    """One compiler for live intake, artifact reload and explicit receipt migration."""
    from .semantic_contract import _require,_output_overlay,compile_plan,bind_semantic_rules,BINDING_ERRORS
    from .native_semantics import compile_native,explicit_manifest
    from .data import load_training_task
    from .format_intake import adapt_training,DECODE_ERRORS
    def compile_inner(inner):
        version=inner.get('version') if isinstance(inner,dict) else None
        if explicit_manifest(files) or base_task is not None:
            _require(version in ('semantic-output-v1','semantic-output-v2'),'An explicit input contract permits only an output overlay')
        if version in ('semantic-output-v1','semantic-output-v2'):
            task=base_task
            if task is None:
                try:task=load_training_task(package,read_declarations=True)
                except BINDING_ERRORS:raise
                except DECODE_ERRORS:task=adapt_training(package)
            task=_output_overlay(task,workspace,inner)
        elif version in ('semantic-native-v1','semantic-native-v2','semantic-native-v3'):
            task=compile_native(files,workspace,inner,prediction_files=prediction_files)
        else:task=compile_plan(files,workspace,inner,prediction_files=prediction_files)
        return bind_semantic_rules(task,workspace)
    if isinstance(response,dict) and response.get('version')==VERSION:
        with scope(response,workspace) as (inner,state):
            task=compile_inner(inner)
            _require(state['used']==set(range(len(state['rules']))),'Rule did not resolve an unrecognized declaration')
            task.spec['rule_resolution']=dict(version=VERSION,rules=state['rules'],
                source_citations_checked=True,unrecognized_clauses_replayed=True,semantic_entailment_certified=False)
            return task
    return compile_inner(response)
