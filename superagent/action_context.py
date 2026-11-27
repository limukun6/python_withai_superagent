"""Bounded, non-executable structural summaries of unsuccessful actions."""
import json
import math

MAX_BYTES=8192
VERSIONS={'semantic-task-v1','semantic-task-v2','semantic-task-v3','semantic-native-v1',
          'semantic-native-v2','semantic-output-v1','semantic-output-v2'}
TOOLS={'inspect_header','inspect_array','inspect_image'}
CHECKS=dict(task_adaptation='Decode the documented training input with an installed reader.',
    contract_probe='Compile the cited task contract and replay prediction features and output rules before fitting.',
    structure_inspection='Inspect the cited member in the correct package and role without exposing values.',
    training_method='Execute the selected registered candidate and account for its actual classifier fits.',
    response_format='Return one requested JSON object after the preceding completed response was settled.')
CHECKS['prediction_delivery']='Replay the new input and output ID sources against the unchanged frozen model contract; do not train.'


def summarize_action(action):
    truncated=False
    def scalar(value):
        nonlocal truncated
        if isinstance(value,str):
            if len(value)>192:truncated=True
            return value[:192]
        if value is None or type(value) in (bool,int):return value
        if type(value) is float and math.isfinite(value):return value
        truncated=True
        return '<non-scalar omitted>'
    def fields(value,keys):
        return {k:scalar(value[k]) for k in keys if k in value} if isinstance(value,dict) else {}
    def limited(values,count):
        nonlocal truncated
        if len(values)>count:truncated=True
        return values[:count]
    def mapping(value):
        nonlocal truncated
        if not isinstance(value,dict):return {}
        if len(value)>64:truncated=True
        return {str(scalar(k)):scalar(v) for k,v in list(value.items())[:64]}
    def descriptor(value):
        result=fields(value,('member','collection','delimiter','format','path','target_path','features_key','target_key','prefix'))
        if isinstance(value,dict) and isinstance(value.get('field_map'),dict):result['field_map']=mapping(value['field_map'])
        return result
    def output(value):
        nonlocal truncated
        result=fields(value,('version','delimiter','encoding','header'))
        if not isinstance(value,dict):return result
        columns=value.get('columns')
        if isinstance(columns,list):
            if len(columns)>16:truncated=True
            result['columns']=[fields(c,('name','kind','field','unique','start','class_label')) for c in columns[:16]]
            for source,column in zip(columns[:16],result['columns']):
                if isinstance(source,dict) and isinstance(source.get('mapping'),dict):column['mapping']=mapping(source['mapping'])
            result['column_count']=len(columns)
        result['order']=fields(value.get('order'),('mode','member','id_column'))
        return result
    def tool(value):
        return fields(value,('action','member','source','collection','delimiter','key'))

    if action is None:return None
    result=dict(version='failed-action-summary-v1',not_executable=True)
    name=action.get('action') if isinstance(action,dict) else None
    name=name if isinstance(name,str) else None
    version=action.get('version') if isinstance(action,dict) else None
    version=version if isinstance(version,str) else None
    if not isinstance(action,dict):
        result['kind']='unrecognized_envelope';result['input_type']=type(action).__name__
    elif name in TOOLS:
        result.update(kind='structure_tool',tool=tool(action))
    elif name=='inspect_batch':
        values=action.get('requests');values=values if isinstance(values,list) else []
        result.update(kind='structure_batch',requests=[tool(r) for r in values[:2]],request_count=len(values))
        if len(values)>2:truncated=True
    elif version=='prediction-adapter-v1':
        result.update(kind='prediction_adapter',selected_input=descriptor(action.get('input')),
                      id_fields=mapping(action.get('id_fields')))
    elif version in VERSIONS:
        result.update(kind='task_contract',proposal_version=action['version'])
        plan=action.get('plan')
        if isinstance(plan,dict):
            selected=fields(plan,('target','metric','modality','positive_label'))
            for name in ('training','prediction'):
                if name in plan:selected[name]=descriptor(plan[name])
            columns=plan.get('columns')
            if isinstance(columns,list):
                selected.update(columns=[fields(c,('name','type')) for c in columns[:64]],column_count=len(columns))
                if len(columns)>64:truncated=True
            if isinstance(plan.get('image_preprocessing'),dict):
                selected['image_preprocessing']=fields(plan['image_preprocessing'],('width','height','mode','resize'))
            if isinstance(plan.get('split'),dict):
                split=plan['split'];selected['split']=fields(split,('field',))
                if isinstance(split.get('values'),dict):
                    selected['split']['values']={k:([scalar(v) for v in limited(value,16)] if isinstance(value,list) else scalar(value))
                        for k,value in split['values'].items() if k in ('train','test','validation')}
            if isinstance(plan.get('output'),dict):selected['output']=output(plan['output'])
            result['selected']=selected
        elif isinstance(action.get('output'),dict):result['selected_output']=output(action['output'])
        evidence=action.get('evidence')
        if isinstance(evidence,dict):
            result['evidence_refs']={k:[scalar(r) for r in limited(refs,4)] for k,refs in evidence.items()
                if k in ('training','prediction','target','columns','modality','metric','output','split') and isinstance(refs,list)}
    elif set(action)<= {'candidate','split'}:
        result.update(kind='registered_method',selected=fields(action,('candidate','split')))
    else:result['kind']='unrecognized_envelope'
    result['truncated']=truncated
    if len(json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf-8'))>MAX_BYTES:
        # Do not turn a truncated preview into a runnable replacement proposal.
        result=dict(version='failed-action-summary-v1',not_executable=True,kind=result['kind'],truncated=True,
                    omitted_due_to_size=True,reason='Structural summary exceeds byte cap; inspect cited structure or propose a full revised contract.')
    return result
