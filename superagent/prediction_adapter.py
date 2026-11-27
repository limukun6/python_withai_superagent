"""Query-bound prediction-only adaptation; frozen training/output meaning stays intact."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .events import atomic_text

VERSION='prediction-adapter-v1'
READERS=('prediction_adapter.py','predict.py','data.py','format_intake.py','native_semantics.py',
         'output_recipe.py','record_streams.py','format_sources.py')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def reader_identity():
    return {name:sha(Path(__file__).parent/name) for name in READERS}


def require(condition,message):
    from .semantic_contract import ContractProbeError
    if not condition:raise ContractProbeError('Prediction adapter: '+message)


def apply_proposal(spec,proposal,known_refs):
    from .format_sources import checked_name,roles
    from .semantic_contract import _not_label_vector
    require(isinstance(proposal,dict) and set(proposal)=={'version','input','id_fields','evidence_refs'}
            and proposal['version']==VERSION,'invalid response envelope')
    refs=proposal['evidence_refs']
    require(isinstance(refs,list) and refs and all(isinstance(r,str) and r in known_refs for r in refs),'missing real document citations')
    result=deepcopy(spec);source=proposal['input']
    require(isinstance(source,dict),'input descriptor is required')
    if source.get('format')=='records':
        require(set(source)=={'format','member','collection','delimiter','field_map'},'invalid records descriptor')
        require(bool(spec.get('input_recipe')) and bool(spec.get('input_schema',{}).get('columns')),'model has no replayable record recipe')
        member=checked_name(source['member']);_not_label_vector(member)
        require(not roles(member)&{'validation'},'cannot select validation data')
        require(source['collection'] is None or isinstance(source['collection'],str),'invalid collection')
        require(source['delimiter'] is None or isinstance(source['delimiter'],str) and len(source['delimiter'])==1,'invalid delimiter')
        names=[c['name'] for c in spec['input_schema']['columns']];mapping=source['field_map']
        require(isinstance(mapping,dict) and set(mapping)<=set(names) and all(isinstance(v,str) and v for v in mapping.values()),'invalid frozen feature mapping')
        actual=[mapping.get(n,n) for n in names]
        require(len(set(actual))==len(actual) and spec['input_schema'].get('target') not in actual,'mapping duplicates a field or uses the target')
        recipe=result['input_recipe']
        require(recipe.get('codec')!='fixed-width','fixed-width prediction layout changes are not supported by this adapter')
        recipe.update(prediction_member=member,prediction_collection=source['collection'],
                      prediction_delimiter=source['delimiter'],field_map=deepcopy(mapping))
    else:
        require(bool(spec.get('native_input_recipe')),'model has no replayable native recipe')
        from .native_semantics import descriptor
        result['native_input_recipe']['prediction']=descriptor(source,training=False,modality=spec['native_input_recipe']['modality'])
    ids=proposal['id_fields'];require(isinstance(ids,dict),'id_fields must be an object')
    recipe=result.get('output_recipe')
    columns=recipe.get('columns',[]) if recipe else []
    allowed={c['name'] for c in columns if c['kind']=='input_id'}
    require(set(ids)<=allowed and all(isinstance(v,str) and v for v in ids.values()),'only existing ID source fields may change')
    for field in ids.values():
        _not_label_vector(field)
        require(field!=spec.get('input_schema',{}).get('target'),'ID field cannot be the target')
    for column in columns:
        if column['name'] in ids:column['field']=ids[column['name']]
    return result


def preflight(query,spec):
    from .data import load_prediction_features
    from .output_recipe import probe_output_context
    kind=spec.get('model_kind') or spec['task_kind'].removesuffix('_classification')
    features=load_prediction_features(query,kind,spec)
    require(len(features)>0,'empty prediction input')
    result=dict(rows=len(features),performed_before_prediction=True,feature_values_disclosed=False)
    if spec.get('output_recipe') is not None:result['output']=probe_output_context(query,spec,len(features))
    return result


def write_adapter(destination,query,model_dir,proposal,known_refs,preflight_result):
    model_dir=Path(model_dir);selection=json.loads((model_dir/'final_selection.json').read_text(encoding='utf-8'))
    record=dict(version=VERSION,proposal=proposal,known_document_refs=sorted(known_refs),
        query_sha256=sha(query),model_sha256=sha(model_dir/selection['model_file']),task_spec_sha256=sha(model_dir/'task_spec.json'),
        reader_sha256=reader_identity(),preflight=preflight_result,training_semantics_changed=False)
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Do not overwrite prediction adapters')
    with atomic_text(destination) as stream:stream.write(json.dumps(record,ensure_ascii=False,indent=2))
    return record


def load_adapter(path,query,model_dir,spec,selection):
    path=Path(path);require(path.stat().st_size<=65536,'adapter artifact exceeds 64 KiB')
    record=json.loads(path.read_text(encoding='utf-8'))
    require(record.get('version')==VERSION,'unsupported adapter artifact')
    require(record.get('query_sha256')==sha(query),'query package changed')
    require(record.get('model_sha256')==sha(Path(model_dir)/selection['model_file']),'model changed')
    require(record.get('task_spec_sha256')==sha(Path(model_dir)/'task_spec.json'),'frozen task specification changed')
    require(record.get('reader_sha256')==reader_identity(),'prediction reader version changed')
    return apply_proposal(spec,record['proposal'],set(record['known_document_refs']))
