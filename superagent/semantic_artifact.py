"""Data-only, source-bound reuse of an already compiled task interpretation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .events import atomic_text


VERSION='semantic-artifact-v1'
MAX_BYTES=1024*1024
READERS=('semantic_artifact.py','semantic_contract.py','format_intake.py','format_sources.py',
         'record_streams.py','schema.py','native_schema.py','data.py','metrics.py','metric_declarations.py',
         'table_input.py','task_files.py','shared_table.py','image_files.py','folder_images.py','output_recipe.py','native_semantics.py','task_questions.py','structure_tools.py','recovery.py','task_constraints.py','task_declarations.py','worked_metrics.py','contract_extraction.py','action_context.py')


class ArtifactUnavailable(ValueError):pass


def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def file_digest(path):
    path=Path(path)
    if not path.is_file():raise ArtifactUnavailable('Semantic artifact v1 requires a file package; use fresh interpretation for directory inputs')
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def reader_identity():
    root=Path(__file__).resolve().parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in READERS}


def array_digest(values):
    data=np.asarray(getattr(values,'values',values))
    if data.dtype.hasobject:
        def cell(value):
            if isinstance(value,np.generic):value=value.item()
            if isinstance(value,float):return ['float',value.hex()]
            if isinstance(value,(str,int,bool)) or value is None:return [type(value).__name__,value]
            raise ArtifactUnavailable('Unsupported scalar in normalized training content')
        body=encoded([cell(v) for v in data.ravel()])
    else:body=np.ascontiguousarray(data).tobytes()
    return hashlib.sha256(encoded(dict(shape=list(data.shape),dtype=data.dtype.str))+body).hexdigest()


def task_digest(task):
    from .metrics import MetricSpec
    return digest(dict(kind=task.kind,X=array_digest(task.X_train),y=array_digest(task.y_train),
        input_schema=task.spec.get('input_schema'),metric=MetricSpec.for_task(task).record(),
        positive_label=task.spec.get('positive_label'),input_recipe=task.spec.get('input_recipe'),native_input_recipe=task.spec.get('native_input_recipe'),
        output_recipe=task.spec.get('output_recipe'),evaluation_protocol=task.spec.get('evaluation_protocol'),
        uncertainty_resolution=task.spec.get('uncertainty_resolution')))


def last_usage(log):
    if log is None or not log.usage_path.is_file():return dict(kind='no_metered_provider_receipt')
    entries=[json.loads(line) for line in log.usage_path.read_text(encoding='utf-8').splitlines()]
    entries=[e for e in entries if e.get('event')=='llm_call']
    if not entries:return dict(kind='no_metered_provider_receipt')
    record=entries[-1]
    return dict(kind='recorded_provider_usage',**{key:record.get(key) for key in
        ('budget_request_id','served_model','input_tokens','output_tokens','course_cost_rmb','persistent_settlement_complete')})


def write_artifact(destination,package,prediction_package,task,proposal,*,provenance=None):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Do not overwrite an existing semantic artifact')
    training_hash=file_digest(package)
    if prediction_package is None:mode='unbound';prediction_hash=None
    else:
        prediction_hash=file_digest(prediction_package)
        mode='embedded' if Path(package).resolve()==Path(prediction_package).resolve() else 'external'
    payload=dict(version=VERSION,training_package_sha256=training_hash,prediction_mode=mode,
        prediction_package_sha256=prediction_hash,proposal=proposal,task_sha256=task_digest(task),
        reader_sha256=reader_identity(),provenance=provenance or dict(kind='no_metered_provider_receipt'))
    envelope=dict(payload=payload,payload_sha256=digest(payload))
    raw=json.dumps(envelope,ensure_ascii=False,indent=2,allow_nan=False)
    if len(raw.encode('utf-8'))>MAX_BYTES:raise ArtifactUnavailable('Semantic artifact exceeds its size limit')
    with atomic_text(destination) as stream:stream.write(raw+'\n')
    return dict(artifact_sha256=file_digest(destination),task_sha256=payload['task_sha256'],
                prediction_mode=mode,provider_provenance=payload['provenance'])


def load_artifact(path,package,prediction_package=None,log=None):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES:raise ValueError('Semantic artifact exceeds its size limit')
    envelope=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(envelope,dict) or set(envelope)!={'payload','payload_sha256'}:
        raise ValueError('Malformed semantic artifact envelope')
    payload=envelope['payload']
    required={'version','training_package_sha256','prediction_mode','prediction_package_sha256',
              'proposal','task_sha256','reader_sha256','provenance'}
    if not isinstance(payload,dict) or set(payload)!=required or payload['version']!=VERSION or digest(payload)!=envelope['payload_sha256']:
        raise ValueError('Semantic artifact content/version mismatch')
    if not isinstance(payload['proposal'],dict) or not isinstance(payload['provenance'],dict):
        raise ValueError('Malformed artifact proposal/provenance')
    if payload['reader_sha256']!=reader_identity():raise ValueError('Semantic artifact reader source changed; revalidate explicitly')
    if payload['training_package_sha256']!=file_digest(package):raise ValueError('Semantic artifact belongs to different training bytes')
    mode=payload['prediction_mode']
    if mode not in ('embedded','external','unbound'):raise ValueError('Invalid artifact prediction binding')
    query=prediction_package
    if query is None and mode=='embedded':query=package
    if mode=='external' and query is None:raise ValueError('Artifact needs its bound external prediction input')
    if mode=='unbound' and query is not None:raise ValueError('Artifact did not bind this prediction input; revalidate explicitly')
    if mode!='unbound' and (query is None or file_digest(query)!=payload['prediction_package_sha256']):
        raise ValueError('Semantic artifact belongs to different prediction bytes')
    from .format_sources import package_files
    from .semantic_contract import BINDING_ERRORS,bind_semantic_rules,compile_plan,evidence_workspace,_output_overlay,probe_prediction_input
    from .data import load_training_task
    from .format_intake import DECODE_ERRORS,adapt_training
    from contextlib import ExitStack
    with ExitStack() as stack:
        files=stack.enter_context(package_files(package));workspace=evidence_workspace(files)
        if payload['proposal'].get('version') in ('semantic-output-v1','semantic-output-v2'):
            try:base=load_training_task(package,read_declarations=True)
            except BINDING_ERRORS:raise
            except DECODE_ERRORS:base=adapt_training(package)
            task=_output_overlay(base,workspace,payload['proposal'])
        elif payload['proposal'].get('version') in ('semantic-native-v1','semantic-native-v2'):
            from .native_semantics import compile_native
            other=stack.enter_context(package_files(query)) if query is not None and mode=='external' else None
            task=compile_native(files,workspace,payload['proposal'],prediction_files=other)
        else:
            other=stack.enter_context(package_files(query)) if query is not None and mode=='external' else None
            task=compile_plan(files,workspace,payload['proposal'],prediction_files=other)
        bind_semantic_rules(task, workspace)
        if task_digest(task)!=payload['task_sha256']:raise ValueError('Restored semantic task differs from its accepted digest')
        if query is not None:probe_prediction_input(query,task,log)
    receipt=dict(artifact_sha256=file_digest(path),task_sha256=payload['task_sha256'],
        source=str(path.resolve()),provider_provenance=payload['provenance'],new_interpretation_API_requests=0,
        scope='task interpretation reuse, not model/search resume or provider attestation')
    if log:
        log.write_json('semantic_artifact_receipt.json',receipt)
        log.event('semantic_artifact_loaded',**receipt)
    return task
