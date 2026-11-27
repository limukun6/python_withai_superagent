"""Bounded post-training input recovery. No training or model mutation occurs here."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import uuid

from .events import RunLogger
from .prediction_adapter import sha,apply_proposal,preflight,write_adapter,load_adapter

INSTRUCTION=(
    'The classifier is already trained. Repair only prediction input adaptation; never change its training '
    'features/order/types, target, classes, metric, preprocessing or visible output format/label/probability meanings. '
    'Return exactly {version:"prediction-adapter-v1",input:<descriptor>,id_fields:{<existing output ID column>:<actual query field>},evidence_refs:[<real doc IDs>]}. '
    'For a frozen record model input is {format:"records",member,collection:null or actual collection,delimiter:null or one character,field_map:{canonical_feature:actual_query_field}}. '
    'Keep the canonical feature names from frozen_contract. field_map changes names, not feature semantics. '
    'For a frozen native model input is {format:"npy",path}, {format:"npz",path,features_key}, or '
    '{format:"image_folder",prefix}; retain the same modality/preprocessing/shape. '
    'id_fields may only change source fields of existing input_id output columns, never add/delete/rename output columns. Use {} if unnecessary. '
    'Cite original or prediction documents for the new adaptation. Prediction documents use prediction:doc:... IDs. '
    'If structure is unknown use inspect_header/inspect_array/inspect_image or inspect_batch with at most two child tools, '
    'each with real evidence_refs, all source="prediction". At most two tools total; requests share training scope/remaining allowance. '
    'The new package contains only prediction features/documents; do not open label vectors or validation sources. '
    'Unresolved goal meaning cannot be invented. Return full data-only JSON; no code, labels, raw rows or invented IDs.'
)


def find_cached(query,root):
    root=Path(root);model=root/'agent'
    if not model.is_dir() or not Path(query).is_file():return None
    candidates=sorted((root/'delivery_recovery').glob('*/prediction_adapter.json'))[-32:][::-1]
    if not candidates:return None
    spec=json.loads((model/'task_spec.json').read_text(encoding='utf-8'))
    selection=json.loads((model/'final_selection.json').read_text(encoding='utf-8'))
    for path in candidates:
        try:load_adapter(path,query,model,spec,selection)
        except (ValueError,OSError,KeyError,TypeError):continue
        return path
    return None


def training_context(root):
    from .prediction_adapter import require
    record=json.loads((Path(root)/'training_protocol.json').read_text(encoding='utf-8'))
    package=Path(record.get('task_package') or '')
    require(package.is_file() and bool(record.get('task_package_sha256')),'original file task package binding unavailable')
    require(sha(package)==record['task_package_sha256'],'original task package changed')
    return record,package


def bound_client(root,directory):
    from .prediction_adapter import require
    from .unified import configured_flash
    protocol,_=training_context(root)
    require(protocol.get('feedback_reflection') not in ('control','local','opening-local')
            or protocol.get('task_interpretation')=='proactive','explicit local/control path must not start paid delivery recovery')
    scope=os.getenv('SUPERAGENT_TASK_BUDGET_ID');ledger=os.getenv('SUPERAGENT_PLANNER_LEDGER')
    require(bool(scope) and scope==protocol.get('task_budget_id'),'prediction must use the original training budget scope')
    require(bool(ledger) and Path(ledger).resolve()==Path(protocol.get('budget_ledger') or '').resolve(),'prediction must use the original cumulative ledger')
    client=configured_flash(directory)
    require(client is not None and getattr(client,'account',None) is not None,'configured cumulative account required')
    account=client.account
    with account.ledger._db() as db:used=db.execute('SELECT COUNT(*) FROM requests WHERE scope=?',(scope,)).fetchone()[0]
    remaining=min(3,getattr(account,'max_requests',3))-used
    require(remaining>0,'no remaining shared training/prediction requests')
    client.max_requests=min(client.max_requests,remaining)
    return client,used


def recover_prediction(query,root,output,error):
    """Return a verified adapter, or None while retaining the frozen model."""
    from .semantic_contract import evidence_workspace,_inspect_header,_diagnostic,ContractProbeError
    from .structure_tools import inspect_structure
    from .format_sources import package_files
    from .format_intake import DECODE_ERRORS
    from .recovery import RecoveryController
    from .experiment_llm import MeteredResponseFormatError
    root=Path(root);model=root/'agent';directory=root/'delivery_recovery'/uuid.uuid4().hex
    directory.mkdir(parents=True,exist_ok=False);log=RunLogger(directory);log.initialize_usage_ledger()
    controller=RecoveryController(log,fit_limit=0,deadline=time.monotonic()+120)
    controller.state['installed_capabilities']=['inspect_header','inspect_array','inspect_image','inspect_batch','repair_prediction_adapter','retain_model']
    snapshot=None;client=None
    try:
        protocol,package=training_context(root)
        spec=json.loads((model/'task_spec.json').read_text(encoding='utf-8'))
        selection=json.loads((model/'final_selection.json').read_text(encoding='utf-8'))
        snapshot={str(p):sha(p) for p in [model/selection['model_file'],model/'task_spec.json',model/'final_selection.json']}
        if (model/'classifier_fits.json').is_file():snapshot[str(model/'classifier_fits.json')]=sha(model/'classifier_fits.json')
        controller.state['prior_training_classifier_fits']=(json.loads((model/'classifier_fits.json').read_text(encoding='utf-8'))
            .get('attempted_classifier_fits')) if (model/'classifier_fits.json').is_file() else None
        if not (spec.get('input_recipe') or spec.get('native_input_recipe')):
            raise ContractProbeError('Prediction recovery needs a replayable record or native input recipe')
        if spec.get('input_recipe',{}).get('codec')=='fixed-width':
            raise ContractProbeError('Fixed-width prediction layout repair is not implemented')
        if not Path(query).is_file():raise ContractProbeError('Prediction adapter v1 requires a file query package')
        cause=error.__cause__ or error
        diagnostic=_diagnostic(cause,'prediction_delivery')
        controller.failure('prediction_delivery',diagnostic,action=dict(version='prediction-adapter-v1',
            input=deepcopy(spec.get('native_input_recipe',{}).get('prediction') or spec.get('input_recipe')),id_fields={}))
        controller.advance('prediction_delivery',goal=dict(metric=spec.get('metric'),classes=spec.get('metric_spec',{}).get('labels')),
                           artifacts=[str(model/selection['model_file']),str(model/'task_spec.json')])
        with ExitStack() as stack:
            files=stack.enter_context(package_files(package));queries=stack.enter_context(package_files(query))
            workspace=evidence_workspace(files);query_workspace=evidence_workspace(queries)
            workspace['documents'] += [dict(doc,id='prediction:'+doc['id']) for doc in query_workspace['documents']]
            workspace['prediction_members']=query_workspace['members']
            if not workspace['documents']:raise ContractProbeError('No task evidence for prediction adaptation')
            known={doc['id'] for doc in workspace['documents']}
            client,used=bound_client(root,directory);controller.client=client;client.set_deadline(controller.deadline)
            log.write_json('protocol.json',dict(stage='prediction_delivery',training_scope=protocol['task_budget_id'],
                prior_scope_requests=used,maximum_new_requests=client.max_requests,scope_request_limit=3,
                original_task_sha256=sha(package),query_sha256=sha(query),frozen_files=snapshot,recovery_classifier_fit_limit=0))
            observations=[];tool_steps=0
            while not client.blocked and client.request_count<client.max_requests:
                payload=dict(instruction=INSTRUCTION,evidence_workspace=workspace,
                    frozen_contract={k:spec.get(k) for k in ('input_schema','feature_shape','metric','metric_spec','output_recipe')},
                    current_record_recipe=spec.get('input_recipe'),current_native_recipe=spec.get('native_input_recipe'),
                    diagnostic=diagnostic,tool_observations=observations,recovery_state=controller.context())
                try:proposal=client.ask(payload,log,'prediction_recovery')
                except MeteredResponseFormatError as issue:
                    if not issue.settlement_complete or client.blocked:raise
                    diagnostic=dict(stage='prediction_delivery',code=type(issue).__name__,message='Settled response was not a valid JSON object; return the requested adapter or structure tool.')
                    controller.failure('prediction_delivery',diagnostic);continue
                try:
                    if isinstance(proposal,dict) and proposal.get('action') in ('inspect_header','inspect_array','inspect_image','inspect_batch'):
                        batch=proposal.get('requests') if proposal['action']=='inspect_batch' else [proposal]
                        if not isinstance(batch,list) or not 1<=len(batch)<=2-tool_steps:raise ContractProbeError('Prediction inspection allowance reached')
                        for requested in batch:
                            if not isinstance(requested,dict) or requested.get('action') not in ('inspect_header','inspect_array','inspect_image') or requested.get('source','prediction')!='prediction':
                                raise ContractProbeError('Delivery may inspect only prediction structure')
                            request=dict(requested,source='prediction');tool_steps+=1
                            observation=(_inspect_header(queries,workspace,request) if request['action']=='inspect_header'
                                         else inspect_structure(queries,workspace,request))
                            observations.append(observation);controller.observed(request['action'],observation)
                        continue
                    adapted=apply_proposal(spec,proposal,known)
                    probe=preflight(query,adapted)
                except OSError:raise
                except DECODE_ERRORS as issue:
                    diagnostic=_diagnostic(issue,'prediction_delivery');controller.failure('prediction_delivery',diagnostic,action=proposal)
                    log.event('prediction_adapter_rejected',diagnostic=diagnostic,proposal=proposal);continue
                destination=directory/'prediction_adapter.json'
                write_adapter(destination,query,model,proposal,known,probe)
                if snapshot!={name:sha(name) for name in snapshot}:raise RuntimeError('Frozen model or fit accounting changed during prediction adaptation')
                controller.accepted('prediction_adapter_preflight',stage='prediction_ready')
                log.write_json('completion.json',dict(status='adapter_verified',actual_classifier_fits=0,
                    new_api_requests=client.request_count,known_course_cost_rmb=client.spent,frozen_files_unchanged=True))
                return destination
        controller.exhausted('Shared request allowance exhausted without a verified prediction adapter')
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as issue:
        controller.exhausted('Prediction recovery unavailable: '+type(issue).__name__)
        log.event('prediction_recovery_unavailable',diagnostic=_diagnostic(issue,'prediction_delivery'))
    try:intact=snapshot is None or snapshot=={name:sha(name) for name in snapshot}
    except OSError:intact=False
    log.write_json('completion.json',dict(status='unresolved',model_preserved=intact,actual_classifier_fits=0,
        new_api_requests=client.request_count if client else 0,known_course_cost_rmb=client.spent if client else 0))
    return None
