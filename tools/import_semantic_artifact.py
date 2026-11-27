"""Explicitly revalidate a previous accepted, metered reply into a current-reader artifact. No API."""
import argparse
import json
from pathlib import Path

from superagent.data import load_training_task
from superagent.format_sources import package_files
from superagent.format_intake import DECODE_ERRORS,adapt_training
from superagent.semantic_contract import evidence_workspace,compile_plan,_output_overlay,probe_prediction_input
from superagent.semantic_artifact import write_artifact,load_artifact,file_digest,digest,VERSION
from superagent.task_budget import TaskBudgetLedger
from superagent.usage_audit import money_units


def original_metering(folder,proposal):
    """Bind the accepted semantic response, not an unrelated later planner call."""
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted=[i for i,e in enumerate(events) if e.get('kind')=='semantic_contract_accepted']
    if not accepted:raise ValueError('Original semantic acceptance event is missing')
    before=events[:accepted[-1]]
    responses=[i for i,e in enumerate(before) if e.get('kind')=='llm_response']
    if not responses:raise ValueError('Original semantic response is missing')
    at=responses[-1];content=before[at]['content'].strip()
    if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
    if json.loads(content)!=proposal:raise ValueError('Accepted proposal does not match the recorded semantic response')
    requests=[e for e in before[:at] if e.get('kind')=='llm_request']
    if not requests or not requests[-1].get('budget_request_id'):raise ValueError('Original response has no recorded request ID')
    key=requests[-1]['budget_request_id']
    usage=[json.loads(s) for s in (folder/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
    matching=[e for e in usage if e.get('event')=='llm_call' and e.get('budget_request_id')==key]
    if len(matching)!=1:raise ValueError('Original response lacks unique metered usage')
    return matching[0]


def original_binding(folder,package,prediction,proposal):
    """An older reader fingerprint is allowed only through this explicit tool."""
    original=folder/'semantic_artifact.json'
    if not original.is_file():return 'not_recorded_in_legacy_receipt'
    envelope=json.loads(original.read_text(encoding='utf-8'));payload=envelope.get('payload')
    if (not isinstance(payload,dict) or payload.get('version')!=VERSION or digest(payload)!=envelope.get('payload_sha256')
            or payload.get('proposal')!=proposal):raise ValueError('Original artifact content/proposal mismatch')
    if payload.get('training_package_sha256')!=file_digest(package):raise ValueError('Original artifact belongs to different training bytes')
    mode=payload.get('prediction_mode');query=prediction
    if query is None and mode=='embedded':query=package
    if mode not in ('unbound','embedded','external'):raise ValueError('Original artifact prediction binding is invalid')
    if mode=='unbound' and query is not None:raise ValueError('Original artifact did not bind a prediction package')
    if mode!='unbound' and (query is None or payload.get('prediction_package_sha256')!=file_digest(query)):
        raise ValueError('Original artifact belongs to different prediction bytes')
    return 'original_artifact_content_and_package_hashes_verified'


def migrate(package,prediction,accepted_receipt,ledger_path,destination):
    accepted_receipt=Path(accepted_receipt)
    receipt=json.loads(accepted_receipt.read_text(encoding='utf-8'))
    if receipt.get('probe')!='passed' or not receipt.get('source_citations_checked'):
        raise ValueError('Only an accepted compiled reply can be migrated')
    proposal=receipt['proposal'];folder=accepted_receipt.parent
    last=original_metering(folder,proposal)
    binding=original_binding(folder,package,prediction,proposal)
    ledger=TaskBudgetLedger(ledger_path)
    with ledger._db() as db:
        native=db.execute('SELECT status,payload,course_units FROM requests WHERE id=?',(last['budget_request_id'],)).fetchone()
    if native is None or native['status']!='known':raise ValueError('Original response has no known canonical usage record')
    resolution=json.loads(native['payload'])['resolution']
    if (resolution['input_tokens']!=last['input_tokens'] or resolution['output_tokens']!=last['output_tokens']
            or resolution['model']!=last.get('served_model') or native['course_units']!=money_units(last['course_cost_rmb'])):
        raise ValueError('Original usage record differs from metering')
    from contextlib import ExitStack
    from superagent.semantic_contract import BINDING_ERRORS,bind_semantic_rules
    with ExitStack() as stack:
        files=stack.enter_context(package_files(package));workspace=evidence_workspace(files)
        from superagent.rule_resolution import compile_response
        other=stack.enter_context(package_files(prediction)) if prediction is not None and Path(prediction).resolve()!=Path(package).resolve() else None
        task=compile_response(package,files,workspace,proposal,prediction_files=other)
        if prediction is not None:probe_prediction_input(prediction,task)
    provenance=dict(kind='explicit_migration_revalidated_with_current_readers',
        original_receipt_sha256=file_digest(accepted_receipt),original_request_id=last['budget_request_id'],
        served_model=last.get('served_model'),input_tokens=last['input_tokens'],output_tokens=last['output_tokens'],
        original_course_cost_rmb=last['course_cost_rmb'],new_API_requests=0,
        original_source_binding=binding,
        note='Metered local records, not a provider signature; revalidated migration, not old-source identity')
    metadata=write_artifact(destination,package,prediction,task,proposal,provenance=provenance)
    load_artifact(destination,package,prediction)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True);parser.add_argument('--prediction-input',type=Path)
    parser.add_argument('--accepted-contract',type=Path,required=True);parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    metadata=migrate(args.task,args.prediction_input,args.accepted_contract,args.ledger,args.output)
    print(json.dumps(metadata,ensure_ascii=False))


if __name__=='__main__':main()
