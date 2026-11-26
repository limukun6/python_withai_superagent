"""Explicitly revalidate a previous accepted, metered reply into a current-reader artifact. No API."""
import argparse
import json
from pathlib import Path

from superagent.data import load_training_task
from superagent.format_sources import package_files
from superagent.format_intake import DECODE_ERRORS,adapt_training
from superagent.semantic_contract import evidence_workspace,compile_plan,_output_overlay,probe_prediction_input
from superagent.semantic_artifact import write_artifact,load_artifact,file_digest
from superagent.task_budget import TaskBudgetLedger


def migrate(package,prediction,accepted_receipt,ledger_path,destination):
    accepted_receipt=Path(accepted_receipt)
    receipt=json.loads(accepted_receipt.read_text(encoding='utf-8'))
    if receipt.get('probe')!='passed' or not receipt.get('source_citations_checked'):
        raise ValueError('Only an accepted compiled reply can be migrated')
    proposal=receipt['proposal'];folder=accepted_receipt.parent
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    bodies=[]
    for event in events:
        if event.get('kind')!='llm_response':continue
        content=event['content'].strip()
        if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
        bodies.append(json.loads(content))
    if not bodies or bodies[-1]!=proposal:raise ValueError('Accepted proposal does not match the final recorded response')
    usage=[json.loads(s) for s in (folder/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
    usage=[e for e in usage if e.get('event')=='llm_call'];last=usage[-1]
    ledger=TaskBudgetLedger(ledger_path)
    with ledger._db() as db:
        native=db.execute('SELECT status,payload FROM requests WHERE id=?',(last['budget_request_id'],)).fetchone()
    if native is None or native['status']!='known':raise ValueError('Original response has no known canonical usage record')
    resolution=json.loads(native['payload'])['resolution']
    if resolution['input_tokens']!=last['input_tokens'] or resolution['output_tokens']!=last['output_tokens']:
        raise ValueError('Original usage record differs from metering')
    from contextlib import ExitStack
    with ExitStack() as stack:
        files=stack.enter_context(package_files(package));workspace=evidence_workspace(files)
        if proposal.get('version')=='semantic-output-v1':
            try:base=load_training_task(package,read_declarations=True)
            except DECODE_ERRORS:base=adapt_training(package)
            task=_output_overlay(base,workspace,proposal)
        else:
            other=stack.enter_context(package_files(prediction)) if prediction is not None and Path(prediction).resolve()!=Path(package).resolve() else None
            task=compile_plan(files,workspace,proposal,prediction_files=other)
        if prediction is not None:probe_prediction_input(prediction,task)
    provenance=dict(kind='explicit_migration_revalidated_with_current_readers',
        original_receipt_sha256=file_digest(accepted_receipt),original_request_id=last['budget_request_id'],
        served_model=last.get('served_model'),input_tokens=last['input_tokens'],output_tokens=last['output_tokens'],
        original_course_cost_rmb=last['course_cost_rmb'],new_API_requests=0,
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
