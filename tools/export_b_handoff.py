"""Export observed run facts and aligned predictions for trusted B scheduler code."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
from superagent.prediction_evidence import fingerprint


def export(run_dir,fit_limit=11,llm_limit=3):
    run=Path(run_dir).resolve()
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    history=read('trials.json');task=read('task_contract.json');split=read('split.json')
    # Prefer current registered/static menu already frozen in decision states.
    states=read('decision_states.json')
    last=states[-1];tried={r['candidate'] for r in history}
    available=[c for c in last['available_candidates'] if c['id'] not in tried]
    summary=read('summary.json') if (run/'summary.json').is_file() else {}
    manifest=read('run_manifest.json') if (run/'run_manifest.json').is_file() else {}
    payload_path=run/'inner_data.joblib'
    payload=joblib.load(payload_path)
    payload_sha=hashlib.sha256(payload_path.read_bytes()).hexdigest()
    completed=sum(r.get('classifier_fits',1) for r in history if r['status']=='completed')
    total=summary.get('classifier_fit_accounting',{}).get('completed_classifier_fits',completed)
    failed=sum(r['status']!='completed' for r in history)
    predictions=[]
    for row in history:
        if row['status']!='completed':continue
        path=run/f'trial_{row["trial"]}.joblib'
        artifact=joblib.load(path);record=artifact['prediction_record']
        values=record['predictions'].tolist()
        if (record['prediction_sha256']!=fingerprint(values) or len(values)!=len(split['search_indices'])
                or record['label_sha256']!=fingerprint(payload['y_search'])
                or record['evaluation_payload_sha256']!=payload_sha):
            raise ValueError('Invalid prediction identity or row order')
        predictions.append(dict(trial=row['trial'],candidate_id=row['candidate'],split_id='primary',
            row_ids=split['search_indices'],row_id_scope='positions within this run training input',
            predictions=values,evaluation_payload_sha256=record['evaluation_payload_sha256'],
            label_sha256=record['label_sha256'],prediction_sha256=record['prediction_sha256']))
    usage=[json.loads(s) for s in (run/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
    calls=sum(r['event']!='ledger_initialized' for r in usage)
    return dict(ab_schema_version=1,task=task,history=history,available_candidates=available,
        prediction_evidence=predictions,budget=dict(fit_limit=fit_limit,completed_classifier_fits=total,
            attempted_classifier_fits=total if not failed else None,unknown_failed_attempts=failed,
            remaining_fit_allowance=max(0,fit_limit-total) if not failed else None,
            reserved_final_refit=0 if summary.get('refit_succeeded') else 1,
            llm_request_limit=llm_limit,used_llm_requests=calls,
            elapsed_seconds=summary.get('elapsed_seconds'),time_limit_seconds=manifest.get('time_limit_seconds')),
        provenance=dict(source='existing run artifacts; no training or request during export',
            task_contract_sha256=hashlib.sha256((run/'task_contract.json').read_bytes()).hexdigest(),
            split_sha256=hashlib.sha256((run/'split.json').read_bytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    # Exclusive creation avoids silently replacing an agreed interface example.
    value=export(args.run_dir)
    with output.open('x',encoding='utf-8') as file:json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)
    print(f'Wrote scheduler handoff: {output}')

if __name__=='__main__':main()
