"""Export observed run facts and aligned predictions for trusted B scheduler code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import joblib
from superagent.prediction_evidence import fingerprint


def export_ab(run):
    read=lambda name:json.loads((run/name).read_text(encoding='utf-8'))
    if not (run/'run_outcome.json').is_file():raise ValueError('AB run is still active; only export a terminal debugging snapshot')
    manifest=read('run_manifest.json');outcome=read('run_outcome.json');history=read('history.json')
    from superagent.ab_catalog import untried_candidates
    from superagent.metrics import MetricSpec
    from superagent.ab_scheduler import CONFIRMATIONS
    metric=MetricSpec.from_payload({'metric_spec':manifest['task_contract']['metric']})
    if metric.requires_scores:raise ValueError('AB score-vector export is not enabled for AUC')
    predictions=read('prediction_evidence.json');bytrial={r['trial']:r for r in history};splits={};provenance={}
    if len(bytrial)!=len(history):raise ValueError('Duplicated history trial identity')
    if len({p['trial'] for p in predictions})!=len(predictions):raise ValueError('Duplicated trial prediction evidence')
    for item in predictions:
        split=item['split_id']
        if split not in ('primary',*CONFIRMATIONS):raise ValueError('Unknown evidence split')
        if split not in splits:
            path=run/(split+'.joblib');payload=joblib.load(path);ids=read(split+'_split.json')['search_indices']
            splits[split]=(payload,ids,hashlib.sha256(path.read_bytes()).hexdigest())
            provenance[split]=dict(payload_sha256=splits[split][2],split_sha256=hashlib.sha256((run/(split+'_split.json')).read_bytes()).hexdigest())
        payload,ids,digest=splits[split];row=bytrial[item['trial']]
        if row.get('primary_metric')!=metric.name or metric.check_identity(row.get('metric_spec'))['status']!='confirmed':
            raise ValueError('Observation metric identity differs from task')
        if (row['status']!='completed' or row['candidate']!=item['candidate_id'] or row['split_id']!=split
                or item['row_ids']!=ids or item['row_id_scope']!='positions within this run training input'
                or item['evaluation_payload_sha256']!=digest or item['label_sha256']!=fingerprint(payload['y_search'])
                or item['prediction_sha256']!=fingerprint(item['predictions'])):
            raise ValueError('Invalid split-aware prediction identity')
        measured=metric.evaluate(payload['y_search'],item['predictions'])['score']
        if not math.isclose(measured,row['search_score'],abs_tol=1e-10):raise ValueError('Stored score differs from local training labels')
    if {r['trial'] for r in history if r['status']=='completed'}!={p['trial'] for p in predictions}:
        raise ValueError('Every completed observation needs one bound prediction record')
    fits=outcome['classifier_fit_accounting'];limit=manifest['fit_limit']
    state=dict(ab_schema_version=1,task=manifest['task_contract'],history=history,
        available_candidates=untried_candidates(manifest['capabilities'],history),prediction_evidence=predictions,
        registered_confirmation_splits=list(CONFIRMATIONS),
        budget=dict(fit_limit=limit,**fits,remaining_fit_allowance=max(0,limit-fits['attempted_classifier_fits']),
            reserved_final_refit=0,llm_request_limit=manifest['llm_request_limit'],used_llm_requests=outcome['new_api_requests'],
            elapsed_seconds=outcome['elapsed_seconds'],time_limit_seconds=manifest['time_limit_seconds']),
        provenance=dict(source='terminal AB artifacts; training-only evidence, no model loading',
            executable_for_continuation=False,terminal_delivery_status=outcome.get('delivery_status'),
            requested_planner_completed=outcome['requested_planner_completed'],
            manifest_sha256=hashlib.sha256((run/'run_manifest.json').read_bytes()).hexdigest(),
            outcome_sha256=hashlib.sha256((run/'run_outcome.json').read_bytes()).hexdigest(),splits=provenance))
    if (run/'training_profile.json').is_file():
        profile=read('training_profile.json')
        if read('state_1.json').get('training_profile')!=profile:raise ValueError('Training profile differs from the actual initial state')
        state['training_profile']=profile
    return state


def export(run_dir,fit_limit=11,llm_limit=3):
    run=Path(run_dir).resolve()
    if (run/'primary_split.json').is_file():return export_ab(run)
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
