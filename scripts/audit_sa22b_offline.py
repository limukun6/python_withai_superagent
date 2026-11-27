"""Audit post-failure continuation and preserve the unresolved request without new calls."""
import argparse
import json
from pathlib import Path

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','original-private','offline','private','ledger','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from superagent.task_budget import TaskBudgetLedger
    from superagent.accepted_history import row_digest
    from superagent.prediction_adapter import load_adapter
    from superagent.data import load_prediction_features
    from scripts.sa7_multiturn_cases import array_fingerprint
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();offline=read(a.offline/'completion.json')
    save(a.output/'protocol.json',dict(batch='SA22B-independent-audit',offline_completion_sha256=sha(a.offline/'completion.json'),
        audit_script_sha256=sha(Path(__file__)),maximum_new_fit=0,maximum_new_API=0))
    cases=[]
    for row in read(a.prepared/'cases/registry.json'):
        name=row['id'];root=a.original_private/name/'run';model=root/'agent';spec=read(model/'task_spec.json');selection=read(model/'final_selection.json')
        query=a.prepared/'cases'/row['query'];adapters=list((root/'delivery_recovery').glob('*/prediction_adapter.json'))
        if len(adapters)!=1:raise ValueError('Expected a single accepted adapter')
        record=read(adapters[0]);adapted=load_adapter(adapters[0],query,model,spec,selection)
        actual=load_prediction_features(query,spec['model_kind'],adapted)
        frozen=read(a.offline/name/'frozen_files.json');completion=read(adapters[0].parent/'completion.json')
        paths=[a.original_private/name/'reference/predictions.csv',a.private/name/'first/predictions.csv',a.private/name/'second/predictions.csv']
        checks=dict(features=array_fingerprint(actual)==read(a.prepared/'cases'/(name+'.oracle.json'))['expected']['prediction_features_sha256'],
            exports=len({sha(p) for p in paths})==1,
            frozen_files=frozen['before']==frozen['after']=={n:sha(n) for n in frozen['before']},
            immutable_goal=all(spec[k]==adapted[k] for k in ('input_schema','metric','metric_spec','feature_shape')),
            no_new_fit=read(a.offline/name/'completion.json')['new_classifier_fits']==0,
            no_new_API=offline['all_ledger_rows_unchanged'],repeat_receipts=len(list((adapters[0].parent/'deliveries').glob('*.json')))>=2)
        if name=='D1':
            events=[json.loads(s) for s in (adapters[0].parent/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            content=[e['content'].strip() for e in events if e['kind']=='llm_response'][-1]
            if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
            checks['real_LLM_adapter']=json.loads(content)==record['proposal']
        else:
            state=read(adapters[0].parent/'recovery_state.json');observation=state['observed_evidence'][-1]
            checks.update(local_provenance=completion.get('decision_source')=='unique_structural_candidate',
                no_local_LLM=completion['new_api_requests']==0,single_array=len(observation['arrays'])==1,
                frozen_shape=observation['arrays'][0]['shape'][1:]==spec['feature_shape'])
        cases.append(dict(id=name,passed=all(checks.values()),checks=checks,rows=len(actual),decision_source=read(a.offline/name/'completion.json')['decision_source']))
    request_id='d5663eee-60d5-47df-9410-1b0d4d227a17'
    with ledger._db() as db:unknown=dict(db.execute('SELECT * FROM requests WHERE id=?',(request_id,)).fetchone())
    preserved=unknown['status']=='unknown' and unknown['course_units'] is None
    save(a.output/'new_unknown_preserved.json',dict(request_id=request_id,status=unknown['status'],row_sha256=row_digest(unknown),
        course_units=unknown['course_units'],accepted_for_future_paid_calls=False,reason='Application recovery deadline expired before final usage; no retry'))
    result=dict(status='passed' if preserved and all(r['passed'] for r in cases) and before==ledger.project_summary() else 'failed',
        cases=cases,actual_classifier_fits=0,real_API_requests=0,ledger_unchanged=before==ledger.project_summary(),new_unknown_preserved=preserved,
        evidence_boundary='successful offline continuation after a failed initial paid batch; array recovery is a local structural decision')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
