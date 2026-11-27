"""Independent audit: immutable models, equivalent features/exports, known usage and cache replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','live','private','ledger','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from superagent.prediction_adapter import load_adapter
    from superagent.data import load_prediction_features
    from superagent.task_budget import TaskBudgetLedger
    from scripts.sa7_multiturn_cases import array_fingerprint
    from scripts.run_sa23_delivery_live import identity
    ledger=TaskBudgetLedger(a.ledger);summary=read(a.live/'completion.json');protocol=read(a.live/'protocol.json')
    save(a.output/'protocol.json',dict(batch='SA23-independent-delivery-audit',live_completion_sha256=sha(a.live/'completion.json'),
        audit_script_sha256=sha(Path(__file__)),actual_classifier_fits=0,real_API_requests=0))
    cases=[];inputs=outputs=units=0
    for row in read(a.prepared/'cases/registry.json'):
        name=row['id'];public=a.live/name;root=a.private/name/'run';model=root/'agent';completion=read(public/'completion.json')
        spec=read(model/'task_spec.json');selection=read(model/'final_selection.json');query=a.prepared/'cases'/row['query']
        adapters=list((root/'delivery_recovery').glob('*/prediction_adapter.json'))
        if len(adapters)!=1:raise ValueError('Expected one accepted recovery adapter')
        artifact=read(adapters[0]);adapted=load_adapter(adapters[0],query,model,spec,selection)
        features=load_prediction_features(query,spec['model_kind'],adapted)
        oracle=read(a.prepared/'cases'/(name+'.oracle.json'))
        freeze=read(public/'frozen_files.json')
        protected=all(spec[k]==adapted[k] for k in spec if k not in ('input_recipe','native_input_recipe','output_recipe'))
        before_output=deepcopy(spec['output_recipe']);after_output=deepcopy(adapted['output_recipe'])
        for recipe in (before_output,after_output):
            for col in recipe['columns']:
                if col['kind']=='input_id':col.pop('field',None)
        preserved_recipes=True
        for key,allowed in [('input_recipe',{'prediction_member','prediction_collection','prediction_delimiter','field_map'}),('native_input_recipe',{'prediction'})]:
            left=spec.get(key,{});right=adapted.get(key,{})
            preserved_recipes &= {k:v for k,v in left.items() if k not in allowed}=={k:v for k,v in right.items() if k not in allowed}
        calls=read(public/'metered_usage.json');delta=read(public/'ledger_delta.json');byid={r['id']:r for r in delta};receipts=[]
        for call in calls:
            with ledger._db() as db:canonical=dict(db.execute('SELECT * FROM requests WHERE id=?',(call['budget_request_id'],)).fetchone())
            resolution=json.loads(canonical['payload'])['resolution']
            checks=dict(row_matches=canonical==byid[call['budget_request_id']],known=canonical['status']=='known',
                model=resolution['model']==call['served_model']=='DeepSeek-V4.1-Flash',
                tokens=(resolution['input_tokens'],resolution['output_tokens'])==(call['input_tokens'],call['output_tokens']),
                cost=canonical['course_units']==round(call['course_cost_rmb']*1e9),settled=call['persistent_settlement_complete'])
            receipts.append(dict(id=call['budget_request_id'],checks=checks));inputs+=call['input_tokens'];outputs+=call['output_tokens'];units+=canonical['course_units']
        events=[json.loads(s) for s in (adapters[0].parent/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        replies=[e['content'].strip() for e in events if e['kind']=='llm_response'];content=replies[-1]
        parsed=[]
        for text in replies:
            if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
            parsed.append(json.loads(text))
        tool_actions=[r for r in parsed if isinstance(r,dict) and r.get('action') in ('inspect_header','inspect_array','inspect_image','inspect_batch')]

        if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
        train_calls=[json.loads(s) for s in (model/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(s).get('event')=='llm_call']
        paths=[a.private/name/f'{folder}/predictions.csv' for folder in ('reference','delivery','replay')]
        checks=dict(protected_training_semantics=protected,preserved_training_reader=preserved_recipes,
            visible_output_meaning=before_output==after_output,feature_fingerprint=array_fingerprint(features)==oracle['expected']['prediction_features_sha256'],
            frozen_files_current=freeze['before']==freeze['after']=={n:sha(n) for n in freeze['before']},
            equal_exports=len({sha(path) for path in paths})==1,scope_limit=len(delta)<=3,
            real_accepted_adapter=json.loads(content)==artifact['proposal'],all_receipts=len(receipts)==len(delta) and all(all(r['checks'].values()) for r in receipts),
            no_training_API=len(train_calls)==0,recovery_fits_zero=completion['recovery_new_fits']==0,
            cached_replay_API_zero=completion['cached_replay_new_API']==0,
            same_scope=read(root/'training_protocol.json')['task_budget_id']==delta[0]['scope'],real_JSON_tool_used=bool(tool_actions),
            actual_failed_input=bool(read(adapters[0].parent/'recovery_state.json')['failed_actions']),
            source_bound=protocol['source_sha256']==identity(),historical_rows_preserved=summary['historical_rows_unchanged'])
        cases.append(dict(id=name,passed=all(checks.values()),checks=checks,receipts=receipts,JSON_tool_actions=tool_actions,
            recovery_requests=len(delta),new_classifier_fits=0,rows=len(features)))
    result=dict(status='passed' if all(r['passed'] for r in cases) else 'failed',cases=cases,input_tokens=inputs,output_tokens=outputs,
                audited_course_cost_rmb=units/1e9,actual_classifier_fits=0,real_API_requests=0)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
