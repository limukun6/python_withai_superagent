"""Frozen-factor effects and actual proposal use; never train or call an API."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def run(args):
    completion=read(args.run/'completion.json')
    if completion['status']!='completed':raise ValueError('Only a completed predeclared study can be summarized')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);protocol=read(args.run/'protocol.json');tasks={};all_usage=[]
    records=read(args.run/'results.json')
    for name in sorted({r['task'] for r in records}):
        details={};predictions={};contexts={}
        for row in (r for r in records if r['task']==name):
            arm=row['arm'];directory=args.run/(name+'_'+arm);target=root/directory.name;target.mkdir()
            for file in ('run_manifest.json','final_selection.json','classifier_fits.json','verification.json','llm_usage.jsonl'):
                shutil.copy2(directory/file,target/file)
            history=read(directory/'history.json');events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            requests=[json.loads(e['messages'][1]['content']) for e in events if e['kind']=='llm_request']
            usage=[json.loads(line) for line in (directory/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
            calls=[r for r in usage if r['event']=='llm_call'];all_usage.extend(calls)
            proposals=[r for r in history if r.get('proposal')]
            contexts[arm]={k:v for k,v in requests[0].items() if k not in ('training_profile','instruction','context_version')}
            with (directory/'predictions/rows.csv').open(encoding='utf-8',newline='') as file:predictions[arm]=[r['prediction'] for r in csv.DictReader(file)]
            manifest=read(directory/'run_manifest.json');outcome=row['outcome']
            details[arm]=dict(profile=manifest['task_profile'],contract=manifest['proposal_contract'],score=row['evaluation']['score'],
                metric=row['evaluation']['primary_metric'],rows=row['evaluation']['rows'],selected=outcome['selected_candidate'],
                classifier_fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],seconds=row['process_seconds'],
                requests=outcome['new_api_requests'],input_tokens=sum(r['input_tokens'] for r in calls),output_tokens=sum(r['output_tokens'] for r in calls),
                known_course_cost_rmb=outcome['known_course_cost_rmb'],reload_identical=row['evaluation']['reload_identical'],
                null_expectations=sum(r['proposal'].get('expected_change') is None for r in proposals),
                purposes=dict(Counter(r['proposal'].get('purpose','strict-v1-improvement') for r in proposals)),
                expectation_statuses=dict(Counter((r.get('hypothesis_check') or {}).get('status') for r in proposals)),
                expected_directions=dict(Counter((r['proposal'].get('expected_change') or {}).get('direction','unstated') for r in proposals)),
                explored_routes=sorted({manifest['capabilities'][r['candidate']]['route'] for r in history if r['split_id']=='primary'}),
                first_payload_sha256=digest(requests[0]),history_sha256=sha(directory/'history.json'),events_sha256=sha(directory/'events.jsonl'),
                predictions_sha256=sha(directory/'predictions/rows.csv'))
            slim=[dict(trial=r['trial'],candidate=r['candidate'],purpose=r['proposal'].get('purpose'),
                uncertainty=r['proposal'].get('uncertainty'),expected_change=r['proposal'].get('expected_change'),
                hypothesis_check=r.get('hypothesis_check')) for r in proposals]
            save(target/'proposal_measurements.json',slim)
        equal=all(context==next(iter(contexts.values())) for context in contexts.values())
        factors=None
        if len(details)==4:
            scores={a:r['score'] for a,r in details.items()}
            factors=dict(training_profile_average_delta=((scores['10']-scores['00'])+(scores['11']-scores['01']))/2,
                calibrated_contract_average_delta=((scores['01']-scores['00'])+(scores['11']-scores['10']))/2,
                interaction=scores['11']-scores['10']-scores['01']+scores['00'])
        contrasts={}
        pairs=[('10','00'),('11','01'),('01','00'),('11','10'),('11','00')] if len(details)==4 else [('11','00')]
        for left,right in pairs:
            contrasts[left+'-'+right]=dict(score_delta=details[left]['score']-details[right]['score'],
                prediction_disagreements=sum(a!=b for a,b in zip(predictions[left],predictions[right],strict=True)))
        tasks[name]=dict(arms=details,factor_effects=factors,contrasts=contrasts,
            first_contexts_identical_after_removing_treatments=equal,scope='Single known-task development split, one LLM run per cell; no significance claim')
    if len({r['budget_request_id'] for r in all_usage})!=len(all_usage):raise ValueError('Request double counting')
    summary=dict(tasks=tasks,completion=completion,input_tokens=sum(r['input_tokens'] for r in all_usage),
        output_tokens=sum(r['output_tokens'] for r in all_usage),real_api_requests=len(all_usage),
        known_course_cost_rmb=sum(r['course_cost_rmb'] for r in all_usage),price_scope='Frozen course accounting, actual platform debit unknown',
        original_historical_unknown=11,valid_for_budgeted_final_report=False)
    save(root/'summary.json',summary)
    for file in ('protocol.json','completion.json','all_models_frozen.json','physical_fits.jsonl'):shutil.copy2(args.run/file,root/file)
    print(json.dumps(dict(tasks={k:dict(scores={a:r['score'] for a,r in v['arms'].items()},factor_effects=v['factor_effects']) for k,v in tasks.items()},
        real_api_requests=len(all_usage),actual_classifier_fits=completion['actual_classifier_fits'],known_course_cost_rmb=summary['known_course_cost_rmb'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
