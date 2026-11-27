"""Continue the preserved SA22 models without any API or fitting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows
from scripts.run_sa21_context_live import execute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','original-private','output','private-output','ledger','checks'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    from superagent.task_budget import TaskBudgetLedger
    from superagent.prediction_adapter import load_adapter
    from superagent.data import load_prediction_features
    from scripts.sa7_multiturn_cases import array_fingerprint
    root=Path(__file__).resolve().parents[1];clock=time.monotonic()
    checks=read(a.checks/'completion.json')
    if checks['status']!='passed' or not checks['source_unchanged']:raise ValueError('Offline checks must pass')
    if any(sha(root/n)!=h for n,h in read(a.checks/'protocol.json')['source_sha256'].items()):raise ValueError('Checked code changed')
    identity={p.relative_to(root).as_posix():sha(p) for p in (root/'superagent').glob('*.py')};identity['unified_agent.py']=sha(root/'unified_agent.py')
    plan=root/'docs/batches/SA22B_OFFLINE_RECOVERY_PLAN_20261127_zh.md'
    save(a.output/'protocol.json',dict(batch='SA22B-offline-continuation',source_sha256=identity,plan_sha256=sha(plan),
        actual_fit_limit=0,real_API_limit=0,original_failures_preserved=True))
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();old=ledger_rows(ledger);rows=[]
    env={k:v for k,v in os.environ.items() if not k.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_'))}
    env.pop('SUPERAGENT_TASK_BUDGET_ID',None);env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for row in read(a.prepared/'cases/registry.json'):
        name=row['id'];public=a.output/name;private=a.private_output/name;public.mkdir();private.mkdir()
        run=a.original_private/name/'run';model=run/'agent';selection=read(model/'final_selection.json')
        frozen={str(f):sha(f) for f in [model/selection['model_file'],model/'task_spec.json',model/'final_selection.json',model/'classifier_fits.json']}
        reference=a.original_private/name/'reference/predictions.csv';query=a.prepared/'cases'/row['query']
        result=dict(id=name,status='failed')
        try:
            outputs=[]
            for label in ('first','second'):
                destination=private/label/'predictions.csv'
                command=[sys.executable,'-m','unified_agent','predict',str(query.resolve()),'--run-dir',str(run.resolve()),'--output',str(destination.resolve())]
                save(public/(label+'_invocation.json'),dict(command=command,offline=True,billing_configuration_removed=True))
                if execute(command,env,private/(label+'.log'),90):raise RuntimeError('Offline '+label+' prediction failed')
                outputs.append(destination)
            adapters=list((run/'delivery_recovery').glob('*/prediction_adapter.json'))
            if len(adapters)!=1:raise ValueError('Expected exactly one accepted adapter for this preserved model')
            spec=read(model/'task_spec.json');adapted=load_adapter(adapters[0],query,model,spec,selection)
            features=load_prediction_features(query,spec['model_kind'],adapted)
            oracle=read(a.prepared/'cases'/(name+'.oracle.json'))
            correct_features=array_fingerprint(features)==oracle['expected']['prediction_features_sha256']
            same_exports=len({sha(reference),*[sha(path) for path in outputs]})==1
            intact=frozen=={n:sha(n) for n in frozen}
            if not all((correct_features,same_exports,intact)):raise ValueError('Frozen files, features or exports differ')
            completion=read(adapters[0].parent/'completion.json')
            save(public/'prediction_adapter.json',read(adapters[0]));save(public/'recovery_completion.json',completion)
            save(public/'recovery_state.json',read(adapters[0].parent/'recovery_state.json'))
            save(public/'frozen_files.json',dict(before=frozen,after={n:sha(n) for n in frozen}))
            save(public/'prediction_manifest.json',read(outputs[0].parent/'prediction_manifest.json'))
            result.update(status='passed',new_classifier_fits=0,new_API_requests=0,feature_fingerprint_verified=True,
                all_exports_equal=True,frozen_files_unchanged=True,rows=len(features),
                decision_source='existing_real_LLM_adapter' if name=='D1' else completion.get('decision_source'))
        except Exception as issue:result['error']=dict(type=type(issue).__name__,message=str(issue))
        save(public/'completion.json',result);rows.append(result);print(json.dumps(result),flush=True)
    current=ledger_rows(ledger);unchanged=old==current
    result=dict(status='passed' if unchanged and all(r['status']=='passed' for r in rows) else 'failed',cases=rows,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,all_ledger_rows_unchanged=unchanged,
        ledger_before=before,ledger_after=ledger.project_summary(),source_unchanged=identity=={n:sha(root/n) for n in identity},
        elapsed_seconds=time.monotonic()-clock,scope='post-failure offline continuation, not a successful first SA22 paid batch')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
