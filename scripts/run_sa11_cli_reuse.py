"""Verify actual standalone solve processes with bound semantic artifacts; max four fits and no API."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_sa5_semantic_panel import read,save,sha
from tools.import_semantic_artifact import migrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--original-run',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--private-output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];start=time.monotonic();identity={p.relative_to(root).as_posix():sha(p) for p in (root/'superagent').glob('*.py')}
    identity['unified_agent.py']=sha(root/'unified_agent.py');ledger_before=sha(a.ledger)
    save(a.output/'protocol.json',dict(batch='SA11-real-CLI',source_sha256=identity,
        plan_sha256=sha(root/'docs/batches/SA11_semantic_artifact_plan_20261126_zh.md'),
        runner_sha256=sha(Path(__file__)),maximum_classifier_fits=4,fit_limit_per_path=2,maximum_real_API_requests=0,
        entry='standalone python -m unified_agent solve; no injected resolver, classifier, candidate or client'))
    records=[];error=None;fits=0
    try:
        registry=read(a.cases/'registry.json')
        for case in ['V1','V3']:
            row=next(r for r in registry if r['id']==case)
            if sha(a.cases/row['package'])!=row['package_sha256'] or sha(a.cases/row['query'])!=row['query_sha256']:
                raise ValueError('Original real task packages changed')
            artifact=a.private_output/(case+'_semantic_artifact.json')
            meta=migrate(a.cases/row['package'],a.cases/row['query'],a.original_run/case/'agent/semantic_contract.json',a.ledger,artifact)
            run=a.private_output/case;destination=run/'delivery/predictions.csv'
            env=os.environ.copy()
            for key in list(env):
                if key.startswith('SUPERAGENT_PLANNER_') or key.startswith('SUPERAGENT_RESEARCH_') or key=='SUPERAGENT_TASK_BUDGET_ID':env.pop(key)
            env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            command=[sys.executable,'-m','unified_agent','solve',str(a.cases/row['package']),
                '--prediction-input',str(a.cases/row['query']),'--run-dir',str(run),'--output',str(destination),
                '--semantic-artifact-from',str(artifact),'--task-analysis','local','--protected-self-review',
                '--fit-limit','2','--time-limit','120']
            clock=time.monotonic()
            proc=subprocess.run(command,cwd=root,env=env,stdin=subprocess.DEVNULL,capture_output=True,
                text=True,encoding='utf-8',errors='replace',timeout=180)
            (a.private_output/(case+'_solve.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if proc.returncode:raise RuntimeError('Official solve failed for '+case+'; inspect private log')
            accounting=read(run/'agent/classifier_fits.json');selection=read(run/'agent/final_selection.json')
            used=accounting['attempted_classifier_fits'];fits+=used
            if used>2 or fits>4 or not selection.get('refit'):raise ValueError('Expected a complete bounded refit')
            artifact_receipt=read(run/'agent/semantic_artifact_receipt.json')
            if artifact_receipt['new_interpretation_API_requests']!=0:raise ValueError('Cache unexpectedly requested a model')
            usage=[json.loads(s) for s in (run/'agent/llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
            if any(e.get('event') in ('llm_call','unresolved_call','missing_usage') for e in usage):raise ValueError('Unexpected real model usage')
            recipe=read(a.cases/(case+'.oracle.json'))['expected']['output']
            with destination.open(encoding=recipe['encoding'],newline='') as stream:rows=list(csv.DictReader(stream,delimiter=recipe['delimiter']))
            if len(rows)!=row['prediction_rows'] or list(rows[0])!=[c['name'] for c in recipe['columns']]:
                raise ValueError('Standalone export header/row mismatch')
            if [r[recipe['columns'][0]['name']] for r in rows]!=[f'{i:06d}' for i in range(len(rows))]:raise ValueError('Standalone original IDs/order changed')
            item=dict(case=case,status='passed',physical_fits=used,real_API_requests=0,family=selection['family'],
                prediction_rows=len(rows),artifact_sha256=meta['artifact_sha256'],
                original_request_id=artifact_receipt['provider_provenance']['original_request_id'],
                model_sha256=selection['model_sha256'],prediction_sha256=sha(destination),elapsed_seconds=time.monotonic()-clock)
            target=a.output/case;target.mkdir();save(target/'classifier_fits.json',accounting)
            save(target/'artifact_receipt.json',artifact_receipt);save(target/'final_selection.json',selection)
            records.append(item);print(json.dumps(item),flush=True)
    except Exception as e:error=dict(type=type(e).__name__,message=str(e)[:900])
    fits=sum(read(p)['attempted_classifier_fits'] for p in a.private_output.glob('*/agent/classifier_fits.json'))
    unchanged=all(sha(root/name)==digest for name,digest in identity.items())
    result=dict(status='passed' if len(records)==2 and error is None and unchanged and ledger_before==sha(a.ledger) else 'failed',
        error=error,cases=records,actual_classifier_fits=fits,real_API_requests=0,course_cost_rmb=0,
        ledger_unchanged=ledger_before==sha(a.ledger),source_unchanged=unchanged,elapsed_seconds=time.monotonic()-start,
        scope='real official CLI task-interpretation reuse; new model training, no search/model resume or quality comparison')
    save(a.output/'results.json',result);save(a.output/'completion.json',result)
    print(json.dumps(dict(status=result['status'],fits=fits,API=0,error=error),ensure_ascii=False));return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
