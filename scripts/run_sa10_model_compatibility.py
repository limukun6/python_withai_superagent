"""Check the three frozen real-data models against the stricter CSV mapping guard, without fitting or API."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--recovery-run',type=Path,required=True)
    p.add_argument('--original-run',type=Path,required=True);p.add_argument('--recovery-evidence',type=Path,required=True)
    p.add_argument('--ledger',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--private-output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];started=time.monotonic();source={r.relative_to(root).as_posix():sha(r) for r in (root/'superagent').glob('*.py')}
    save(a.output/'protocol.json',dict(batch='SA10-compatibility',production_source_sha256=source,
        script_sha256=sha(Path(__file__)),plan_sha256=sha(root/'docs/batches/SA10_csv_class_guard_plan_20261126_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0))
    ledger_before=sha(a.ledger);records=[];error=None
    try:
        registry=read(a.cases/'registry.json');cases=read(a.recovery_evidence/'results.json')['cases']
        for case in cases:
            row=next(r for r in registry if r['id']==case['case'])
            run=a.original_run/row['id'] if row['id']=='V2' else a.recovery_run/row['id']
            selected=read(run/'agent/final_selection.json');model=run/'agent'/selected['model_file'];spec=run/'agent/task_spec.json'
            before=dict(model=sha(model),spec=sha(spec));folder=a.private_output/row['id'];folder.mkdir()
            dest=folder/'predictions.csv';env=os.environ.copy()
            for key in list(env):
                if key.startswith('SUPERAGENT_PLANNER_') or key.startswith('SUPERAGENT_RESEARCH_'):env.pop(key)
            proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str(a.cases/row['query']),
                '--run-dir',str(run),'--output',str(dest)],cwd=root,env=env,capture_output=True,
                text=True,encoding='utf-8',errors='replace',timeout=90)
            (folder/'predict.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if proc.returncode:raise RuntimeError('Prediction failed for '+row['id'])
            unchanged=before==dict(model=sha(model),spec=sha(spec))
            if not unchanged or sha(dest)!=case['prediction_sha256']:raise ValueError('Frozen model/spec or prediction changed')
            manifest=read(folder/'prediction_manifest.json')
            records.append(dict(case=row['id'],status='identical',model_sha256=before['model'],task_spec_sha256=before['spec'],
                prediction_sha256=sha(dest),prediction_rows=manifest['prediction_count'],model_and_spec_unchanged=unchanged))
    except Exception as e:error=dict(type=type(e).__name__,message=str(e)[:800])
    source_unchanged=all(sha(root/name)==value for name,value in source.items())
    receipt=dict(status='passed' if len(records)==3 and error is None and source_unchanged and sha(a.ledger)==ledger_before else 'failed',
        error=error,cases=records,source_unchanged=source_unchanged,ledger_unchanged=sha(a.ledger)==ledger_before,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,elapsed_seconds=time.monotonic()-started,
        scope='prediction compatibility only; models were trained before the CSV validation patch')
    save(a.output/'results.json',receipt);save(a.output/'completion.json',receipt)
    print(json.dumps(dict(status=receipt['status'],models=len(records),fits=0,API=0,error=error),ensure_ascii=False))
    return receipt['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
