"""Revalidate two real SA14 replies, then reload their existing models; zero new fits/API."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa14_native_delivery import grade,validate_registry
from superagent.semantic_artifact import load_artifact
from tools.import_semantic_artifact import migrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cases','original-run','ledger','output','private-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];before=sha(a.ledger);clock=time.monotonic()
    save(a.output/'protocol.json',dict(batch='SA15-explicit-revalidation',base='e3aaee4',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(root/'docs/batches/SA15_default_handoff_plan_20261127_zh.md'),maximum_classifier_fits=0,maximum_real_API_requests=0,
        ledger_before_sha256=before,prediction_answers_opened=False))
    rows=[]
    for row in validate_registry(a.cases):
        case=row['id'];original=a.original_run/case;target=a.private_output/(case+'_artifact.json')
        old_artifact=original/'agent/semantic_artifact.json';old_hash=sha(old_artifact)
        try:
            load_artifact(old_artifact,a.cases/row['package'],a.cases/row['query']);rejected=False
        except ValueError as error:
            rejected='reader source changed' in str(error)
        if not rejected:raise ValueError('Expected explicit migration to be required after reader change')
        metadata=migrate(a.cases/row['package'],a.cases/row['query'],original/'agent/semantic_contract.json',a.ledger,target)
        restored=load_artifact(target,a.cases/row['package'],a.cases/row['query'])
        verification=grade(restored,a.cases/row['query'],read(a.cases/(case+'.oracle.json'))['expected'])
        env=os.environ.copy()
        for key in list(env):
            if key.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_')):env.pop(key)
        env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        destination=a.private_output/case/'predictions.csv'
        proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str((a.cases/row['query']).resolve()),
            '--run-dir',str(original.resolve()),'--output',str(destination.resolve())],cwd=root,env=env,stdin=subprocess.DEVNULL,
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        (a.private_output/(case+'_predict.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        identical=proc.returncode==0 and sha(destination)==sha(original/'delivery/predictions.csv')
        rows.append(dict(case=case,status='passed' if verification['complete'] and identical and old_hash==sha(old_artifact) else 'failed',
            stale_artifact_refused=True,new_explicit_artifact_sha256=metadata['artifact_sha256'],
            provider_provenance=metadata['provider_provenance'],semantic_grade=verification,fresh_prediction_identical=identical,
            original_artifact_unchanged=old_hash==sha(old_artifact)))
    result=dict(status='passed' if all(r['status']=='passed' for r in rows) and before==sha(a.ledger) else 'failed',cases=rows,
                ledger_unchanged=before==sha(a.ledger),actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
                elapsed_seconds=time.monotonic()-clock,scope='old real replies explicitly revalidated; not new LLM samples or model/search resume')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False));return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
