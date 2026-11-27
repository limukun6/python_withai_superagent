"""Seal the exact SA15 optional-positive-class rejection, no API or fitting."""
import argparse
import json
from pathlib import Path
import socket
from unittest.mock import patch

from scripts.run_sa5_semantic_panel import read,save,sha
from superagent.format_sources import package_files
from superagent.native_semantics import compile_native
from superagent.semantic_contract import evidence_workspace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    trace=root/'research/sa15_default_live_01/semantic_trace.json';captured=next(r['proposal'] for r in read(trace) if r['kind']=='semantic_contract_rejected')
    accepted=read(root/'research/sa15_default_live_01/semantic_contract.json')['proposal']
    save(a.output/'protocol.json',dict(batch='SA17-old-question-baseline',base='1ecc4c9',captured_trace_sha256=sha(trace),
        native_reader_sha256=sha(root/'superagent/native_semantics.py'),maximum_classifier_fits=0,maximum_real_API_requests=0))
    rejected=False;error=None
    with patch.object(socket,'create_connection',side_effect=AssertionError('No API')):
        with package_files(root/'runs/sa15_cases/D1.zip') as files,package_files(root/'runs/sa15_cases/D1.query.zip') as query:
            try:compile_native(files,evidence_workspace(files),captured,prediction_files=query)
            except ValueError as exc:rejected=True;error=dict(type=type(exc).__name__,message=str(exc))
    usage=read(root/'research/sa15_default_live_01/metered_usage.json');repair=next(r for r in usage if r['role']=='task_semantic_repair')
    result=dict(status='baseline_sealed' if rejected and captured['plan']==accepted['plan'] else 'failed',
        original_reply_rejected=rejected,original_plan_equals_later_accepted_plan=captured['plan']==accepted['plan'],error=error,
        original_question=captured['uncertainties'],associated_historical_repair_course_cost_rmb=repair['course_cost_rmb'],
        associated_historical_repair_seconds=repair['latency_seconds'],actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        scope='old reply replay only; not a new response, automatic annotation or already-realized cost saving')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
