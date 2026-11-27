"""Independent no-fit/no-API audit of SA13's sealed live delivery and tool continuation."""
import argparse
import json
from pathlib import Path
import socket
from unittest.mock import patch

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(batch='SA13-sealed-live-audit',runner_sha256=sha(Path(__file__)),
        completion_sha256=sha(a.live/'completion.json'),maximum_classifier_fits=0,maximum_real_API_requests=0,
        scope='original sealed outputs, no development labels or new model generation'))
    from superagent.semantic_artifact import load_artifact
    from superagent.ab_run import Executor
    from tools.submission_equivalence import grade_with_csv_semantics
    with patch.object(Executor,'worker',side_effect=AssertionError('No fitting')),patch.object(socket,'create_connection',side_effect=AssertionError('No API')):
        artifact=a.run/'agent/semantic_artifact.json'
        task=load_artifact(artifact,a.cases/'V1.zip',a.cases/'V1.query.zip')
        grade=grade_with_csv_semantics(task,a.cases/'V1.query.zip',read(a.cases/'V1.oracle.json'))
    events=[json.loads(s) for s in (a.run/'agent/events.jsonl').read_text(encoding='utf-8').splitlines()]
    observations=[(i,r) for i,r in enumerate(events) if r['kind']=='semantic_structure_observed']
    continuation=[(i,r) for i,r in enumerate(events) if r['kind']=='llm_request' and r.get('role')=='task_semantic_continuation']
    responses=[(i,r) for i,r in enumerate(events) if r['kind']=='llm_response']
    proposal=read(a.live/'semantic_contract.json')['proposal']
    idfields=[c['field'] for c in proposal['output']['columns'] if c['kind']=='input_id']
    used_fields=bool(observations) and all(f in observations[-1][1]['observation']['fields'] for f in idfields)
    continuation_receives_observation=bool(continuation) and any(
        m['role']=='user' and json.loads(m['content']).get('tool_observations')
        for m in continuation[0][1].get('messages',[]))
    usage=read(a.live/'metered_usage.json');delta=read(a.live/'ledger_delta.json')
    logs_linked=len(usage)==len(delta)==2 and {r['budget_request_id'] for r in usage}=={r['id'] for r in delta}
    for row in delta:
        receipt=next(x for x in usage if x['budget_request_id']==row['id'])
        if row['status']!='known' or row['course_units']!=round(receipt['course_cost_rmb']*1e9):logs_linked=False
    result=dict(status='passed' if grade['complete'] and used_fields and continuation_receives_observation and logs_linked else 'failed',
        semantic_grade=grade,natural_tool_calls=len(observations),observed_fields_used_in_accepted_ID_contract=used_fields,
        real_followup_prompt_contains_tool_observation=bool(continuation_receives_observation),provider_usage_linked_to_real_ledger=logs_linked,
        real_live_API_requests=len(usage),real_live_input_tokens=sum(r['input_tokens'] for r in usage),
        real_live_output_tokens=sum(r['output_tokens'] for r in usage),real_live_course_cost_rmb=sum(r['course_cost_rmb'] for r in usage),
        actual_classifier_fits=0,new_API_requests=0,new_course_cost_rmb=0,quality_evaluation=False,
        boundary='one previously used development task; observational repair evidence, not a randomized ablation or independent transfer')
    save(a.output/'results.json',result);save(a.output/'completion.json',result)
    print(json.dumps(result,ensure_ascii=False));return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
