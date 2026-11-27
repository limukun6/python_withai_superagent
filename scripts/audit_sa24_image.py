"""Independent SA24 receipts, chosen pixels, immutable delivery and provenance audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.run_sa5_semantic_panel import read,save,sha
from scripts.run_sa16_auto_live import ledger_rows
from scripts.run_sa21_context_live import clean_env
from scripts.run_sa24_image_live import identity,harness_identity,ROOT,PLAN
from scripts.sa24_image_cases import grade


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','live','private','ledger','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from superagent.task_budget import TaskBudgetLedger
    ledger=TaskBudgetLedger(a.ledger);summary=read(a.live/'completion.json');protocol=read(a.live/'protocol.json')
    old=read(a.live/'historical_rows_before.json');current=ledger_rows(ledger)
    historical=all(current.get(k)==v for k,v in old.items())
    save(a.output/'protocol.json',dict(batch='SA24-independent-audit',script_sha256=sha(Path(__file__)),
        live_completion_sha256=sha(a.live/'completion.json'),actual_classifier_fits=0,real_API_requests=0))
    registry={r['id']:r for r in read(a.prepared/'cases/registry.json')};rows=[];inputs=outputs=units=0
    for result in summary['cases']:
        name=result['id'];public=a.live/name;private=a.private/name;model=private/'run/agent';row=registry[result['case']]
        cwd=Path(protocol['baseline_root']) if result['version']=='old' else ROOT
        completion=read(public/'completion.json');calls=read(public/'metered_usage.json');delta=read(public/'ledger_delta.json')
        byid={r['id']:r for r in delta};receipts=[]
        for call in calls:
            with ledger._db() as db:canonical=dict(db.execute('SELECT * FROM requests WHERE id=?',(call['budget_request_id'],)).fetchone())
            resolution=json.loads(canonical['payload'])['resolution']
            checks=dict(row_matches=canonical==byid[call['budget_request_id']],known=canonical['status']=='known',
                model=resolution['model']==call['served_model']=='DeepSeek-V4.1-Flash',
                tokens=(resolution['input_tokens'],resolution['output_tokens'])==(call['input_tokens'],call['output_tokens']),
                cost=canonical['course_units']==round(call['course_cost_rmb']*1e9),settled=call['persistent_settlement_complete'])
            receipts.append(dict(id=call['budget_request_id'],checks=checks));inputs+=call['input_tokens'];outputs+=call['output_tokens'];units+=canonical['course_units']
        checks=dict(all_receipts=len(receipts)==len(delta) and all(all(r['checks'].values()) for r in receipts),
            same_scope=all(r['scope']==read(public/'invocation.json')['scope'] for r in delta),
            resources=len(delta)<=3 and completion['actual_classifier_fits']<=2,historical_rows=historical,
            frozen_source=protocol['source_sha256']==identity(ROOT) and protocol['baseline_source_sha256']==identity(Path(protocol['baseline_root'])) and protocol['harness_sha256']==harness_identity() and protocol['plan_sha256']==sha(PLAN))
        details={}
        if completion['complete_delivery']:
            proposal=read(public/'semantic_contract.json');spec=read(model/'task_spec.json');oracle=read(a.prepared/'cases'/(row['id']+'.oracle.json'))
            package=(a.prepared/'cases'/row['package']).resolve();query=(a.prepared/'cases'/row['query']).resolve()
            independent=grade(package,query,proposal,spec,oracle)
            accepted=read(model/'semantic_contract.json')['proposal'];replies=[]
            for event in read(public/'semantic_events.json'):
                if event['kind']!='llm_response':continue
                content=event['content'].strip()
                if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1])
                try:replies.append(json.loads(content))
                except json.JSONDecodeError:pass
            protected=read(public/'frozen_files.json')
            replay=private/'replay/predictions.csv';delivery=private/'delivery/predictions.csv'
            code=('import sys,json;from superagent.semantic_artifact import load_artifact;'
                't=load_artifact(sys.argv[1],sys.argv[2],sys.argv[3]);'
                'print(json.dumps(dict(kind=t.kind,rows=len(t.y_train),decisions=t.spec.get("design_decisions"))))')
            proc=subprocess.run([sys.executable,'-c',code,str((model/'semantic_artifact.json').resolve()),str(package),str(query)],
                cwd=cwd,env=clean_env(),stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=60)
            (a.output/(name+'.artifact_reload.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            checks.update(independent_semantic_pixels=independent['complete'],equal_exports=sha(replay)==sha(delivery)==completion['prediction_sha256'],
                frozen_files=protected=={n:sha(n) for n in protected},real_reply=accepted==proposal and proposal in replies,
                artifact_reload=proc.returncode==0)
            if result['version']=='new':checks['new_design_provenance']=independent['provenance_correct']
            details=dict(chosen=independent['chosen'],provenance_checks=independent['provenance_checks'],
                proposal_version=proposal['version'],semantic_checks=independent['checks'])
        else:
            # A genuine failed path is preserved, never turned into task completion by this audit.
            details['failure']=completion.get('error')
        rows.append(dict(id=name,passed=all(checks.values()),task_completed=completion['complete_delivery'],checks=checks,receipts=receipts,details=details))
    result=dict(status='passed' if len(rows)==4 and all(r['passed'] for r in rows) else 'failed',cases=rows,
        task_completions=sum(r['task_completed'] for r in rows),input_tokens=inputs,output_tokens=outputs,
        audited_course_cost_rmb=units/1e9,actual_classifier_fits=0,real_API_requests=0,
        historical_rows_unchanged=historical,ledger_now=ledger.project_summary(),
        boundary='independent code path recheck by same A developer; not human semantic annotation or quality evaluation')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
