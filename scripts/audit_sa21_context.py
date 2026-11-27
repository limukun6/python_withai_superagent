"""Independent no-fit/no-API audit of recovery, receipts and autonomous provenance."""
import argparse
import json
from pathlib import Path
import time

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','live','private','ledger','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    from superagent.task_budget import TaskBudgetLedger
    ledger=TaskBudgetLedger(a.ledger);summary=read(a.live/'completion.json');registry=read(a.prepared/'cases/registry.json')
    save(a.output/'protocol.json',dict(batch='SA21-independent-audit',live_completion_sha256=sha(a.live/'completion.json'),
        registry_sha256=sha(a.prepared/'cases/registry.json'),actual_classifier_fits=0,real_API_requests=0))
    cases=[];known=0;inputs=outputs=0
    for row in registry:
        name=row['id'];out=a.live/name;agent=a.private/name/'run/agent';completion=read(out/'completion.json')
        events=[json.loads(s) for s in (agent/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        usage=read(out/'metered_usage.json');delta=read(out/'ledger_delta.json')
        byid={r['id']:r for r in delta};matches=[];receipts=[]
        for u in usage:
            with ledger._db() as db:r=db.execute('SELECT * FROM requests WHERE id=?',(u['budget_request_id'],)).fetchone()
            live=dict(r);resolution=json.loads(live['payload'])['resolution']
            checks=dict(canonical_row=live==byid[u['budget_request_id']],settled=live['status']=='known' and u['persistent_settlement_complete'],
                model=resolution['model']==u['served_model']=='DeepSeek-V4.1-Flash',
                tokens=(resolution['input_tokens'],resolution['output_tokens'])==(u['input_tokens'],u['output_tokens']),
                cost=live['course_units']==round(u['course_cost_rmb']*1e9))
            matches.append(all(checks.values()));receipts.append(dict(request_id=u['budget_request_id'],role=u['role'],checks=checks))
            known+=live['course_units'];inputs+=u['input_tokens'];outputs+=u['output_tokens']
        accepted=[];last=None;requests=[]
        for e in events:
            if e['kind']=='llm_request':
                packet=json.loads(e['messages'][-1]['content']);requests.append(dict(role=e['role'],packet=packet))
            if e['kind']=='llm_response':
                content=e['content'].strip()
                if content.startswith('```'):content='\n'.join(content.splitlines()[1:-1]).strip()
                last=json.loads(content)
            if e['kind']=='semantic_contract_accepted':accepted.append(last)
        state=read(agent/'recovery_state.json')
        checks=dict(receipts=all(matches) and len(matches)==len(delta),fits_bounded=completion['actual_classifier_fits']<=4,
            requests_bounded=len(usage)<=3,has_recovery_state=state['version']=='recovery-state-v1',
            no_oracle_in_prompt=all('oracle.json' not in json.dumps(r['packet']) for r in requests),
            shared_state_in_prompts=all('recovery_state' in r['packet'] for r in requests),
            source_bound=summary['source_unchanged'],history_preserved=summary['historical_rows_unchanged'])
        if completion.get('complete_delivery'):
            contract=read(agent/'semantic_contract.json')
            checks.update(real_accepted_reply=bool(accepted) and accepted[-1]==contract['proposal'],
                fresh_process_equal=completion['fresh_process_equal'],semantic_oracle=all(completion['semantic_checks'].values()),
                recorded_prior_failure=bool(state['failed_actions']),training_completed=state['status']=='training_completed')
        elif name=='R4':
            explicit=agent/'unresolved_task_report.json'
            if explicit.is_file():
                report=read(explicit)
                checks.update(real_unresolved_reply=report['proposal']==last,
                    target_question=[q['field'] for q in report['questions']]==['target'],
                    not_claimed_semantic_proof=not report['semantic_insufficiency_certified'])
            checks.update(no_false_completion=not completion.get('false_completion',False),
                unresolved=state['status']=='unresolved',no_classifier_fit=completion['actual_classifier_fits']==0)
        else:checks['complete_delivery']=False
        observations=state['observed_evidence']
        checks['physical_structure_tools_bounded']=len(observations)<=2
        from superagent.action_context import MAX_BYTES
        summary_records=[r for request in requests for r in request['packet']['recovery_state']['failed_actions'] if r.get('action_summary') is not None]
        checks['summary_byte_cap']=all(len(json.dumps(r['action_summary'],ensure_ascii=False).encode())<=MAX_BYTES for r in summary_records)
        checks['summary_not_executable']=all(r['action_summary'].get('not_executable') is True for r in summary_records)
        checks['summary_window']=all(sum(r.get('action_summary') is not None for r in request['packet']['recovery_state']['failed_actions'])<=2 for request in requests)
        checks['repair_instruction_present']=all('recovery_instruction' in r['packet'] for r in requests if r['role'].startswith('task_semantic'))
        contract_rejections=[e for e in events if e['kind']=='semantic_contract_rejected']
        tool_rejections=[e for e in events if e['kind']=='semantic_tool_rejected']

        checks['structure_values_not_disclosed']=all(o.get('row_values_disclosed',o.get('values_disclosed')) is False for o in observations)
        checks['no_premature_unresolved']=not completion.get('premature_unresolved',False)
        if completion.get('complete_delivery'):
            checks['total_budget_not_expanded']=completion['real_API_requests']<=3

        cases.append(dict(id=name,passed=all(checks.values()),checks=checks,receipts=receipts,
            actions=[dict(role=r['role'],action=r['packet'].get('diagnostic'),remaining=r['packet']['recovery_state']['remaining_requests']) for r in requests],
            structure_tools=[dict(member=o['member'],source=o['source'],format=o.get('format',o.get('codec'))) for o in observations],
            natural_contract_failures=len(contract_rejections),natural_tool_failures=len(tool_rejections),prompt_summary_records=len(summary_records),
            repeated_failures=sum(f['repeated_without_new_evidence'] for f in state['failed_actions'])))
    result=dict(status='passed' if all(r['passed'] for r in cases) else 'failed',cases=cases,
        audited_course_cost_rmb=known/1e9,input_tokens=inputs,output_tokens=outputs,
        actual_classifier_fits=0,real_API_requests=0,elapsed_seconds=time.monotonic()-start)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
