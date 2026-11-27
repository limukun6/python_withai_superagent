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
    save(a.output/'protocol.json',dict(batch='SA19-independent-audit',live_completion_sha256=sha(a.live/'completion.json'),
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
            checks.update(no_false_completion=not completion.get('false_completion',False),
                unresolved=state['status']=='unresolved',no_classifier_fit=completion['actual_classifier_fits']==0)
        else:checks['complete_delivery']=False
        observations=state['observed_evidence']
        cases.append(dict(id=name,passed=all(checks.values()),checks=checks,receipts=receipts,
            actions=[dict(role=r['role'],action=r['packet'].get('diagnostic'),remaining=r['packet']['recovery_state']['remaining_requests']) for r in requests],
            structure_tools=[dict(member=o['member'],source=o['source'],format=o.get('format',o.get('codec'))) for o in observations],
            repeated_failures=sum(f['repeated_without_new_evidence'] for f in state['failed_actions'])))
    result=dict(status='passed' if all(r['passed'] for r in cases) else 'failed',cases=cases,
        audited_course_cost_rmb=known/1e9,input_tokens=inputs,output_tokens=outputs,
        actual_classifier_fits=0,real_API_requests=0,elapsed_seconds=time.monotonic()-start)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
