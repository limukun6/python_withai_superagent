"""Analyze sealed fixed-state decisions, including partial failures, without fits or API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run','inputs','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    save(out/'protocol.json',dict(batch='A146-evaluate',source_files=source,runner_sha256=sha(Path(__file__)),
        run_completion_sha256=sha(a.run/'completion.json'),maximum_fits=0,maximum_API_requests=0,
        scope='Typed decision-basis checks and bounded prose diagnostics; no independent human semantic gold'))
    try:
        from scripts import a38_predict_guard
        from superagent.ab_flash import validate_proposal
        from superagent.grounded_proposal import assess,GroundingError
        completion=read(a.run/'completion.json');receipts=read(a.run/'results.json')['answers'];records=[]
        if read(a.run/'protocol.json')['source_files']!=source:raise ValueError('Evaluation source differs from sealed producer')
        for receipt in receipts:
            task,condition,mode=(receipt[k] for k in ('task','condition','mode'));name=condition+'_'+mode
            folder=a.run/task/name;inputs=a.inputs/task;answer=read(folder/'answer.json')
            if sha(folder/'answer.json')!=receipt['answer_sha256'] or sha(inputs/(name+'.json'))!=receipt['context_sha256']:
                raise ValueError('Sealed answer/context bytes changed')
            state=read(inputs/(condition+'_state.json'));caps=read(inputs/'capabilities.json');ids=[c['id'] for c in state['available_candidates']]
            report=None;error=None
            try:
                if mode=='grounded':proposal,report=assess(answer,state,caps,ids)
                else:proposal=validate_proposal(answer,state,caps,ids,'calibrated-v2')
                valid=True
            except (ValueError,TypeError,KeyError) as e:
                valid=False;error=str(e)
                if isinstance(e,GroundingError):report=e.report
            audit=read(folder/'prose_audit.json') if (folder/'prose_audit.json').exists() else {}
            usage=receipt['usage'];prose=audit.get('prose_claims',{});issues=report.get('issues',[]) if report else []
            record=dict(task=task,condition=condition,mode=mode,valid_action=valid,error=error,candidate=answer.get('candidate'),
                purpose=answer.get('purpose'),input_tokens=usage['input_tokens'],output_tokens=usage['output_tokens'],
                cost_rmb=usage['course_cost_rmb'],latency_seconds=usage['latency_seconds'],response_id=usage.get('response_id'),
                assertion_status_counts=dict(Counter(c['status'] for c in report.get('assertion_checks',[]))) if report else None,
                issue_status_counts=dict(Counter(i['status'] for i in issues)) if report else None,
                controlled_issue_proposal_corrected=(condition=='controlled' and mode=='grounded' and valid
                    and bool(issues) and all(i['status']=='addressed_pending_check' for i in issues)),
                execution_closure_tested=False,freeform_prose_certified=False,
                prose_contradicted=prose.get('contradicted',0),prose_unverifiable=prose.get('unverifiable',0),
                parameter_status_counts=dict(Counter(i['status'] for i in audit.get('parameter_semantics',{}).get('items',[]))))
            records.append(record)
        aggregate={}
        for mode in ('legacy','grounded'):
            rows=[r for r in records if r['mode']==mode]
            aggregate[mode]=dict(calls=len(rows),valid_actions=sum(r['valid_action'] for r in rows),
                input_tokens=sum(r['input_tokens'] for r in rows),output_tokens=sum(r['output_tokens'] for r in rows),
                cost_rmb=sum(r['cost_rmb'] for r in rows),latency_seconds=sum(r['latency_seconds'] for r in rows),
                typed_assertions=dict(sum((Counter(r['assertion_status_counts'] or {}) for r in rows),Counter())),
                prose_contradicted=sum(r['prose_contradicted'] for r in rows),prose_unverifiable=sum(r['prose_unverifiable'] for r in rows))
        pairs=[]
        for task in ('coupon','sms','flowers'):
            for condition in ('natural','controlled'):
                rows={r['mode']:r for r in records if r['task']==task and r['condition']==condition}
                complete=set(rows)=={'legacy','grounded'} and all(r['valid_action'] for r in rows.values())
                pairs.append(dict(task=task,condition=condition,complete=complete,
                    candidates={m:r['candidate'] for m,r in rows.items()},
                    candidate_changed=rows['legacy']['candidate']!=rows['grounded']['candidate'] if complete else None))
        natural_different=[p for p in pairs if p['condition']=='natural' and p['candidate_changed']]
        result=dict(records=records,by_mode=aggregate,pairs=pairs,complete_paired_trial=completion['status']=='answers_sealed' and len(records)==12,
            next_action_quality_measured=False,independent_human_semantic_labels=False,actual_execution_closure=False,
            downstream_gate_candidates=natural_different,paid_calls_referenced=completion['real_API_requests'],
            cost_is_course_conversion_not_platform_debit=True,
            scope='Three seen task prefixes, one reply per cell; injected-error repair is not a natural error rate')
        if source!=hashes(ROOT):raise ValueError('Source changed during evaluation')
        save(out/'results.json',result);save(out/'completion.json',dict(status='evaluated',actual_classifier_fits=0,real_API_requests=0,
            paid_calls_analyzed=len(records),source_unchanged=True,elapsed_seconds=time.monotonic()-start,
            results_sha256=sha(out/'results.json'),protocol_sha256=sha(out/'protocol.json')))
        print(json.dumps(dict(by_mode=aggregate,pairs=pairs,complete_paired_trial=result['complete_paired_trial'])))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
