"""Evaluate sealed proposals using bounded diagnostics; no claims of human semantic gold."""
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
    p.add_argument('--run',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();source=hashes(ROOT)
    completion=read(a.run/'completion.json')
    save(out/'protocol.json',dict(batch='A140-evaluate',source_files=source,runner_sha256=sha(Path(__file__)),
         sealed_run_completion_sha256=sha(a.run/'completion.json'),fit_cap=0,API_cap=0,
         evaluator_scope='Existing bounded prose/C diagnostics; not independent human labels or complete error rate'))
    try:
        from scripts import a38_predict_guard
        from superagent.ab_flash import validate_proposal
        from superagent.explanation_audit import audit_explanation
        records=[];sealed=read(a.run/'answers_sealed.json')
        for receipt in sealed:
            task,view=receipt['task'],receipt['view'];folder=a.run/task/view;answer=read(folder/'answer.json')
            if sha(folder/'answer.json')!=receipt['answer_sha256'] or sha(a.inputs/task/(view+'.json'))!=receipt['context_sha256']:
                raise ValueError('Sealed response/context changed')
            state=read(a.inputs/task/'state.json');caps=read(a.inputs/task/'capabilities.json')
            ids=[c['id'] for c in state['available_candidates']]
            try:proposal=validate_proposal(answer,state,caps,ids,'calibrated-v2');valid=True
            except ValueError:proposal=None;valid=False
            if valid:
                candidate=dict(id=proposal.candidate,params=caps[proposal.candidate]['params'])
                report=audit_explanation(answer,state['history'],candidate,parameter_semantics=True,
                     selected_facts=dict(classifier=caps[proposal.candidate]['estimator'],source='shared_ab_catalogue'))
                save(out/(task+'_'+view+'_diagnostics.json'),report)
                prose=report['prose_claims'];parameters=report.get('parameter_semantics',{}).get('items',[])
                statuses=dict(Counter(i['status'] for i in parameters))
            else:prose={};statuses={};report={}
            usage=receipt['usage']
            records.append(dict(task=task,view=view,valid_action=valid,candidate=answer.get('candidate'),
                purpose=answer.get('purpose'),input_tokens=usage['input_tokens'],output_tokens=usage['output_tokens'],
                cost_rmb=usage['course_cost_rmb'],latency_seconds=usage['latency_seconds'],response_id=usage.get('response_id'),
                prose_items=prose.get('claims_checked',0),prose_contradicted=prose.get('contradicted',0),
                prose_unverifiable=prose.get('unverifiable',0),parameter_status_counts=statuses,
                warning_codes=dict(Counter(w['code'] for w in report.get('warnings',[]))),
                prose_fully_verified=False,action_quality_measured=False))
        aggregate={}
        for view in ('raw','full','compact'):
            rows=[r for r in records if r['view']==view]
            aggregate[view]=dict(calls=len(rows),valid_actions=sum(r['valid_action'] for r in rows),
                 input_tokens=sum(r['input_tokens'] for r in rows),output_tokens=sum(r['output_tokens'] for r in rows),
                 course_cost_rmb=sum(r['cost_rmb'] for r in rows),latency_seconds=sum(r['latency_seconds'] for r in rows),
                 prose_contradicted=sum(r['prose_contradicted'] for r in rows),
                 prose_items=sum(r['prose_items'] for r in rows),prose_unverifiable=sum(r['prose_unverifiable'] for r in rows),
                 parameter_status_counts=dict(sum((Counter(r['parameter_status_counts']) for r in rows),Counter())))
        full,compact=aggregate['full'],aggregate['compact'];paired=len(records)==9 and completion['status']=='answers_sealed'
        comparisons={task:{r['view']:r['candidate'] for r in records if r['task']==task} for task in ('coupon','sms','flowers')}
        response_ids=[r['response_id'] for r in records if r['response_id']]
        summary=dict(records=records,by_view=aggregate,candidate_by_task=comparisons,complete_paired_trial=paired,
             repeated_response_ids={k:v for k,v in Counter(response_ids).items() if v>1},
             compact_vs_full_input_token_delta=compact['input_tokens']-full['input_tokens'] if paired else None,
             compact_vs_full_total_course_cost_delta_rmb=compact['course_cost_rmb']-full['course_cost_rmb'] if paired else None,
             semantic_gold_labels_available=False,downstream_quality_measured=False,
             independent_tasks=3,replicates_per_state_view=1,
             scope='Three seen task states, single responses; bounded diagnostic flags do not establish full semantic accuracy')
        save(out/'results.json',summary)
        save(out/'completion.json',dict(status='evaluated',actual_classifier_fits=0,real_API_requests=0,
             paid_calls_analyzed=len(records),source_unchanged=source==hashes(ROOT),elapsed_seconds=time.monotonic()-started,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps({k:summary[k] for k in ('by_view','candidate_by_task','complete_paired_trial',
              'compact_vs_full_input_token_delta','compact_vs_full_total_course_cost_delta_rmb')}),flush=True)
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
