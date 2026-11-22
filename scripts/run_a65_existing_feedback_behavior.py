"""Read-only paired behavior/quality table for archived A55 proposals; no LLM judging."""
import argparse
import json
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    plan=Path(__file__).resolve().parents[1]/'docs/batches/A65_existing_feedback_behavior_plan.md'
    save(root/'protocol.json',dict(batch='A65',plan_sha256=sha(plan),runner_sha256=sha(Path(__file__)),
        answers_sha256=sha(a.answers/'answers.json'),comparison_sha256=sha(a.evaluation/'comparison.json'),
        posthoc_descriptive_analysis=True,actual_classifier_fits=0,real_api_requests=0,formal_test_or_H_opened=False))
    try:
        comparisons={r['state_id']:r for r in read(a.evaluation/'comparison.json')}
        for answer in read(a.answers/'answers.json'):
            state_path=Path(answer['state_file'])
            if sha(state_path)!=answer['state_sha256']:raise ValueError('Archived state changed')
            packet=read(state_path);state=packet['state'];caps=packet['capabilities']
            if packet['R']!={k:v for k,v in packet['E'].items() if k not in ('checks','memory')}:
                raise ValueError('Common R/E context differs')
            history={r['trial']:r for r in state['history']}
            failed=[t for t,r in history.items() if r['status']!='completed']
            contradicted=[t for t,r in history.items() if (r.get('configuration_fact_check') or {}).get('status')=='contradicted']
            unsupported=[t for t,r in history.items() if (r.get('hypothesis_check') or {}).get('status')=='not_supported']
            behavior={}
            for mode in ('R','E'):
                proposed=answer[mode]['proposal'];raw=proposed['raw_proposal'];name=answer[mode]['candidate']
                refs=raw['evidence_trials']
                checks=dict(eligible_candidate=name in packet['eligible_ids'],
                    registered_configuration=proposed['params']==caps[name]['params'],
                    registered_classifier=raw['configuration_claim']['classifier']==caps[name]['estimator'],
                    completed_references=all(t in history and history[t]['status']=='completed' for t in refs))
                if not all(checks.values()):raise ValueError('Archived proposal violates its mechanical contract')
                behavior[mode]=dict(candidate=name,checks=checks,evidence_trials=refs,
                    expected_change_is_null=raw['expected_change'] is None,purpose=raw['purpose'],
                    referenced_failed_trials=sorted(set(refs)&set(failed)),
                    referenced_configuration_contradictions=sorted(set(refs)&set(contradicted)),
                    referenced_unsupported_hypotheses=sorted(set(refs)&set(unsupported)))
            comp=comparisons[answer['state_id']]
            if any(comp[m]['candidate']!=behavior[m]['candidate'] for m in ('R','E')):raise ValueError('Quality table refers to another action')
            rows.append(dict(state_id=answer['state_id'],task=answer['task'],common_context_identical=True,
                observed_history_trials=len(history),failed_trials=failed,configuration_contradictions=contradicted,
                unsupported_hypotheses=unsupported,behavior=behavior,quality=comp,
                candidate_changed=behavior['R']['candidate']!=behavior['E']['candidate'],
                free_text_semantic_correctness='not independently labeled',
                successful_correction_rate=None))
        save(root/'results.json',rows)
        tasks={name:dict(states=sum(r['task']==name for r in rows),
            changed_candidates=sum(r['candidate_changed'] for r in rows if r['task']==name),
            mean_action_E_minus_R=sum(r['quality']['E_minus_R'] for r in rows if r['task']==name)/sum(r['task']==name for r in rows))
            for name in sorted({r['task'] for r in rows})}
        summary=dict(status='completed',paired_states=len(rows),archived_answers=2*len(rows),
            mechanically_valid_answers=2*len(rows),null_numeric_expectations=sum(r['behavior'][m]['expected_change_is_null'] for r in rows for m in ('R','E')),
            states_with_failed_trial=sum(bool(r['failed_trials']) for r in rows),
            states_with_configuration_contradiction=sum(bool(r['configuration_contradictions']) for r in rows),
            states_with_unsupported_hypothesis=sum(bool(r['unsupported_hypotheses']) for r in rows),
            tasks=tasks,correction_rate=None,correction_rate_reason='No independent semantic correction labels; opportunities counted separately',
            actual_classifier_fits=0,real_api_requests=0,elapsed_seconds=time.monotonic()-start)
        save(root/'completion.json',summary);print(json.dumps(summary))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_api_requests=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('answers','evaluation','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
