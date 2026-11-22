"""Posthoc mechanical/prose flags and paired action quality; never self-certifies free text."""
import argparse
from pathlib import Path
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.prose_claims import check_prose


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,help='Separate completed recovery evaluation; original failed directory remains intact')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    evaluation=a.evaluation or a.study/'evaluation'
    if read(evaluation/'completion.json')['status']!='completed':raise ValueError('All answers and action measurements must finish')
    answers=read(a.study/'answers/answers.json');quality={r['state_id']:r for r in read(evaluation/'comparison.json')};rows=[]
    save(a.output/'protocol.json',dict(batch='A70-posthoc-behavior',answers_sha256=sha(a.study/'answers/answers.json'),
        comparison_sha256=sha(evaluation/'comparison.json'),runner_sha256=sha(Path(__file__)),actual_classifier_fits=0,real_api_requests=0,
        original_failed_evaluation_retained=a.evaluation is not None))
    for item in answers:
        packet=read(Path(item['state_file']))
        if sha(Path(item['state_file']))!=item['state_sha256']:raise ValueError('Prepared state changed')
        state=packet['state'];caps=packet['capabilities'];history=state['history'];by_trial={r['trial']:r for r in history};modes={}
        for mode in ('R','E'):
            proposal=item[mode]['proposal'];raw=proposal['raw_proposal'];name=item[mode]['candidate']
            checks=dict(eligible=name in packet['eligible_ids'],configuration=proposal['params']==caps[name]['params'],
                classifier=raw['configuration_claim']['classifier']==caps[name]['estimator'],
                completed_trial_references=all(t in by_trial and by_trial[t]['status']=='completed' for t in raw['evidence_trials']))
            if not all(checks.values()):raise ValueError('Invalid archived proposal identity')
            flags=check_prose({k:raw[k] for k in ('reason','review','hypothesis')},{'classifier':caps[name]['estimator']},history,state['task']['kind'])
            modes[mode]=dict(candidate=name,checks=checks,numeric_expectation_is_null=raw['expected_change'] is None,
                prose_machine_flags=flags,free_text_semantic_correctness='not independently labeled')
        q=quality[item['state_id']]
        if any(q[m]['candidate']!=modes[m]['candidate'] for m in ('R','E')):raise ValueError('Quality action identity differs')
        opportunities=dict(failed_trials=[r['trial'] for r in history if r['status']!='completed'],
            configuration_contradictions=[r['trial'] for r in history if (r.get('configuration_fact_check') or {}).get('status')=='contradicted'],
            unsupported_hypotheses=[r['trial'] for r in history if (r.get('hypothesis_check') or {}).get('status')=='not_supported'])
        rows.append(dict(state_id=item['state_id'],task=item['task'],modes=modes,quality=q,recorded_opportunities=opportunities,
            changed_candidate=modes['R']['candidate']!=modes['E']['candidate']))
    save(a.output/'results.json',rows)
    tasks={}
    for name in sorted({r['task'] for r in rows}):
        group=[r for r in rows if r['task']==name];deltas=[r['quality']['E_minus_R'] for r in group]
        tasks[name]=dict(states=len(group),changed_candidates=sum(r['changed_candidate'] for r in group),mean_E_minus_R=sum(deltas)/len(deltas),
            E_better_states=sum(d>1e-12 for d in deltas),tied_states=sum(abs(d)<=1e-12 for d in deltas),E_worse_states=sum(d<-1e-12 for d in deltas))
    save(a.output/'completion.json',dict(status='completed',paired_states=len(rows),answers=2*len(rows),tasks=tasks,
        mechanically_valid_answers=2*len(rows),numeric_expectation_null=sum(r['modes'][m]['numeric_expectation_is_null'] for r in rows for m in ('R','E')),
        states_with_recorded_error_opportunity=sum(any(r['recorded_opportunities'].values()) for r in rows),
        machine_prose_contradiction_flags=sum(r['modes'][m]['prose_machine_flags']['contradicted'] for r in rows for m in ('R','E')),
        successful_correction_rate=None,independently_labeled_free_text_errors=None,
        actual_classifier_fits=0,real_api_requests=0,scope='Posthoc descriptive analysis; machine flags are not independent error labels'))
