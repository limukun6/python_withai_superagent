"""Evaluate preregistered development cases and old real replies with fitting/network forbidden."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a120_parameter_semantics import materialize

ROOT=Path(__file__).resolve().parents[1]
FOCUS={'29491f1bd1d1','69897653704d','4dae4566c926','a312f009229e'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A121',source_files=source,runner_sha256=sha(Path(__file__)),
        cases_sha256=sha(a.cases),plan_sha256=sha(ROOT/'docs/batches/A120_A124_next_5h_plan_20261123_zh.md'),
        developer_authored_cases=True,old_replies_seen=True,independent_blind_test=False,
        maximum_classifier_fits=0,real_API_requests=0))
    try:
        from scripts import a38_predict_guard
        from superagent.parameter_semantics import check_parameter_semantics,definition
        from superagent.configuration_facts import configuration_change,choose_reference
        from superagent.proposal_consistency import direction_diagnostics
        cases=read(a.cases)
        if len(cases)!=40 or len({r['id'] for r in cases})!=40:raise ValueError('Case count/IDs changed')
        docs={k:definition(k) for k in ('logistic','linear_svm','rbf_svm')}
        if a.validate_only:
            save(a.output/'completion.json',dict(status='validated',actual_classifier_fits=0,real_API_requests=0));return
        rows=[]
        for row in cases:
            fields,candidate,history,facts,kind=materialize(row)
            ref,rule=choose_reference(history,candidate)
            baseline=direction_diagnostics(fields,configuration_change(ref,candidate,rule))
            report=check_parameter_semantics(fields,candidate,history,facts,kind)
            states=[r['status'] for r in report['items']]
            observed='contradicted' if 'contradicted' in states else 'confirmed' if 'confirmed' in states else 'unknown'
            rows.append(dict(id=row['id'],expected=row['expected'],observed=observed,passed=observed==row['expected'],
                baseline_warning=bool(baseline['warnings']),report=report))
        original=read(ROOT/'research/a70_independent_review/blinded_reviews.json');replies=[]
        for row in original:
            fields={k:row['proposal'].get(k,'') for k in ('reason','review','hypothesis')}
            candidate=row['selected_registered_configuration']
            begin=time.perf_counter()
            report=check_parameter_semantics(fields,candidate,row['history'],
                {'classifier':candidate['estimator'],'source':'shared_ab_catalogue'},row['task_contract']['kind'])
            replies.append(dict(review_id=row['review_id'],focus=row['review_id'] in FOCUS,
                elapsed_seconds=time.perf_counter()-begin,report=report))
        stats=dict(cases=40,agreements=sum(r['passed'] for r in rows),
            false_contradictions=sum(r['expected']!='contradicted' and r['observed']=='contradicted' for r in rows),
            contradictions_expected=sum(r['expected']=='contradicted' for r in rows),
            contradictions_detected=sum(r['expected']==r['observed']=='contradicted' for r in rows),
            original_baseline_warnings=sum(r['baseline_warning'] for r in rows),
            unknowns=sum(r['observed']=='unknown' for r in rows),
            mismatches=[{k:r[k] for k in ('id','expected','observed')} for r in rows if not r['passed']],
            old_replies=24,focus_errors=4,focus_detected=sum(r['focus'] and bool(r['report']['warnings']) for r in replies),
            focus_uncovered=[r['review_id'] for r in replies if r['focus'] and not r['report']['warnings']],
            actual_prompt_tokens=None,actual_API_cost=None,real_API_requests=0,
            runtime_definitions=docs,limitations=['Developer fixtures, not blind generalization',
                'Absolute low-C/low-regularization region is not certified without an explicit comparator',
                'No measured accuracy benefit or actual Flash correction'],source_unchanged=hashes(ROOT)==source)
        save(a.output/'case_results.json',rows);save(a.output/'reply_results.json',replies);save(a.output/'results.json',stats)
        save(a.output/'completion.json',dict(status='completed_development_evaluation',actual_classifier_fits=0,
            real_API_requests=0,A_cumulative_fits_after=1505,elapsed_seconds=time.monotonic()-started,
            source_unchanged=stats['source_unchanged']))
        print(json.dumps({k:stats[k] for k in ('agreements','false_contradictions','contradictions_expected','contradictions_detected','focus_detected','mismatches')}),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
