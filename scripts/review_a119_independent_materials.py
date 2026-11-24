"""Replay original B cases and preserve B labels alongside an explicitly post-label A review."""
import argparse
from collections import Counter
import importlib.metadata
from pathlib import Path
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]
FOCUS={
 '29491f1bd1d1':dict(A_judgement='clear_parameter_direction_error',basis='C=.1→1弱化正则；hypothesis明确说强度增加，与同句less constrained亦冲突。'),
 '69897653704d':dict(A_judgement='clear_parameter_direction_error',basis='C=1→10弱化正则，reason正确而hypothesis称强度增加，回答内部自相矛盾。'),
 '4dae4566c926':dict(A_judgement='clear_parameter_direction_error',basis='低C的参数区域被称为low-regularization region，按正则强度含义方向相反；不是新模型实测错误。'),
 'a312f009229e':dict(A_judgement='clear_parameter_direction_error',basis='reason明确称C=.05→.1为higher regularization strength，实际相反。'),
 '11ef4b2366e7':dict(A_judgement='ambiguous_parameter_terminology',basis='low regularization setting可误读成强度弱，也可能意指低C；may be underfitting更接近低C强正则解释。保留B原错标，A不把此句算无歧义事实错。'),
 '5044e26af2b5':dict(A_judgement='clear_history_scope_error',basis='review先承认一条completed primary，再称No candidate has primary-split evidence yet；实际trial1已有F1=.46694。只应说新word_char候选无证据。'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--B-receipts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    from scripts import a38_predict_guard
    from scripts.run_a67_case_registry import evaluate
    from superagent.ab_run import sources
    from tools.validate_a1_case_pack import validate
    from sklearn.linear_model import LogisticRegression
    source=sources();folder=a.B_receipts/'research/b_integrated_offline_1123'
    case_file=folder/'a1_cases.json';annotation_file=folder/'a2_annotated_reviews.json'
    save(a.output/'protocol.json',dict(batch='A119',runner_sha256=sha(Path(__file__)),source_files=source,
         plan_sha256=sha(ROOT/'docs/batches/A118_A119_five_task_and_independent_review_plan.md'),
         B_cases_sha256=sha(case_file),B_annotations_sha256=sha(annotation_file),
         post_label_review=True,A_reviewer_has_seen_B_labels=True,strict_blind_or_human_gold_standard=False,
         actual_classifier_fits=0,real_API_requests=0))
    try:
        cases=read(case_file);errors=validate(cases,'B')
        if errors:raise ValueError('Case schema invalid: '+repr(errors))
        prior={r['id']:r for r in read(folder/'a1_check/results.json')}
        rows=[];inconsistent=[]
        for case in cases:
            observed,detail=evaluate(case)
            row=dict(id=case['id'],category=case['category'],expected=case['expected'],observed=observed,
                     passed=observed==case['expected'],B_original_observed=prior[case['id']]['observed'],detail=detail)
            rows.append(row)
            payload=case['payload']
            if case['operation']=='slice' and not payload.get('missing_slice'):
                correct=payload['after']*20
                if abs(correct-round(correct))>1e-9:
                    inconsistent.append(dict(id=case['id'],accuracy=payload['after'],rows=20,implied_correct=correct,
                         interpretation='Abstract threshold fixture cannot be a realizable 20-row accuracy; original case retained'))
        bad_ids={r['id'] for r in inconsistent}
        sensitivity=[r for r in rows if r['id'] not in bad_ids]
        matrix=Counter((r['expected'],r['observed']) for r in rows)
        annotations=read(annotation_file);original=read(ROOT/'research/a70_independent_review/blinded_reviews.json')
        key={r['review_id']:r for r in read(ROOT/'research/a70_independent_review/condition_key_keep_separate.json')}
        ratings={condition:Counter() for condition in ('R','E')};focus=[]
        for old,row in zip(original,annotations,strict=True):
            tags=row['independent_annotation']
            if {**old,'independent_annotation':tags}!=row:raise ValueError('Annotated payload changed')
            label={True:'yes',False:'no',None:'unknown'}[tags['factual_error']]
            metadata=key[row['review_id']];ratings[metadata['condition']][label]+=1
            if tags['factual_error'] is True:
                if row['review_id'] not in FOCUS:raise ValueError('New unexpected error label needs review')
                focus.append(dict(review_id=row['review_id'],state_id=metadata['state_id'],condition=metadata['condition'],
                    B_factual_error=True,**FOCUS[row['review_id']],B_label_basis=tags['label_basis'],
                    selected_configuration=row['selected_registered_configuration']))
        text=LogisticRegression.__doc__;start=text.index('C : float');stop=text.index('fit_intercept',start)
        C_doc=text[start:stop].strip()
        if 'Inverse of regularization strength' not in C_doc:raise ValueError('Runtime C semantics document changed')
        if source!=sources():raise ValueError('Source changed')
        summary=dict(original_cases=48,agreements=sum(r['passed'] for r in rows),
            error_confirmations=sum(r['observed']=='confirmed' and r['expected']!='confirmed' for r in rows),
            expected_unknown_but_asserted=sum(r['expected']=='unknown' and r['observed']!='unknown' for r in rows),
            confusion={e:{o:matrix[e,o] for o in ('confirmed','contradicted','unknown')} for e in ('confirmed','contradicted','unknown')},
            discrepancies=[{k:r[k] for k in ('id','expected','observed')} for r in rows if not r['passed']],
            changes_from_B_published=[r['id'] for r in rows if r['observed']!=r['B_original_observed']],
            arithmetic_fixture_inconsistencies=inconsistent,
            sensitivity_excluding_flagged_originals=dict(cases=len(sensitivity),agreements=sum(r['passed'] for r in sensitivity),posthoc=True),
            original_labels_or_payloads_modified=False,checker_modified=False,
            B_review_factual_error_by_condition={k:dict(v) for k,v in ratings.items()},
            A_targeted_reviews=focus,A_focus_judgement_counts=dict(Counter(r['A_judgement'] for r in focus)),
            runtime_parameter_document=dict(package='scikit-learn',version=importlib.metadata.version('scikit-learn'),C_documentation=C_doc),
            limitations=['A review is post-label, not blind or human validation','B one-Codex labels are not ground truth',
                         'R/E each 12 replies, repeated task prefixes correlated; do not infer population error rate',
                         'No checker revision or regenerated answers; the original 48 remain the primary denominator'])
        save(a.output/'case_results.json',rows);save(a.output/'results.json',summary)
        save(a.output/'completion.json',dict(status='completed_review',original_cases=48,agreements=summary['agreements'],
            annotations_received=24,targeted_error_reviews=len(focus),actual_classifier_fits=0,real_API_requests=0,
            checker_unchanged=True,source_unchanged=True,A_cumulative_fits_after=1505,elapsed_seconds=time.monotonic()-started))
        print(dict(agreements=summary['agreements'],originals=48,fixture_caveats=len(inconsistent),
                   ratings=summary['B_review_factual_error_by_condition'],A_focus=summary['A_focus_judgement_counts']),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
