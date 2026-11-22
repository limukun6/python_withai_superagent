"""Prepare independently specified A-side development cases, then evaluate frozen cases."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha,read
from superagent.ab_run import sources

ROOT=Path(__file__).resolve().parents[1]


def cases():
    rows=[]
    def add(group,op,items):
        for i,(expected,payload,basis) in enumerate(items,1):
            rows.append(dict(id=f'{group}_{i}',category=group,operation=op,subset='A_development',
                author='A',expected=expected,expectation_basis=basis,payload=payload))
    add('task_semantics','semantics',[
        ('confirmed',{'meaning':'preference','candidate':'word_C0.05'},'Preference allows signed differences and flipped augmentation.'),
        ('confirmed',{'meaning':'symmetric','candidate':'symmetric_word_C0.3'},'Symmetric classifier preserves pair order.'),
        ('contradicted',{'meaning':'symmetric','candidate':'word_C0.05'},'A flipping method contradicts preserved labels.'),
        ('contradicted',{'meaning':'preference','candidate':'symmetric_word_C0.3'},'Preserved-label method contradicts directional preference.'),
        ('unknown',{'meaning':None,'candidate':'word_C0.05'},'No task declaration cannot establish intended pair semantics.'),
        ('unknown',{'meaning':None,'candidate':'symmetric_word_C0.3'},'A method property cannot establish unknown task intent.')])
    add('configuration','configuration',[
        ('confirmed',{'claim':'ridge','observed':'ridge'},'Declared family equals supplied controlled artifact fact.'),
        ('confirmed',{'claim':'poly_svm','observed':'poly_svm'},'Kernel-specific family agrees.'),
        ('contradicted',{'claim':'linear_svm','observed':'logistic'},'Log loss classifier is not a hinge-loss SVM.'),
        ('contradicted',{'claim':'catboost','observed':'extra_trees'},'Different estimator family cannot be borrowed.'),
        ('unknown',{'claim':'ridge','observed':None},'Missing observed family provides no confirmation.'),
        ('unknown',{'claim':None,'observed':'logistic'},'No explicit family claim was stated.')])
    history=[dict(trial=7,status='completed',primary_metric='macro_f1',search_score=.73,search_accuracy=.81,train_accuracy=.94),
             dict(trial=11,status='failed',primary_metric='macro_f1',search_score=.99)]
    def measurement(trial,split,metric,value):return dict(history=history,claim=dict(trial=trial,split=split,metric=metric,value=value))
    add('trial_identity','measurement',[
        ('confirmed',measurement(7,'search','accuracy',.81),'Trial 7 actual search accuracy is .81.'),
        ('confirmed',measurement(7,'search','macro_f1',.73),'Trial 7 actual primary F1 is .73.'),
        ('contradicted',measurement(3,'search','accuracy',.81),'Trial 3 never ran; cannot borrow trial 7.'),
        ('contradicted',measurement(17,'train','accuracy',.94),'Trial 17 never ran.'),
        ('unknown',measurement(11,'search','macro_f1',.99),'Failed trial residual .99 is not a valid measurement.'),
        ('unknown',dict(history=history+[deepcopy(history[0])],claim=dict(trial=7,split='search',metric='accuracy',value=.81)), 'Duplicate identity is ambiguous.')])
    add('metric_partition','measurement',[
        ('confirmed',measurement(7,'train','accuracy',.94),'Training accuracy is .94.'),
        ('confirmed',measurement(7,'search','macro_f1',.73),'Primary F1 differs from diagnostic accuracy.'),
        ('contradicted',measurement(7,'search','accuracy',.94),'Search cannot borrow train accuracy.'),
        ('contradicted',measurement(7,'search','macro_f1',.81),'Macro F1 cannot borrow accuracy.'),
        ('unknown',measurement(7,'search','roc_auc',.73),'No probability-based AUC measurement supplied.'),
        ('unknown',measurement(7,'train','macro_f1',.94),'No training F1 supplied.')])
    add('negation_future','prose',[
        ('confirmed',{'text':'This is logistic regression.','fact':'logistic'},'Positive present family statement matches the supplied fact.'),
        ('confirmed',{'text':'The classifier is a linear SVM.','fact':'linear_svm'},'Explicit matching present statement.'),
        ('contradicted',{'text':'This is an SVM.','fact':'logistic'},'A definite conflicting family claim.'),
        ('contradicted',{'text':'The classifier is CatBoost.','fact':'ridge'},'Definite family mismatch.'),
        ('unknown',{'text':'This is not logistic regression.','fact':'logistic'},'Negation must not certify the positive classifier mention.'),
        ('unknown',{'text':'We might use logistic regression.','fact':'logistic'},'Future possibility is not an executed classifier fact.')])
    add('slice_scope','slice',[
        ('confirmed',{'after':.72},'Same slice improves by .12, exceeds required .05.'),
        ('contradicted',{'after':.61},'Same slice improves only .01, not required .05.'),
        ('unknown',{'after':.9,'changed_threshold':True},'Different threshold means another slice.'),
        ('unknown',{'after':.9,'missing_slice':True},'No target slice measurement exists.'),
        ('unknown',{'after':.9,'changed_split':True},'Different evaluation split cannot establish matched slice improvement.'),
        ('unknown',{'after':.9,'changed_payload':True},'Different evaluation payload cannot establish matched slice improvement.')])
    add('memory_state','memory',[
        ('confirmed',{'check':'supported','claim':'measurement_supported'},'Recorded numerical support is retained as measurement support.'),
        ('confirmed',{'check':'not_supported','claim':'measurement_not_supported'},'Refuted numerical prediction stays refuted.'),
        ('contradicted',{'check':'not_supported','claim':'measurement_supported'},'A failed prediction cannot become support in memory.'),
        ('contradicted',{'check':'supported','claim':'causally_verified'},'Numerical support is never causal verification.'),
        ('unknown',{'check':'not_evaluable','claim':'measurement_supported'},'No evaluable numerical test exists.'),
        ('unknown',{'check':'not_stated','claim':'measurement_supported'},'No numerical prediction was made.')])
    add('failure_comparison','overall',[
        ('confirmed',{'after':.8},'Comparable completed primary score rises from .6 by at least .1.'),
        ('contradicted',{'after':.62},'Comparable change .02 falls short of .1.'),
        ('unknown',{'after':.99,'failed_result':True},'Failed execution has no valid score despite residual .99.'),
        ('unknown',{'after':.99,'failed_reference':True},'Failed reference cannot anchor a comparison.'),
        ('unknown',{'after':.99,'changed_metric':True},'Primary metrics differ; subtraction cannot test expected primary gain.'),
        ('unknown',{'after':.99,'changed_split':True},'Scores on different partitions do not verify a paired expected improvement.')])
    return rows


def evaluate(case):
    p=case['payload'];op=case['operation']
    if op=='semantics':
        from superagent.models import search_candidates
        from superagent.task_semantics import validate_candidate_semantics
        c=next(c for c in search_candidates('pair_text','all') if c.family==p['candidate'])
        try:r=validate_candidate_semantics('pair_text',p['meaning'],c)
        except ValueError as e:return 'contradicted',dict(error=str(e))
        return ('unknown' if p['meaning'] is None else 'confirmed'),r
    if op=='configuration':
        from superagent.estimator_facts import verify_configuration_claim
        r=verify_configuration_claim({'configuration_claim':{'classifier':p['claim']}},{'family':p['observed']})
        return r['status'] if r['status'] in ('confirmed','contradicted') else 'unknown',r
    if op=='measurement':
        from superagent.measurement_claims import check_measurement_claim
        r=check_measurement_claim(p['claim'],p['history']);return r['status'] if r['status']!='unverifiable' else 'unknown',r
    if op=='prose':
        from superagent.prose_claims import check_prose
        r=check_prose({'reason':p['text']},{'classifier':p['fact']},[])
        observed='contradicted' if r['contradicted'] else 'unknown' if r['unverifiable'] or not r['claims_checked'] else 'confirmed'
        return observed,r
    if op in ('slice','overall'):
        before=dict(trial=1,status='completed',primary_metric='accuracy',search_score=.6,split_id='primary',evaluation_payload_sha256='a'*64)
        after=dict(before,trial=2,search_score=p['after'])
        if p.get('changed_split'):after['split_id']='confirmation_1'
        if p.get('changed_payload'):after['evaluation_payload_sha256']='b'*64
        if op=='slice':
            from superagent.targeted_feedback import check_slice_expectation
            s=dict(slice_id='long',feature='length',threshold_from_training=12,relation='high',rows=20,accuracy=.6)
            before['error_slices']={'slices':[s]};t=dict(s,accuracy=p['after'])
            if p.get('changed_threshold'):t['threshold_from_training']=19
            after['error_slices']={'slices':[] if p.get('missing_slice') else [t]}
            r=check_slice_expectation(dict(reference_trial=1,slice_id='long',min_delta=.05),after,[before])
        else:
            from superagent.hypothesis_agent import check_expectation
            if p.get('failed_result'):after['status']='failed'
            if p.get('failed_reference'):before['status']='failed'
            if p.get('changed_metric'):after['primary_metric']='macro_f1'
            r=check_expectation({'expected_change':dict(metric='search_score',reference_trial=1,direction='increase',min_delta=.1)},after,[before])
        return {'supported':'confirmed','not_supported':'contradicted','not_evaluable':'unknown'}[r['status']],r
    if op=='memory':
        from superagent.hypothesis_memory import build_memory
        h=[dict(trial=1,candidate='word_C0.05',params={'C':.05},status='completed',primary_metric='accuracy',search_score=.6,search_accuracy=.6,
                proposal=dict(hypothesis='Controlled hypothesis',evidence_trials=[],expected_change=None),hypothesis_check={'status':p['check']})]
        r=build_memory(h,[],'accuracy');status=r['hypotheses'][0]['status']
        if status in ('not_evaluable','no_numerical_expectation'):return 'unknown',r
        if p['claim']=='causally_verified':return ('confirmed' if r['hypotheses'][0]['causal_explanation_verified'] else 'contradicted'),r
        return ('confirmed' if status==p['claim'] else 'contradicted'),r
    raise ValueError('Unsupported case operation')


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if a.mode=='prepare':
        rows=cases();save(root/'A_development.json',rows)
        save(root/'B_independent_reserved_request.json',dict(ready=False,required_cases=48,per_category=6,categories=sorted({r['category'] for r in rows}),
            cases=[],instruction='Supply independently authored inputs, expected confirmed/contradicted/unknown and expectation_basis before checking output; do not copy A templates and rename values.'))
        save(root/'completion.json',dict(status='prepared',A_development_cases=len(rows),independent_reserved_cases=0,
            original_96_case_acceptance_complete=False,actual_classifier_fits=0,real_api_requests=0));return
    from scripts import a38_predict_guard
    rows=read(a.cases);frozen=sources();start=time.monotonic();results=[]
    save(root/'protocol.json',dict(batch='A67',source_files=frozen,runner_sha256=sha(Path(__file__)),cases_sha256=sha(a.cases),
        plan_sha256=sha(ROOT/'docs/batches/A67_unified_case_registry_plan.md'),scope='A-authored prospective development cases, not independent sealed evaluation',
        classifier_fitting_forbidden=True,network_forbidden=True))
    for c in rows:
        if c['expected'] not in ('confirmed','contradicted','unknown') or not c['expectation_basis']:raise ValueError('Missing independent case label basis')
        try:status,detail=evaluate(c)
        except Exception as error:
            save(root/'results.json',results)
            save(root/'completion.json',dict(status='stopped',case=c['id'],error=repr(error),completed_cases=len(results),actual_classifier_fits=0,real_api_requests=0))
            raise
        results.append(dict(id=c['id'],category=c['category'],expected=c['expected'],observed=status,passed=status==c['expected'],detail=detail))
        save(root/'results.json',results)
    save(root/'results.json',results)
    groups={g:{'cases':sum(r['category']==g for r in results),'passed':sum(r['category']==g and r['passed'] for r in results),
        'false_confirmations':sum(r['category']==g and r['expected']!='confirmed' and r['observed']=='confirmed' for r in results),
        'valid_not_confirmed':sum(r['category']==g and r['expected']=='confirmed' and r['observed']!='confirmed' for r in results)} for g in sorted({r['category'] for r in results})}
    save(root/'completion.json',dict(status='passed' if all(r['passed'] for r in results) else 'failed_cases',cases=len(results),passed=sum(r['passed'] for r in results),
        by_category=groups,actual_classifier_fits=0,real_api_requests=0,source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,
        independent_reserved_evaluated=False))
    print(json.dumps(read(root/'completion.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run']);p.add_argument('--cases',type=Path)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
