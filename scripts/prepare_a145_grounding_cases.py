"""Freeze program-oracle semantic cases before implementing grounded verification; no fits/API."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import sha

ROOT=Path(__file__).resolve().parents[1]


def encoded(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(encoded(v).encode()).hexdigest()


def create_case(group,index,caps):
    small='numeric_tabular_logistic_C0.1';large='numeric_tabular_logistic_C1.0';other='numeric_tabular_logistic_C10.0'
    task=dict(contract_version=1,objective='classification',kind='numeric_tabular',input_schema={},
              metric=dict(name='accuracy',labels=[0,1],direction='maximize'),
              goal_resolution={'fields':{'primary_metric':{'status':'confirmed','source':'declared_manifest','value':'accuracy'}}})
    binding=dict(run_sha256=digest(['fixture',group,index]),source_sha256=digest(hashes(ROOT)),origin='controlled_fixture',
                 protocol_facts={'statistical_test_performed':False,'slice_to_global_supported':False})
    score=round(.70+.001*index,3);history=[]
    for trial,name,held in [(1,small,score),(2,other,round(score+.03,3))]:
        history.append(dict(trial=trial,candidate=name,params=deepcopy(caps[name]['params']),status='completed',split_id='primary',
            primary_metric='accuracy',metric_spec=task['metric'],search_score=held,train_score=.9,score_gap=round(.9-held,3),
            search_accuracy=held,train_accuracy=.9,search_metrics={'accuracy':held,'rows':1000},train_metrics={'accuracy':.9,'rows':4000},
            metric_identity={'status':'confirmed','expected':task['metric'],'observed':task['metric']},
            estimator={'family':'logistic','class':'LogisticRegression','source':'fitted_estimator'},
            evaluation_payload_sha256=digest('payload'),label_sha256=digest('labels'),row_ids_sha256=digest('rows')))
    state=dict(task=task,history=history,available_candidates=[deepcopy(c) for n,c in caps.items() if n not in (small,other)],
               budget=dict(fit_limit=11,attempted_classifier_fits=2,completed_classifier_fits=2,remaining_fit_allowance=9,
                           reserved_final_refit=1,llm_request_limit=3,used_llm_requests=0))
    role=index%4;expected=['confirmed','confirmed','contradicted','unknown'][role]
    if group=='object_identity':
        ref='t1.classifier' if role!=1 else 'c:'+large+'.classifier';value='extra_trees' if role==2 else 'logistic'
        claim=dict(kind='value',ref=ref,value=value,expected_role='declared' if role==1 else 'observed')
        sentence=('The selected candidate uses ' if role==1 else 'Trial 1 uses ')+('ExtraTrees.' if role==2 else 'logistic regression.')
        if role==3:history[0]['estimator']['family']=None
    elif group=='measurement_scope':
        op='<' if role==2 else '==' if role==1 else '>'
        claim=dict(kind='compare',left='t1.search.accuracy' if role==1 else 't2.search.accuracy',op=op,right='t1.search.accuracy')
        sentence=f'Trial 2 search accuracy was {100*history[1]["search_score"]:.2f}%, '+('lower than' if role==2 else 'higher than')+f' trial 1 search accuracy {100*score:.2f}%.'
        if role==3:
            history[1]['split_id']='confirmation_1';history[1]['evaluation_payload_sha256']=digest('different payload')
    elif group=='parameter_direction':
        target=small if role==1 else 'numeric_tabular_rbf_svm_C1.0' if role==3 else large
        direction='unchanged' if role==1 else 'stronger_regularization' if role==2 else 'weaker_regularization'
        claim=dict(kind='parameter_relation',before='t1.C',after='c:'+target+'.C',direction=direction)
        sentence='Relative to trial 1, changing from C=0.1 to C=1.0 gives '+('stronger' if role==2 else 'weaker')+' regularization strength.'
    elif group=='causal_slice_scope':
        claim=dict(kind='value',ref='protocol.slice_to_global_supported' if role==1 else 'protocol.statistical_test_performed',value=role==2)
        sentence='A statistical significance test was performed.' if role==2 else 'No statistical significance test was performed.'
        if role==3:claim=dict(kind='causal',text='Increasing C caused the measured gain.');sentence=claim['text']
    elif group=='failed_missing_evidence':
        history[1]['status']='failed'
        claim=dict(kind='value',ref='t2.status',value='completed' if role==2 else 'failed')
        sentence='Trial 2 '+('completed.' if role==2 else 'failed.')
        if role==1:claim=dict(kind='value',ref='t1.search.accuracy',value=score);sentence=f'Trial 1 search accuracy was {100*score:.2f}%.'
        if role==3:claim=dict(kind='value',ref='t2.search.accuracy',value=history[1]['search_score']);sentence=f'Trial 2 search accuracy was {100*history[1]["search_score"]:.2f}%.'
    else:
        claim=dict(kind='value',ref='task.metric' if role==1 else 'task.kind',value='accuracy' if role==1 else 'image' if role==2 else 'numeric_tabular')
        sentence='The task is an image classification task.' if role==2 else 'The task is numeric classification with accuracy as its objective.'
    namespace=digest({'binding':binding,'task':task,'capabilities':caps})
    claim['namespace']=digest('another source version') if group=='task_version_scope' and role==3 else namespace
    return dict(id=group+'_'+str(index),group=group,split='development' if index<4 else 'retained',
                state=state,capabilities=caps,binding=binding,selected_candidate=large,claim=claim,
                legacy_sentence=sentence,expected=expected,oracle_origin='developer-authored controlled executor records; not trained models')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(out/'protocol.json',dict(batch='A145-cases',source_files=hashes(ROOT),runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A145_grounded_protocol_execution_plan_20261124_zh.md'),maximum_fits=0,maximum_API_calls=0,
         count=48,groups=6,per_half_per_group={'confirmed':2,'contradicted':1,'unknown':1}))
    from superagent.ab_run import registered_capabilities
    task=dict(contract_version=1,objective='classification',kind='numeric_tabular',input_schema={},metric={'name':'accuracy','labels':[0,1],'direction':'maximize'})
    caps=registered_capabilities(task,200);all_cases=[]
    for group in ('object_identity','measurement_scope','parameter_direction','causal_slice_scope','failed_missing_evidence','task_version_scope'):
        all_cases.extend(create_case(group,i,caps) for i in range(8))
    for split in ('development','retained'):save(out/(split+'.json'),[c for c in all_cases if c['split']==split])
    save(out/'completion.json',dict(status='cases_frozen',actual_classifier_fits=0,real_API_requests=0,
         development_sha256=sha(out/'development.json'),retained_sha256=sha(out/'retained.json'),
         elapsed_seconds=time.monotonic()-start,retained_evaluated=False))
    print('Frozen 48 program-oracle cases: development24, retained24; no fits or API.')


if __name__=='__main__':main()
