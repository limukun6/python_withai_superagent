"""Freeze developer-authored cases before implementing/running the semantic checker."""
import argparse
import copy
from pathlib import Path
import time
from scripts.a_batch_common import save, hashes
from scripts.run_a10_real_task_pilot import sha

ROOT=Path(__file__).resolve().parents[1]


def cases():
    rows=[]
    for family,kind,prefix in [('logistic','numeric_tabular','numeric_tabular_logistic'),
                               ('linear_svm','single_text','single_text_linear_svm'),
                               ('rbf_svm','numeric_tabular','numeric_tabular_rbf_svm')]:
        for old,new,right,wrong in [(.1,1.,'Decreasing','Increasing'),(10.,1.,'Increasing','Decreasing')]:
            for verb,expected in [(right,'confirmed'),(wrong,'contradicted')]:
                rows.append(dict(id=f'change_{family}_{old}_{new}_{expected}',group='explicit_change',
                    family=family,kind=kind,prefix=prefix,old=old,new=new,
                    text=f'{verb} regularization strength from C={old} to C={new} may change accuracy.',expected=expected))
    for family,kind,prefix in [('logistic','numeric_tabular','numeric_tabular_logistic'),
                               ('linear_svm','single_text','single_text_linear_svm')]:
        for text,status in [('Increasing C weakens regularization.','confirmed'),
                            ('Increasing C strengthens regularization.','contradicted'),
                            ('Decreasing C strengthens regularization.','confirmed'),
                            ('Decreasing C weakens regularization.','contradicted'),
                            ('C is inverse regularization strength.','confirmed'),
                            ('C is directly proportional to regularization strength.','contradicted')]:
            rows.append(dict(id=f'definition_{family}_{len(rows)}',group='paired_definitions',family=family,kind=kind,
                             prefix=prefix,old=.1,new=1.,text=text,expected=status))
    specials=[
      ('The previous model had stronger regularization; the selected model may improve.',{}),
      ('Unlike the baseline, higher regularization strength C=1.0 might help.',{}),
      ('Increasing regularization strength from C=0.1 to C=1.0 is not our hypothesis.',{}),
      ('Whether increasing C strengthens regularization is unknown.',{}),
      ('C=1.0 means stronger regularization than C=0.1 for a different representation.',{'different_route':True}),
      ('Increasing regularization strength from C=0.2 to C=1.0 may help.',{}),
      ('Increasing regularization strength from C=0.1 to C=1.0 may help.',{'failed_reference':True}),
      ('Increasing regularization strength from C=0.1 to C=1.0 may help.',{'missing_reference':True}),
      ('This low regularization setting may be underfitting.',{}),
      ('C=0.1 matches a low-regularization region.',{}),
      ('C=1.0 will improve accuracy.',{}),
      ('增大C会增强正则。',{}),
      ('Increasing regularization strength from C=0.1 to C=1.0 may help.',{'family':'extra_trees'}),
      ('Increasing regularization strength from C=0.1 to C=1.0 may help.',{'penalty':None}),
      ('Increasing regularization strength from C=0.1 to C=1.0 may help.',{'new':0.}),
      ('Increasing regularization strength from C=0.1 to C=1.0 may help.',{'unverified':True}),
    ]
    for i,(text,extra) in enumerate(specials):
        rows.append(dict(id=f'scope_{i}',group='scope' if i<8 else 'unsupported',family='logistic',kind='numeric_tabular',
                         prefix='numeric_tabular_logistic',old=.1,new=1.,text=text,expected='unknown',options=extra))
    assert len(rows)==40
    return rows


def materialize(row):
    row=copy.deepcopy(row);opts=row.get('options',{})
    family=opts.get('family',row['family']);new=opts.get('new',row['new'])
    candidate={'id':row['prefix']+f'_C{new}','params':{'C':new}}
    if 'penalty' in opts:candidate['params']['penalty']=opts['penalty']
    reference={'trial':1,'candidate':row['prefix']+f'_C{row["old"]}','params':{'C':row['old']},
               'status':'completed','search_score':.7,'primary_metric':'accuracy',
               'estimator':{'family':family,'source':'fitted_estimator'}}
    if opts.get('different_route'):reference['candidate']='other_representation_C0.1'
    if opts.get('failed_reference'):reference['status']='failed'
    history=[] if opts.get('missing_reference') else [reference]
    facts={'classifier':family,'source':'unverified' if opts.get('unverified') else 'registered_training_function'}
    return {'reason':row['text']},candidate,history,facts,row['kind']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A120-case-freeze',source_files=hashes(ROOT),
         runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A120_A124_next_5h_plan_20261123_zh.md'),
         developer_authored=True,independent_blind_test=False,maximum_classifier_fits=0,real_API_requests=0))
    save(a.output/'cases.json',cases())
    save(a.output/'completion.json',dict(status='cases_frozen_before_implementation',cases=40,
         cases_sha256=sha(a.output/'cases.json'),actual_classifier_fits=0,real_API_requests=0,
         elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
