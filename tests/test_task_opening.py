from concurrent.futures import Future
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.unified_released_study import make_zip, read
from superagent.autonomous_run import partitions
from superagent.data import TrainingTask
from superagent.events import RunLogger
from superagent.metrics import MetricSpec
from superagent.opening_evidence import instructions, make_brief, split_audit, verified_windows
from superagent.task_opening import TaskOpening, validate


METRIC=MetricSpec('accuracy',(0,1))


def outcome(check='minority_errors'):
    return dict(goal='Classify the reading.',metric='accuracy',citations=['doc_1','balance'],uncertainty='Rare class coverage',
                check=check,hypothesis='The rare class may have higher out-of-fit error.',if_false='Keep ordinary allocation.')


def training_task():
    y=np.tile(np.r_[np.zeros(10,dtype=int),np.ones(30,dtype=int)],3)
    x=np.arange(len(y)*2,dtype=float).reshape(-1,2)
    task=TrainingTask('numeric_tabular',x,y,{'feature_shape':[2]})
    indices=np.arange(len(y))
    folds={name:(np.delete(indices,np.s_[a:a+40]),indices[a:a+40],{}) for name,a in [('primary',0),('confirmation_1',40),('confirmation_2',80)]}
    return task,folds


class OpeningTests(unittest.TestCase):
    def test_reads_only_documented_role_and_structured_split_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'random.zip'
            make_zip(path,[('README.md',b'Predict a class for previously unseen devices identified by device_id. Subjects can vary between locations. Report accuracy.'),
                           ('test/README.md',b'Predict the SECRET_TARGET.'),('test/labels.csv',b'secret')])
            task=TrainingTask('numeric_tabular',np.zeros((10,2)),np.arange(10)%2,
                 {'input_schema':{'columns':[{'name':'device_id'},{'name':'t'}]}})
            inferred,lines,origin=instructions(path,task)
            self.assertEqual((origin,inferred.spec['evaluation_protocol']),('documented_column',dict(kind='entity',column='device_id')))
            self.assertIn('field_1',lines[0])
            self.assertIn('Subjects can vary between locations',lines)
            self.assertNotIn('SECRET',' '.join(lines))
            task.spec['evaluation_protocol']={'kind':'temporal','column':'t'}
            fixed,_,origin=instructions(path,task)
            self.assertEqual((origin,fixed.spec['evaluation_protocol']['kind']),('structured','temporal'))

    def test_cross_group_and_time_ties_checked_on_actual_training_windows(self):
        from superagent.schema import MixedRows
        groups=np.repeat(np.arange(12),4)
        y=np.tile([0,1,0,1],12)
        schema={'input_schema':{'columns':[{'name':'unit'},{'name':'value'}]}}
        matrix=MixedRows(np.array([[str(g),str(i)] for i,g in enumerate(groups)]),(1,),('unit','value'))
        task=TrainingTask('mixed_tabular',matrix,y,dict(schema,evaluation_protocol=dict(kind='entity',column='unit')))
        folds=verified_windows(task,17,partitions)
        self.assertTrue(all(row['passes'] for row in split_audit(task,folds).values()))
        task.spec['evaluation_protocol']={'kind':'temporal','column':'unit'}
        folds=verified_windows(task,17,partitions)
        for fit,held,_ in folds.values():
            self.assertLess(max(groups[fit]),min(groups[held]))
        self.assertTrue(any(info['tied_training_rows_removed'] for _,_,info in folds.values()))

    def test_brief_and_plan_do_not_depend_on_held_values_and_ignore_untrusted_response(self):
        task,folds=training_task()
        brief=make_brief(task,METRIC,folds,['Predict the class.'],'ordinary')
        task.X_train[folds['primary'][1]]=1e9
        task.y_train[folds['primary'][1]]=0
        self.assertEqual(make_brief(task,METRIC,folds,['Predict the class.'],'ordinary'),brief)
        self.assertEqual(validate(outcome(),brief)['check'],'minority_errors')
        for bad in (dict(metric='macro_f1'),dict(citations=['balance','fiction']),dict(citations=['contract','balance']),
                    dict(check='run_shell'),dict(citations=['doc_1','contract']),dict(split='random')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate(dict(outcome(),**bad),brief)

    def test_data_examples_omit_literal_text_and_task_identity(self):
        task,folds=training_task()
        task.kind='single_text'
        task.X_train=['secret author and private correlation']*len(task.y_train)
        packet=make_brief(task,METRIC,folds,['Classify the sentence.'],'ordinary')
        self.assertNotIn('secret author',json.dumps(packet))
        self.assertEqual(len(packet['anonymized_fitting_examples']),3)
        self.assertEqual(packet['anonymized_fitting_examples'][0]['word_count'],5)

    def test_refutation_preserves_local_experiment_and_supported_check_requires_confirmation(self):
        for affected,confirm_loss in ((False,False),(True,True),(True,False)):
            with self.subTest(affected=affected,confirm_loss=confirm_loss),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp)
                task,folds=training_task()
                opening=TaskOpening('opening-flash',task,METRIC,folds,['Predict the class.'],'ordinary',
                                    RunLogger(folder),None,time.monotonic(),60)
                future=Future()
                future.set_result(outcome())
                opening.future=future
                opening.record['interpretation_status']='pending'
                observed={}
                for fold,(_,held,_) in folds.items():
                    y=task.y_train[held]
                    a=y.copy(); b=y.copy()
                    if affected:
                        a[:6]=1
                        b[:2]=1
                    if confirm_loss and fold!='primary':
                        b[10:18]=0
                    def row(pred):
                        return dict(row_key=fold,labels=y,predictions=pred,probabilities=None,score=float(np.mean(pred==y)))
                    observed[fold]={'base':row(a),'challenger':row(b)}
                exec_=SimpleNamespace(limit=16,caps={n:dict(classifier_fits=1,refit_classifier_fits=1) for n in ('base','challenger')},
                                      accounting=lambda:dict(attempted_classifier_fits=8,unknown_failed_attempts=0))
                incumbent=dict(model_sha256='original',model_file='selected_model.joblib')
                choice=dict(members=['base'])
                with patch('superagent.task_opening.actions',return_value=[]),patch('superagent.autonomous_run.freeze_model',
                     return_value={'model_sha256':'new','model_file':'opening_selected_model.joblib'}) as refit:
                    selected,_=opening.after_delivery({},choice,incumbent,[],observed,{}, {},exec_,5,set())
                self.assertEqual(read(folder/'opening_incumbent.json'),incumbent)
                self.assertEqual(opening.record['changed_experiment'],affected)
                self.assertEqual(refit.called,affected and not confirm_loss)
                self.assertEqual(selected['model_sha256'],'new' if affected and not confirm_loss else 'original')
                opening.finish()
                self.assertTrue((folder/'opening_lesson.json').is_file())

    def test_invalid_or_unavailable_call_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            task,folds=training_task()
            flash=Mock(blocked=False,request_count=0)
            flash.ask.return_value=dict(metric='incorrect')
            run=TaskOpening('opening-flash',task,METRIC,folds,[],'ordinary',RunLogger(Path(tmp)),flash,time.monotonic(),60)
            run.launch()
            run.finish()
            run.collect(0)
            self.assertEqual(run.record['interpretation_status'],'invalid')
            self.assertEqual(flash.ask.call_count,1)

    def test_team_opening_uses_the_same_short_request_and_respects_a_zero_call_budget(self):
        from superagent.autonomous_run import _train
        from superagent.unified import reflection_client
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{},clear=True):
            directory=Path(tmp)
            with patch('superagent.unified.configured_flash') as factory:
                reflection_client(directory,'opening-flash')
                factory.assert_called_once_with(directory,thinking_mode='disabled')
            flash=Mock(blocked=False,request_count=0,max_requests=0)
            with patch('superagent.semantic_contract.load_training_task',side_effect=ValueError('unknown schema')), \
                    patch('superagent.semantic_contract.adapt_training',side_effect=ValueError('unknown target')), \
                    patch('superagent.partial_classification.record_input_failure',return_value=1):
                _train('input.zip',directory/'run',fit_limit=8,seed=42,wall=60,timeout=5,flash=flash,
                       feedback_reflection='opening-flash')
            self.assertEqual(flash.max_requests,0)
            task,folds=training_task()
            run=TaskOpening('opening-flash',task,METRIC,folds,[],'ordinary',RunLogger(directory),flash,time.monotonic(),60)
            run.launch()
            run.finish()
            flash.ask.assert_not_called()
