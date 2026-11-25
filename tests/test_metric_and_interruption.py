import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_training_task
from superagent.metric_declarations import declared_metric, MetricDeclarationError
from superagent.metrics import MetricSpec
from superagent.autonomous_run import train
from superagent.predict import create_predictions


def fixture(root, *, metric=None, prose='',labels=('0','1'),positive=None,legacy=False):
    path=root/'training.zip'
    values=[labels[i%len(labels)] for i in range(48)]
    with zipfile.ZipFile(path,'w') as z:
        if legacy:
            for name,array in [('C_train.npy',np.array([[str(i)] for i in range(48)],dtype=object)),
                               ('y_train.npy',np.asarray(values))]:
                buffer=io.BytesIO();np.save(buffer,array);z.writestr(name,buffer.getvalue())
        else:
            manifest=dict(schema_version=1,modality='tabular',columns=[dict(name='signal',type='numeric')],
                     target='label',splits={'train':dict(format='csv',path='train.csv'),'test':dict(format='csv',path='test.csv')})
            if metric is not None:manifest['metric']=metric
            if positive is not None:manifest['positive_label']=positive
            z.writestr('manifest.json',json.dumps(manifest))
            z.writestr('train.csv','signal,label\n'+''.join(f'{i},{y}\n' for i,y in enumerate(values)))
            z.writestr('test.csv','signal\n7\n3\n41\n12\n')
        if prose:z.writestr('README.md',prose)
    return path


class MetricAndInterruption(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)

    def test_readme_and_manifest_spellings(self):
        for text,expected in [('Please report macro F1.','macro_f1'),('Evaluation metric: macro-averaged F1 score.','macro_f1'),
                ('Leaderboard uses AUROC.','roc_auc'),('评分指标为宏平均F1。','macro_f1'),
                ('Primary metric:\nMacro-F1. Report accuracy for reference.','macro_f1'),
                ('Primary metric: macro-F1, with accuracy for reference.','macro_f1'),
                ('Report macro F1 rather than accuracy.','macro_f1'),
                ('Primary metric: macro F1 and accuracy is secondary.','macro_f1'),
                ('Do not report test accuracy. Primary metric: weighted F1.','weighted_f1')]:
            with self.subTest(text=text):
                self.assertEqual(declared_metric(text),expected)
                p=fixture(self.root,prose=text,legacy=True)
                for t in (load_training_task(p),load_training_task(p,read_declarations=True),load_development_task(p)):
                    self.assertEqual(MetricSpec.for_task(t).name,expected)
                    self.assertEqual(t.spec['goal_sources']['primary_metric'],'declared_task_readme')
        for name,expected in [('Accuracy','accuracy'),('macro-F1','macro_f1'),('MACRO_F1','macro_f1'),
                ('AUC','roc_auc'),('ROC-AUC','roc_auc'),('F1','binary_f1'),('f1','binary_f1')]:
            t=load_training_task(fixture(self.root,metric=name),read_declarations=True)
            self.assertEqual(TaskSpec.from_task(t).metric.name,expected)

    def test_missing_manifest_metric_uses_readme_and_default_is_visible(self):
        t=load_training_task(fixture(self.root,prose='Report macro F1.'),read_declarations=True)
        self.assertEqual((MetricSpec.for_task(t).name,t.spec['goal_sources']['primary_metric']),('macro_f1','declared_task_readme'))
        t=load_training_task(fixture(self.root),read_declarations=True)
        g=TaskSpec.from_task(t).goal_resolution['fields']['primary_metric']
        self.assertEqual((g['value'],g['status']),('accuracy','unknown'))

    def test_manifest_primary_fields_and_ambiguous_metric_lists(self):
        from superagent.metric_declarations import manifest_metric
        examples = [({'primary_metric':'Macro-F1'},'macro_f1'),
                    ({'evaluationMetric':{'name':'F1','average':'weighted'}},'weighted_f1'),
                    ({'metrics':{'primary':'AUC','secondary':'accuracy'}},'roc_auc')]
        for manifest,expected in examples:
            self.assertEqual(manifest_metric(manifest),expected)
        for manifest in ({'metrics':['AUC','accuracy']},{'metric':'accuracy','primaryMetric':'macro-F1'}):
            with self.assertRaises(MetricDeclarationError):
                manifest_metric(manifest)
        self.assertIsNone(declared_metric('For example, report macro F1.'))

    def test_shared_table_preserves_the_readme_metric_and_training_rows(self):
        path=self.root/'shared.zip'
        manifest=dict(schema_version=1,modality='tabular',target='label',split_column='partition',
                      columns=[{'name':'signal','type':'numeric'}],
                      splits={role:{'format':'csv','path':'rows.csv'} for role in ('train','test')})
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('manifest.json',json.dumps(manifest))
            z.writestr('README.md','The primary metric is macro F1.')
            z.writestr('rows.csv','signal,partition,label\n0,train,A\n1,train,B\n2,test,DO_NOT_TRAIN\n')
        task=load_training_task(path,read_declarations=True)
        self.assertEqual(task.y_train.tolist(),['A','B'])
        self.assertEqual((MetricSpec.for_task(task).name,task.spec['goal_sources']['primary_metric']),
                         ('macro_f1','declared_task_readme'))

    def test_instructions_in_utf16_and_held_out_documents(self):
        path=self.root/'package.zip'
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('train.csv','signal,label\n2,a\n1,b\n')
            z.writestr('instructions.txt','Primary metric: macro-F1.'.encode('utf-16'))
            z.writestr('验证/README.md','Primary metric: accuracy.')
        original=zipfile.ZipFile.read
        def read(archive,name,*args,**kwargs):
            self.assertNotIn('验证/',str(name))
            return original(archive,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'read',read):
            for flag in (True,False):
                task=load_training_task(path,read_declarations=flag)
                self.assertEqual(MetricSpec.for_task(task).name,'macro_f1')

    def test_unresolved_metrics_never_rank_models_by_accuracy(self):
        for text in ('Evaluation metric: Matthews correlation coefficient.','Report test accuracy. Report F1.',
                     'Submissions are ranked by the Brier score.','Primary metric: PR-AUC.'):
            with self.subTest(text=text),self.assertRaises(MetricDeclarationError):
                declared_metric(text)
        p=fixture(self.root,metric='F1',labels=('red','green','blue'))
        run=self.root/'run'
        self.assertEqual(train(p,run,fit_limit=2),0)
        outcome=json.loads((run/'run_outcome.json').read_text())
        self.assertEqual((outcome['status'],outcome['reason_code'],outcome['attempted_classifier_fits']),('partial','metric_unresolved',0))
        self.assertIsNone(outcome['selection_metric'])
        self.assertEqual(json.loads((run/'task_spec.json').read_text())['metric'],'unresolved_f1')
        dest=create_predictions(p,run,self.root/'out'/'pred.csv')
        self.assertEqual(len(dest.read_text().splitlines()),5)
        t=load_training_task(fixture(self.root,metric='F1',labels=('no','yes'),positive='yes'))
        self.assertEqual(MetricSpec.for_task(t).record()['name'],'binary_f1')
        t=load_training_task(fixture(self.root,metric='F1',labels=('no','yes')))
        with self.assertRaises(MetricDeclarationError):MetricSpec.for_task(t)

    def test_primary_metric_changes_preference_for_same_predictions(self):
        from scripts.accept_metric_recovery import preference_probe
        result = preference_probe()
        self.assertEqual(result['accuracy']['selected'],'majority_only')
        self.assertEqual(result['macro_f1']['selected'],'finds_minority')
        self.assertTrue(all(r['action']=='stop' for r in result.values()))
        self.assertEqual(result['accuracy']['scores'],{'majority_only':.9,'finds_minority':.8})
        self.assertAlmostEqual(result['macro_f1']['scores']['finds_minority'],.6875)

    def test_an_unsupported_metric_keeps_independently_verified_class_predictions(self):
        for i,(metric,prose) in enumerate([('Matthews correlation coefficient',''),
                        (None,'Report test accuracy. Report macro F1.')]):
            with self.subTest(metric=metric):
                path=fixture(self.root,metric=metric,prose=prose)
                run=self.root/f'run{i}'
                self.assertEqual(train(path,run,fit_limit=2),0)
                outcome=json.loads((run/'run_outcome.json').read_text())
                self.assertEqual((outcome['status'],outcome['reason_code'],outcome['attempted_classifier_fits']),
                                 ('partial','metric_unresolved',0))
                self.assertEqual(json.loads((run/'task_spec.json').read_text())['metric'],'unresolved')
                dest=create_predictions(path,run,self.root/f'out{i}'/'pred.csv')
                self.assertEqual(len(dest.read_text().splitlines()),5)

    def test_a_stream_of_heartbeats_cannot_consume_the_experiment_window(self):
        from scripts.accept_record_formats import simulated_client
        from superagent.events import RunLogger
        client,response=simulated_client(self.root)
        log=RunLogger(self.root)
        log.initialize_usage_ledger()
        now=[1000.]
        client.set_deadline(now[0]+1200)
        def heartbeats():
            while now[0] < 1417:
                now[0] += 1
                yield b': keepalive'
        response.iter_lines.side_effect=heartbeats
        with patch('superagent.experiment_llm.time.monotonic',new=lambda:now[0]),patch('requests.Session.post',return_value=response) as post:
            with self.assertRaisesRegex(RuntimeError,'further batch calls stopped'):
                client.ask({},log,'experiment_planner')
            self.assertEqual(now[0],1045.)
            with self.assertRaises(RuntimeError):
                client.ask({},log,'experiment_planner')
            self.assertEqual(post.call_count,1)
        entry=json.loads(log.usage_path.read_text().splitlines()[-1])
        self.assertIsNone(entry['course_cost_rmb'])
        self.assertFalse(entry['usage_known'])

    def test_no_verified_worker_still_delivers_training_prior(self):
        from superagent.ab_run import Executor
        p=fixture(self.root,metric='AUC');run=self.root/'run'
        with patch.object(Executor,'worker',side_effect=TimeoutError('controlled pre-fit timeout')):
            self.assertEqual(train(p,run,fit_limit=2),0)
        outcome=json.loads((run/'run_outcome.json').read_text())
        self.assertEqual((outcome['status'],outcome['method'],outcome['attempted_classifier_fits']),('partial','training_class_prior',0))
        dest=create_predictions(p,run,self.root/'out'/'pred.csv')
        self.assertEqual(dest.read_text().splitlines()[0],'row_id,prediction,score_0,score_1')
        receipt=json.loads((dest.parent/'prediction_manifest.json').read_text())
        self.assertFalse(receipt['contract_unresolved'])

    def test_torn_export_preserves_the_last_complete_file(self):
        from superagent.events import atomic_text
        output=self.root/'predictions.csv'
        output.write_text('row_id,prediction\n0,A\n')
        with self.assertRaises(OSError):
            with atomic_text(output,newline='') as stream:
                stream.write('row_id,prediction\n')
                raise OSError('interrupted export')
        self.assertEqual(output.read_text(),'row_id,prediction\n0,A\n')

    def test_partition_failure_keeps_the_already_known_prediction_contract(self):
        p=fixture(self.root,metric='macro-F1')
        with patch('superagent.autonomous_run.partitions',side_effect=ValueError('No independent training groups')):
            self.assertEqual(train(p,self.root/'run',fit_limit=2),0)
        outcome=json.loads((self.root/'run'/'run_outcome.json').read_text())
        self.assertEqual((outcome['method'],outcome['attempted_classifier_fits']),('training_class_prior',0))
        self.assertIn('independent',outcome['error'])
        dest=create_predictions(p,self.root/'run',self.root/'out'/'p.csv')
        self.assertEqual(len(dest.read_text().splitlines()),5)
