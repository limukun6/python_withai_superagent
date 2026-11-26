import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.ab_catalog import capabilities_for
from superagent.ab_run import Executor
from superagent.autonomous_run import train
from superagent.contracts import TaskSpec
from superagent.data import load_prediction_features, load_training_task
from superagent.events import RunLogger
from superagent.experiment_policy import actions
from superagent.format_intake import adapt_training
from superagent.open_methods import MethodSpace
from superagent.task_constraints import RestrictionError, documented_flags, permitted
from superagent.task_declarations import DeclarationError
from superagent.task_profile import training_facts


def npy(value):
    stream=io.BytesIO()
    np.save(stream,value)
    return stream.getvalue()


class CitationClient:
    model='test-double'
    max_requests=1
    blocked=False

    def __init__(self,answer):
        self.answer=answer
        self.request_count=0

    def ask(self,packet,log,role):
        assert role=='contract_extraction'
        self.packet=packet
        self.request_count+=1
        return self.answer


class TaskDeclarations(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)

    def package(self,name,files):
        path=self.root/(name+'.zip')
        with zipfile.ZipFile(path,'w') as archive:
            for member,contents in files.items():
                archive.writestr(member,contents)
        return path

    def arrays(self,text,width=3):
        return self.package('anonymous',{'C_train.npy':npy(np.arange(90*width).reshape(90,width).astype(str)),
            'y_train.npy':npy(np.tile([0,1],45)),'README.md':text,
            'C_val.npy':b'not read','y_test.npy':b'not read'})

    def logger(self,name):
        folder=self.root/name
        folder.mkdir()
        return RunLogger(folder)

    def test_anonymous_array_round_trip_and_class_meanings(self):
        path=self.arrays('Predict whether equipment is needed ('+chr(96)+'1'+chr(96)+' = needed, 0 = idle).\n'
            'The fields, from left to right, are rotor, site and shift. Report macro F1.')
        seen=[]
        old=zipfile.ZipFile.read
        def guarded(archive,name,*a,**kw):
            self.assertNotIn(name,('C_val.npy','y_test.npy'))
            seen.append(name)
            return old(archive,name,*a,**kw)
        with patch.object(zipfile.ZipFile,'read',guarded):
            task=load_training_task(path,read_declarations=True)
        contract=TaskSpec.from_task(task).record()
        self.assertEqual([v['name'] for v in contract['input_schema']['columns']],['rotor','site','shift'])
        fields=contract['goal_resolution']['fields']
        self.assertEqual(fields['label_meanings']['value'],{'0':'idle','1':'needed'})
        self.assertEqual(fields['positive_label']['status'],'not_applicable')
        self.assertEqual(fields['split_roles']['value']['validation']['members'],['C_val.npy'])
        self.assertEqual(contract['metric']['name'],'macro_f1')
        self.assertEqual(training_facts(task)['documented_columns'],['rotor','site','shift'])
        self.assertEqual(set(seen),{'README.md','C_train.npy','y_train.npy'})
        for width in (3,2):
            query=self.package('query',{'C_test.npy':npy(np.array([['q']*width]))})
            if width==3:
                self.assertEqual(load_prediction_features(query,task.kind,task.spec).shape,(1,3))
            else:
                with self.assertRaisesRegex(ValueError,'dimensions'):
                    load_prediction_features(query,task.kind,task.spec)

    def test_mapped_positive_class_is_used_for_binary_f1(self):
        path=self.arrays('Labels: 0 = urgent, 1 = routine. The positive class is urgent. Report binary F1.')
        task=load_training_task(path,read_declarations=True)
        contract=TaskSpec.from_task(task)
        self.assertEqual(contract.metric.labels,(1,0))
        self.assertEqual(contract.goal_resolution['fields']['positive_label']['source'],'declared_task_readme')
        self.assertEqual(contract.metric.evaluate([0,1,1],[0,1,0])['score'],2/3)

    def test_named_recall_is_scored_for_the_declared_class(self):
        for request in ('Report recall for urgent.','Report the urgent class recall.','报告 urgent 类的召回率。'):
            with self.subTest(request=request):
                path=self.arrays('Labels: 0 = urgent, 1 = routine. '+request)
                task=load_training_task(path,read_declarations=True)
                metric=TaskSpec.from_task(task).metric
                self.assertEqual((metric.name,metric.labels),('binary_recall',(1,0)))
                result=metric.evaluate([0,0,1,1],[1,0,1,1])
                self.assertEqual((result['score'],result['accuracy']),(.5,.75))
        task=load_training_task(self.arrays('Primary metric: accuracy. Labels: 0 = urgent, 1 = routine. '
            'Also report recall for urgent.'),read_declarations=True)
        self.assertEqual(TaskSpec.from_task(task).metric.name,'accuracy')
        self.assertNotIn('positive_label',task.spec)

    def test_invalid_or_conflicting_descriptions(self):
        for prose in ('The columns in order are a,b.', 'The columns in order are a,a,c.',
                      'The columns in order are a,b,c. The fields in order are c,b,a.',
                      'Labels: 0 = active, 2 = idle.',
                      'Labels: 0 = active, 1 = idle. The positive class is missing.',
                      'Labels: 0 = active, 1 = idle. Labels: 0 = idle, 1 = active.'):
            with self.subTest(prose=prose),self.assertRaises(DeclarationError):
                load_training_task(self.arrays(prose),read_declarations=True)
        task=load_training_task(self.arrays('Column names: a, b, c.\nLabel meanings:\n- 0 = idle\n- 1 = open'),read_declarations=True)
        self.assertEqual(task.spec['declaration_evidence']['fields']['label_meanings']['value'],{'0':'idle','1':'open'})

    def test_manifest_and_prose_restrictions_shared_table(self):
        manifest=dict(schema_version=1,modality='tabular',target='result',metric='Accuracy',split_column='part',
            restrictions={'linear_only':True},columns=[dict(name='value',type='numeric')],
            splits={role:dict(path='rows.csv',format='csv') for role in ('train','test')})
        files={'manifest.json':json.dumps(manifest),'README.md':'No pretrained models. Labels: 0 = idle, 1 = busy.',
            'rows.csv':'value,result,part\n5,0,train\n6,1,train\n7,HIDDEN,test\n'}
        task=load_training_task(self.package('mixed',files),read_declarations=True)
        record=TaskSpec.from_task(task).record()
        self.assertEqual(record['method_restrictions'],['linear_only','no_pretrained_assets'])
        self.assertEqual(record['goal_resolution']['fields']['label_meanings']['value']['0'],'idle')
        self.assertEqual(task.y_train.tolist(),['0','1'])
        for rules in ({'random_only':True},{'linear_only':'false'},{'no_pretrained_assets':False}):
            files['manifest.json']=json.dumps(dict(manifest,restrictions=rules))
            with self.subTest(rules=rules),self.assertRaises(RestrictionError):
                load_training_task(self.package('bad',files),read_declarations=True)

    def test_constrained_static_generated_menus_worker_and_votes(self):
        task=load_training_task(self.arrays('Only linear classifiers are permitted. No external data or pretrained weights.'),read_declarations=True)
        contract=TaskSpec.from_task(task)
        profile=training_facts(task)
        caps,why=MethodSpace(contract.record(),profile,self.root).refresh([],capabilities_for(contract.record(),train_rows=90))
        self.assertTrue(caps)
        self.assertTrue(why['evidence']['excluded_by_task_restrictions'])
        self.assertTrue(all(permitted(cap,profile['method_restrictions']) for cap in caps.values()))
        self.assertFalse(any(cap.get('estimator') in ('catboost','extra_trees','rbf_svm') for cap in caps.values()))
        names=list(caps)[:2]
        obs={n:dict(score=.75,row_key='aligned',labels=np.array([0,1]),probabilities=np.array([[.6,.4],[.4,.6]])) for n in names}
        menu=actions(caps,profile,[],dict(primary=obs,confirmation_1=obs),contract.metric,['primary','confirmation_1'],6)
        self.assertTrue(menu)
        self.assertTrue(all(len(action['members'])==1 and len(action['evidence'].get('target_members',[]))<=1 for action in menu))
        forbidden=dict(origin='b_port',estimator='logistic',params={'encoder':'frozen'},resource='local_frozen_encoder_cpu')
        worker=Executor(self.root,{'not_admitted':forbidden},contract.metric,0,30,30,3,None,restrictions=['no_pretrained_assets'])
        with self.assertRaisesRegex(ValueError,'outside the frozen'):
            worker.worker(None,'not_admitted',None,3)
        self.assertEqual(worker.accounting()['attempted_classifier_fits'],0)

    def test_uncertainty_does_not_authorize_methods(self):
        self.assertEqual(documented_flags('Pretrained models are allowed.'),{})
        self.assertEqual(documented_flags('For example, only use linear classifiers.'),{})
        self.assertEqual(set(documented_flags('不得使用外部数据。不要使用预训练模型。只允许线性模型。')),
            {'no_external_data','no_pretrained_assets','linear_only'})
        with self.assertRaises(RestrictionError):
            documented_flags('Use only decision trees.')
        with self.assertRaises(RestrictionError):
            documented_flags('No external data and no neural networks.')
        self.assertEqual(set(documented_flags('Allowed classifiers: linear classifiers.')),{'linear_only'})
        with self.assertRaises(RestrictionError):
            documented_flags('Permitted methods: naive Bayes.')
        path=self.arrays('Only random forests are permitted.')
        self.assertEqual(train(path,self.root/'unresolved',fit_limit=2),2)
        data=json.loads((self.root/'unresolved/run_outcome.json').read_text())
        self.assertEqual((data['reason_code'],data['attempted_classifier_fits']),('method_restriction_unresolved',0))

    def test_neutral_role_names_and_no_heldout_content_read(self):
        path=self.package('renamed',{'README.md':'alpha.csv contains training examples. omega.csv is reserved for final evaluation. Report Accuracy.',
            'alpha.csv':'x,label\n0,a\n1,b\n','omega.csv':b'never read'})
        old=zipfile.ZipFile.read
        def guarded(archive,name,*args,**kwargs):
            self.assertNotEqual(name,'omega.csv')
            return old(archive,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'read',guarded):
            task=load_training_task(path,read_declarations=True)
        self.assertEqual(task.spec['prediction_member'],'omega.csv')
        self.assertEqual(task.spec['split_roles']['test']['members'],['omega.csv'])
        p=self.package('only_prediction',{'README.md':'omega.jsonl is reserved for final evaluation.',
                                          'omega.jsonl':'{"label":"a","x":1}\n{"label":"b","x":2}'})
        with self.assertRaises(ValueError):
            adapt_training(p)

    def test_evidence_only_cached_once_and_revalidated(self):
        note='Instrument coordinate layout follows the sockets: east, west, height.'
        client=CitationClient({'evidence':0})
        for run in ('first','again'):
            task=load_training_task(self.arrays(note),read_declarations=True,declaration_client=client,log=self.logger(run))
            claim=task.spec['declaration_evidence']['fields']['columns']
            self.assertEqual((claim['value'],claim['status']),(['east','west','height'],'proposed'))
        self.assertEqual(client.request_count,1)
        self.assertEqual(client.packet['quotations'],[note.rstrip('.')])
        for i,answer in enumerate(({'evidence':None},{'evidence':9},{'evidence':0,'method':'tree'})):
            bad=CitationClient(answer)
            task=load_training_task(self.arrays(note+' Version '+str(i)),read_declarations=True,declaration_client=bad,log=self.logger(str(i)))
            self.assertFalse(task.spec.get('input_schema',{}).get('columns'))
            self.assertEqual(bad.request_count,1)
        other=CitationClient({'evidence':0})
        task=load_training_task(self.arrays(note,width=2),read_declarations=True,declaration_client=other,log=self.logger('different'))
        self.assertEqual(other.request_count,0)
        self.assertFalse(task.spec.get('input_schema',{}).get('columns'))

    def test_formal_solve_obeys_the_rule_and_reloads(self):
        raw='signal,position,label\n'+''.join(f'{i%13},{i},{int(i%13>6)}\n' for i in range(96))
        note='Use only linear classifiers. No external data or pretrained encoders.\nLabels: 0 = idle, 1 = moving. Report recall for idle.'
        training=self.package('anonymous',{'README.md':note,'train.csv':raw})
        query=self.package('features',{'test.csv':'signal,position\n2,100\n9,101\n8,102\n'})
        environment={k:v for k,v in os.environ.items() if k not in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER',
                     'SUPERAGENT_TASK_BUDGET_ID','SUPERAGENT_RESEARCH_REFLECTION_OUTBOX')}
        environment['SUPERAGENT_OFFLINE']='1'
        run=self.root/'course'
        for action in ('solve','predict'):
            args=[sys.executable,'-m','unified_agent',action,str(training if action=='solve' else query),
                  '--run-dir',str(run),'--output',str(self.root/action/'result.csv')]
            if action=='solve':
                args.extend(['--prediction-input',str(query),'--fit-limit','4','--time-limit','150'])
            result=subprocess.run(args,cwd=Path(__file__).resolve().parents[1],env=environment,stdin=subprocess.DEVNULL,
                                  text=True,capture_output=True,timeout=200)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual((self.root/'solve/result.csv').read_bytes(),(self.root/'predict/result.csv').read_bytes())
        def read(name):
            return json.loads((run/'agent'/name).read_text())
        outcome=read('run_outcome.json')
        self.assertEqual((outcome['status'],outcome['new_api_requests']),('completed',0))
        self.assertLessEqual(outcome['attempted_classifier_fits'],4)
        self.assertEqual(read('task_contract.json')['metric'],dict(name='binary_recall',labels=['1','0'],direction='maximize'))
        self.assertEqual(read('final_selection.json')['ensemble_components'],[])
        self.assertTrue(all(len(a['members'])==1 for d in read('decisions.json') for a in d['alternatives']))
        self.assertTrue(all(any(row['meaning']=='idle' for row in r['documented_class_recall'])
                            for r in read('history.json') if r['status']=='completed'))
