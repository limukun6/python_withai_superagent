import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.cli import build_parser
from superagent.unified import categorical_opening, training_input


class UnifiedPlannerTests(unittest.TestCase):
    def test_same_command_configuration_for_anonymous_inputs(self):
        parsed=[]
        with patch('superagent.ab_run.command', side_effect=lambda args: parsed.append(vars(args))):
            for label in ('a.zip','xyz.zip'):
                args=build_parser().parse_args(['unified',label,'--run-dir','x'])
                args.handler(args)
        exclude={'task_package','handler'}
        self.assertEqual([{k:v for k,v in obj.items() if k not in exclude} for obj in parsed][0],
                         {k:v for k,v in parsed[1].items() if k not in exclude})
        self.assertEqual(parsed[0]['fit_limit'],10)

    def test_raw_archive_only_opens_training_members(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'unknown.zip'
            with zipfile.ZipFile(path,'w') as z:
                for key,arr in (('C_train',np.array([['x'],['y']])),('y_train',np.array([0,1]))):
                    buf=io.BytesIO()
                    np.save(buf,arr)
                    z.writestr(key+'.npy',buf.getvalue())
                for key in ('C_val','y_val','C_test','y_test'):
                    z.writestr(key+'.npy',b'outside-the-training-contract')
            original=zipfile.ZipFile.open
            opened=[]
            def observe(archive,name,*args,**kwargs):
                opened.append(name)
                return original(archive,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'open',observe):
                task=training_input(path)
            self.assertEqual(set(opened),{'C_train.npy','y_train.npy'})
            self.assertIsNone(task.X_val)
            self.assertIsNone(task.y_val)

    def test_matched_depth_respects_training_evidence_and_eligibility(self):
        def cap(d,l):
            return dict(params={'depth':d,'l2_leaf_reg':l},estimator='tree',representation='native categories',resource='cpu')
        caps={'small':cap(2,2),'mid':cap(5,2),'large':cap(9,2),'stronger penalty':cap(9,8)}
        profile=dict(statistics={'train_rows':500},majority_class_fraction=.55)
        state=dict(task={'kind':'categorical_tabular'},training_profile=profile,prediction_evidence=[])
        self.assertEqual(categorical_opening(state,caps,tuple(reversed(caps)),[])['chosen'],'small')
        state['prediction_evidence']=[{'trial':1,'predictions':[0,1]}]
        first=dict(candidate='small',trial=1,search_score=.7,train_score=.8)
        options=('stronger penalty','mid','large')
        self.assertEqual(categorical_opening(state,caps,options,[first])['chosen'],'large')
        self.assertEqual(categorical_opening(state,caps,('mid','stronger penalty'),[first])['chosen'],'mid')
        self.assertNotIn('coverage_rule',categorical_opening(state,caps,options,[dict(first,train_score=1)]))
