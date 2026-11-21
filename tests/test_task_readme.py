"""Conservative legacy declarations and manifest precedence; no training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from superagent.data import load_training_task,load_development_task,load_prediction_features
from superagent.contracts import TaskSpec


PAIR='The target label_pairwise is 1 if A had the higher rate and 0 if B did.'


class TaskReadmeTests(unittest.TestCase):
    def package(self,root,text,extra_readme=False,manifest=False):
        path=root/'anonymous.zip'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('train.csv','headline_A,headline_B,label_pairwise\na,b,0\nc,d,1\n')
            archive.writestr('README.md',text)
            archive.writestr('__MACOSX/._README.md',b'\xf5\x00')
            archive.writestr('test.csv',b'NEVER OPEN')
            if extra_readme:archive.writestr('other/readme.txt','report test accuracy.')
            if manifest:
                archive.writestr('manifest.json',json.dumps(dict(schema_version=1,objective='classification',
                    modality='pair_text',metric='macro_f1',pair_semantics='symmetric',pair_label_one='equivalent',
                    columns=[dict(name=n,type='text') for n in ('headline_A','headline_B')],target='label_pairwise',
                    splits={'train':dict(path='train.csv',format='csv')})))
        return path

    def test_explicit_declarations_have_same_provenance_in_both_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.package(Path(tmp),PAIR+' Use the test set only for final evaluation and report **test accuracy**.')
            values=[load_training_task(path),load_development_task(path)]
            for task in values:
                spec=TaskSpec.from_task(task).record()
                self.assertEqual(spec['metric']['name'],'accuracy')
                self.assertEqual(spec['input_schema']['pair_label_one'],'left')
                self.assertEqual(spec['input_schema']['pair_semantics'],'preference')
                self.assertEqual(spec['goal_resolution']['fields']['primary_metric']['status'],'confirmed')
                self.assertEqual(task.spec['goal_sources']['primary_metric'],'declared_task_readme')
                self.assertEqual(len(task.spec['task_description_source']['sha256']),64)

    def test_negation_conflict_and_ambiguous_sources_stay_unknown(self):
        cases=[('Do not report test accuracy.',False),('No requirement to report test accuracy.',False),
               ('Should we report test accuracy?',False),('We might report test accuracy.',False),
               ('Report test accuracy. Report F1.',False),
               ('report test accuracy.',True)]
        for text,multiple in cases:
            with self.subTest(text=text),tempfile.TemporaryDirectory() as tmp:
                task=load_training_task(self.package(Path(tmp),text,multiple))
                goal=TaskSpec.from_task(task).record()['goal_resolution']['fields']['primary_metric']
                self.assertEqual(goal['status'],'unknown')

    def test_reverse_or_negated_preference_is_not_guessed(self):
        for text in ('The target label_pairwise is 0 if A had the higher rate and 1 if B did.',
                     'It is not true that '+PAIR,PAIR+' The target label_pairwise is 0 if A wins.'):
            with self.subTest(text=text),tempfile.TemporaryDirectory() as tmp:
                task=load_training_task(self.package(Path(tmp),text))
                self.assertNotIn('pair_semantics',task.spec.get('input_schema',{}))

    def test_manifest_declarations_take_precedence_over_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=load_training_task(self.package(Path(tmp),PAIR+' Report test accuracy.',manifest=True))
            self.assertEqual(task.spec['metric'],'macro_f1')
            self.assertEqual(task.spec['input_schema']['pair_semantics'],'symmetric')
            self.assertNotIn('task_description_source',task.spec)

    def test_only_training_and_document_bytes_are_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=self.package(Path(tmp),PAIR+' Report test accuracy.');opened=[];original=zipfile.ZipFile.read
            def guarded(archive,name,*args,**kwargs):
                value=name.filename if isinstance(name,zipfile.ZipInfo) else name
                self.assertNotIn('test.csv',value);opened.append(value)
                return original(archive,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',new=guarded):load_training_task(path)
            self.assertEqual(set(opened),{'train.csv','README.md'})

    def test_legacy_declaration_freezes_complete_label_free_prediction_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=load_training_task(self.package(root,PAIR+' Report test accuracy.'))
            prediction=root/'features.zip'
            with zipfile.ZipFile(prediction,'w') as archive:
                archive.writestr('test.csv','headline_A,headline_B\nnew left,new right\n')
            self.assertEqual(task.spec['input_format'],'csv')
            values=load_prediction_features(prediction,'pair_text',task.spec)
            self.assertEqual(values,[('new left','new right')])


if __name__=='__main__':unittest.main()
