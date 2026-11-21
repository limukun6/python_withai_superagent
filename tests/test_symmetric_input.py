"""Data-only symmetric labels must not be interpreted as left-side preference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from superagent.ab_catalog import capabilities_for
from superagent.cli import _run,build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_development_task


class SymmetricInputTests(unittest.TestCase):
    def package(self,root,meaning='equivalent',semantics='symmetric'):
        path=root/'task.zip'
        manifest=dict(schema_version=1,objective='classification',modality='pair_text',metric='accuracy',
            pair_semantics=semantics,pair_label_one=meaning,target='label',
            columns=[dict(name='a',type='text'),dict(name='b',type='text')],
            splits={'train':dict(format='csv',path='train.csv')})
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('manifest.json',json.dumps(manifest))
            archive.writestr('train.csv','a,b,label\na text,b text,0\nc text,d text,1\n')
        return path

    def test_explicit_equivalence_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=load_development_task(self.package(root))
            spec=TaskSpec.from_task(task).record()
            self.assertEqual(task.kind,'pair_text')
            self.assertEqual(task.y_train.tolist(),[0,1])
            self.assertEqual(spec['input_schema']['pair_label_one'],'equivalent')
            self.assertEqual(spec['goal_resolution']['fields']['pair_semantics']['value'],'symmetric')

    def test_conflicting_or_missing_label_meaning_refused(self):
        for meaning in ('left',None,'right'):
            with self.subTest(meaning=meaning),tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError,'pair_label_one=equivalent'):
                    load_development_task(self.package(Path(tmp),meaning))

    def test_existing_preference_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            task=load_development_task(self.package(Path(tmp),'left','preference'))
            self.assertEqual(task.spec['input_schema']['pair_label_one'],'left')

    def test_official_entry_refuses_empty_registry_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.package(root)
            args=build_parser().parse_args(['run',str(path),'--scheduler','ab-v1','--portfolio','ab-v1',
                '--run-dir',str(root/'run')])
            with patch('superagent.ab_run.capabilities_for', return_value={}), \
                    patch('superagent.ab_run.train_candidate') as worker,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_run(args),2)
                worker.assert_not_called()
            outcome=json.loads((root/'run/run_outcome.json').read_text(encoding='utf-8'))
            self.assertIn('No registered executable method',outcome['error'])
            self.assertEqual(outcome['classifier_fit_accounting']['attempted_classifier_fits'],0)


if __name__=='__main__':unittest.main()
