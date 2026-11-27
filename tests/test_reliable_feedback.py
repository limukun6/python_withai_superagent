import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.feedback_hypothesis import LessonStore, validate_feedback
from superagent.metric_declarations import declared_metric, MetricDeclarationError
from superagent.worked_metrics import scores

METRICS=['accuracy','micro_f1','balanced_accuracy','macro_f1','weighted_f1']
CASES=[([0,0,0,0,0,1,1,1,2,2],[0,0,0,1,1,0,1,2,2,2]),
       ([0,0,0,0,1,1,2,2,2,2,2],[0,0,0,2,1,2,1,1,2,2,0])]


def examples(metric='macro_f1'):
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    reference={'macro_f1':lambda y,p:f1_score(y,p,average='macro'),
               'balanced_accuracy':balanced_accuracy_score,'accuracy':accuracy_score}
    return dict(candidates=METRICS,labels=[0,1,2],rounding='half_up',
                examples=[dict(truth=list(y),prediction=list(p),score=f'{reference[metric](y,p):.2f}') for y,p in CASES])


def fixture(path, metric=None, evidence=None, doc=None, records=False):
    source='training.jsonl' if records else 'training.csv'
    manifest=dict(schema_version=1,modality='tabular',target='outcome',columns=[dict(name='f0',type='numeric')],
                  splits={'train':dict(format='jsonl' if records else 'csv',path=source)})
    if evidence is not None: manifest['metric_evidence']=evidence
    if metric is not None: manifest['metric']=metric
    rows=[dict(f0=i%11,outcome=i%3) for i in range(120)]
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest))
        z.writestr(source,('\n'.join(json.dumps(r) for r in rows) if records else 'f0,outcome\n'+
                           ''.join(f"{r['f0']},{r['outcome']}\n" for r in rows)))
        if doc is not None:z.writestr('README.md',doc)


class WorkedContract(unittest.TestCase):
    def test_rounded_fenced_and_manifest_inputs(self):
        card=examples()
        note='Primary metric is the recognition index illustrated below.\n## Scoring examples\n```json\n'+json.dumps(card)+'\n```'
        with tempfile.TemporaryDirectory() as tmp:
            for i,(manifest,doc,kind) in enumerate([(None,note,False),(card,None,False),(card,note,True)]):
                p=Path(tmp)/f'p{i}.zip';fixture(p,evidence=manifest,doc=doc,records=kind)
                if kind:
                    from superagent.format_intake import adapt_training
                    t=adapt_training(p)
                else:t=load_training_task(p,read_declarations=True)
                field=TaskSpec.from_task(t).record()['goal_resolution']['fields']['primary_metric']
                self.assertEqual((field['value'],field['source'],field['evidence']['examples']),
                                 ('macro_f1','verified_worked_examples',2))

    def test_equivalent_ambiguous_conflicting_values(self):
        card=examples('accuracy')
        with self.assertRaises(MetricDeclarationError): declared_metric('',evidence=card)
        # An independently named metric resolves otherwise indistinguishable accuracy and micro-F1.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.zip';fixture(p,metric='micro-F1',evidence=card)
            spec=TaskSpec.from_task(load_training_task(p,read_declarations=True))
            self.assertEqual(spec.metric.name,'micro_f1')
        for prose in ('Primary metric is accuracy.','Primary metric is log loss.','Primary metric is precision.'):
            with self.assertRaises(MetricDeclarationError):
                declared_metric(prose,evidence=examples())
        self.assertEqual(declared_metric('Primary metric is the harmonic mean of precision and recall for each class.',
                                         evidence=examples()),'macro_f1')
        card=examples();card['examples'][0]['score']=.60
        with self.assertRaises(MetricDeclarationError): declared_metric('',evidence=card)
        card=examples();card['examples'][0]['truth'][0:0]=[4]
        with self.assertRaises(MetricDeclarationError): declared_metric('',evidence=card)
        with self.assertRaises(MetricDeclarationError): declared_metric('Scoring examples:\n{broken')

    def test_unresolved_examples_still_export_honest_partial_predictions(self):
        environment = dict(os.environ, SUPERAGENT_OFFLINE='1')
        for name in ('SUPERAGENT_PLANNER_KEY_FILE', 'SUPERAGENT_PLANNER_LEDGER',
                     'SUPERAGENT_TASK_BUDGET_ID', 'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX'):
            environment.pop(name, None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root/'features.zip'
            schema = dict(schema_version=1, modality='tabular', target='outcome',
                          columns=[dict(name='f0', type='numeric')],
                          splits={'test':dict(format='csv', path='data.csv')})
            with zipfile.ZipFile(features, 'w') as archive:
                archive.writestr('manifest.json', json.dumps(schema))
                archive.writestr('data.csv', 'f0\n' + '\n'.join(str(v) for v in range(12)))
            cases = [('conflict', 'accuracy', examples()), ('undetermined', None, examples('accuracy'))]
            for tag, explicit, card in cases:
                with self.subTest(case=tag):
                    training = root/(tag+'.zip')
                    fixture(training, metric=explicit, evidence=card)
                    run, output = root/tag, root/(tag+'-delivery')/'predictions.csv'
                    subprocess.run([sys.executable, '-m', 'unified_agent', 'solve', str(training),
                                    '--prediction-input', str(features), '--run-dir', str(run),
                                    '--output', str(output), '--fit-limit', '5'], env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   check=True, timeout=60)
                    outcome = json.loads((run/'agent'/'run_outcome.json').read_text())
                    receipt = json.loads((output.parent/'prediction_manifest.json').read_text())
                    self.assertEqual((outcome['status'], outcome['attempted_classifier_fits'],
                                      outcome['new_api_requests']), ('partial', 0, 0))
                    self.assertIn('MetricDeclarationError', outcome['reason'])
                    self.assertEqual((receipt['delivery_status'], receipt['prediction_count']), ('partial', 12))
                    with output.open(newline='') as stream:
                        self.assertEqual([int(row['row_id']) for row in csv.DictReader(stream)], list(range(12)))

    def test_exact_math_matches_worker_and_score_printing(self):
        import numpy as np
        from superagent.metrics import MetricSpec
        rng=np.random.default_rng(782)
        for k in range(2,8):
            labels=list(range(k))
            for _ in range(7):
                truth=list(range(k))+rng.integers(k,size=31).tolist()
                pred=rng.integers(k,size=len(truth)).tolist()
                for n,v in scores(truth,pred,labels).items():
                    self.assertAlmostEqual(float(v),MetricSpec(n,tuple(labels)).evaluate(truth,pred)['score'])
        card=dict(candidates=['macro_f1','accuracy'],labels=[0,1],examples=[dict(truth=[0]*7+[1],prediction=[0]*8,score='0.87')])
        # 0.875 is on the boundary when the convention is unspecified. Half-up prints 0.88.
        self.assertEqual(declared_metric('Report accuracy.',evidence=card),'accuracy')
        card['rounding']='half_up'
        with self.assertRaises(MetricDeclarationError):declared_metric('Report accuracy.',evidence=card)


class FirstExport(unittest.TestCase):
    def test_input_and_immutable_model_storage_are_protected(self):
        from superagent.first_delivery import guard_output
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, features = root/'run', root/'data.zip'
            snapshot = run/'first_delivery'
            model = snapshot/'agent'
            model.mkdir(parents=True)
            (model/'final_selection.json').write_text('{}')
            (snapshot/'first_delivery.json').write_text('{}')
            for active, collision in [(run,features), (run,model/'x'), (snapshot,snapshot/'predictions.csv'),
                                      (run,root/'delivery'/'prediction_manifest.json')]:
                with self.assertRaises(ValueError):
                    guard_output(collision, active, (features,))
            guard_output(root/'delivery'/'predictions.csv', snapshot, (features,))


class TransferableChecks(unittest.TestCase):
    def test_two_sources_and_measured_trial_are_required(self):
        brief={'facts':{'contract':{'metric':'macro_f1'},'doc_1':'generic task','balance':{},'length':{},'first_trial':{}},
               'checks':{'rare':dict(kind='class',source='balance'),'short':dict(kind='slice',source='length')}}
        answer=dict(goal='predict classes',metric='macro_f1',citations=['doc_1','balance','length','first_trial'],
                    uncertainty='joint errors',check={'all_of':['short','rare']},hypothesis='more errors',if_false='retain')
        self.assertEqual(validate_feedback(answer,brief)['check'],'rare&short')
        for change in [dict(citations=['doc_1','balance','length']),dict(citations=['doc_1','first_trial','length']),
                       dict(check={'all_of':['short','short']}),dict(check={'all_of':['missing','rare']})]:
            with self.assertRaises(ValueError):validate_feedback(dict(answer,**change),brief)

    def test_prior_task_is_not_counted_twice_and_only_compatible_checks_travel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lessons.sqlite'
            old=LessonStore(path,'a','single_text','macro_f1')
            self.assertTrue(old.record(['a','b'],'deferred'))
            self.assertFalse(old.record(['b','a'],'deferred'))
            keys={'a':{},'b':{}}
            self.assertEqual(old.hints(keys),[])
            with patch.dict('os.environ',{'SUPERAGENT_LESSONS_READONLY':'1'}):
                new=LessonStore(path,'b','single_text','macro_f1')
                self.assertEqual(new.hints(keys),[dict(terms=['a','b'],outcomes={'deferred':1})])
                self.assertFalse(new.record(['a','b'],'supported'))
            self.assertEqual(new.hints({'a':{}}),[])
            self.assertEqual(LessonStore(path,'b','image','macro_f1').hints(keys),[])


if __name__=='__main__':
    unittest.main()
