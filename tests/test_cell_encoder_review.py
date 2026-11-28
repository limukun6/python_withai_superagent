import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from superagent.ab_run import sha
from superagent.cell_encoder_review import after_delivery, contest, passes
from superagent.events import RunLogger
from superagent.prediction_evidence import make_prediction_record
from superagent.score_evidence import make_score_record
from superagent.cell_encoders import CellGeometryTree, SCRATCH, capability, resources
from superagent.metrics import MetricSpec
from superagent.models import SoftVoteModel
from superagent.task_constraints import permitted


class ProbabilityArray:
    def __init__(self, p, classes):
        self.classes_ = np.asarray(classes)
        self.p = np.asarray(p)

    def predict_proba(self, rows):
        return np.tile(self.p, (len(rows), 1))


class ComplementFixture:
    classes_ = np.array([0, 1])

    def __init__(self, weak):
        self.weak = weak

    def predict_proba(self, rows):
        index = np.asarray(rows)[:,0].astype(int)
        confidence = np.where(index%6 == self.weak, .48, .91)
        first = np.where(index%2 == 0, confidence, 1-confidence)
        return np.column_stack((first, 1-first))

    def predict(self, rows):
        return self.predict_proba(rows).argmax(1)


class CellEncoderReviewTests(unittest.TestCase):
    def test_selection_changes_when_training_feedback_changes(self):
        metric = MetricSpec('accuracy', (0, 1))
        y = np.array([0, 1, 0, 1])
        a = np.array([[.9,.1],[.1,.9],[.9,.1],[.1,.9]])
        wrong = a[:,::-1]
        self.assertEqual(contest(metric, y, a, {'a':wrong})[0]['encoders'], [])
        self.assertEqual(contest(metric, y, wrong, {'a':a})[0]['share'], 1.)
        left, right = a.copy(), a.copy()
        left[0], right[1] = [.49,.51], [.51,.49]
        best = contest(metric, y, wrong, {'first':left,'second':right})[0]
        self.assertEqual(best['encoders'], ['first','second'])
        self.assertEqual(best['share'], 1.)
        tied = contest(metric, y, a, {'alias':a})
        self.assertEqual(tied[0]['encoders'], [])

    def test_one_weak_confirmation_vetoes_an_improvement(self):
        def panel(gain, brier=-.01):
            return dict(rows=500, incumbent=dict(score=.7,brier=.45), challenger=dict(score=.7+gain,brier=.45+brier))
        self.assertTrue(passes([panel(.006),panel(.004)]))
        self.assertFalse(passes([panel(.04),panel(-.001)]))
        self.assertFalse(passes([panel(.02,.03),panel(.02,0)]))
        self.assertFalse(passes([panel(.001),panel(.001)]))

    def test_class_alignment_and_unknown_categories_survive_weighted_prediction(self):
        rows = np.array([['new']])
        first = ProbabilityArray([.9,.1],[0,1])
        second = ProbabilityArray([.8,.2],[1,0])
        vote = SoftVoteModel('categorical_tabular',[first,second],np.array([0,1]),(.25,.75))
        np.testing.assert_allclose(vote.predict_proba(rows), [[.375,.625]])
        tree = CellGeometryTree()
        tree.encoded = [({'seen':np.array([1.,2.])},2)]
        np.testing.assert_array_equal(tree.matrix(rows), np.array([['new',0.,0.]], dtype=object))
        with self.assertRaises(ValueError):
            SoftVoteModel('categorical_tabular',[first,second],np.array([0,1]),(-1,2)).predict_proba(rows)

    def test_positive_feedback_exports_the_elected_fusion(self):
        metric = MetricSpec('accuracy',(0,1))
        rows = np.column_stack((np.arange(360).astype(str), np.tile(['alpha','beta'],180)))
        y = np.arange(len(rows))%2
        task = SimpleNamespace(kind='categorical_tabular', X_train=rows, y_train=y)
        base, challenger = ComplementFixture(0), ComplementFixture(1)
        with tempfile.TemporaryDirectory() as temp, patch.dict('os.environ',{},clear=True):
            root = Path(temp)
            logger = RunLogger(root)
            joblib.dump({'model':base},root/'incumbent.joblib')
            selection = dict(family='prior',model_file='incumbent.joblib',model_sha256=sha(root/'incumbent.joblib'))
            choice = dict(members=['prior'])
            banks, payloads = {}, {}
            for fold, ix in zip(('primary','confirmation_1','confirmation_2'),np.array_split(np.arange(len(y)),3)):
                path = root/(fold+'.joblib')
                joblib.dump(dict(X_train=rows[~np.isin(np.arange(len(y)),ix)], X_search=rows[ix], y_search=y[ix]),path)
                payloads[fold] = dict(labels=y[ix],sha256=sha(path),path=path)
                banks[fold] = {'prior':dict(probabilities=base.predict_proba(rows[ix]),predictions=base.predict(rows[ix]),
                                            labels=y[ix],row_key=sha(path))}
                logger.write_json(fold+'_split.json',dict(held=ix.tolist()))
            def worker(path, name, destination, timeout):
                if path.name=='refit_data.joblib':
                    return challenger, {}
                data = joblib.load(path)
                matrix = challenger.predict_proba(data['X_search'])
                pred = matrix.argmax(1)
                receipt = make_prediction_record(pred,data['y_search'],sha(path))
                receipt['score_record'] = make_score_record(receipt,matrix,challenger.classes_,metric,name)
                facts = dict(resolved_configuration={'id':name}, search_score=float(np.mean(pred==data['y_search'])))
                joblib.dump(dict(prediction_record=receipt,metrics=facts),destination)
                return challenger,facts
            executor = SimpleNamespace(limit=40,accounting=lambda:dict(attempted_classifier_fits=0,unknown_failed_attempts=0),
                                       restrictions=(),caps={},start=time.monotonic(),wall=60,worker=worker)
            result, elected = after_delivery(task,{},metric,selection,choice,banks,payloads,executor,root,logger,10.)
            report = json.loads((root/'cell_encoder_review.json').read_text())
            self.assertEqual(report['action'],'adopt',report.get('reason'))
            self.assertEqual(result['cell_encoder_recipe']['weights'],[.5,.5])
            self.assertEqual(elected['members'],['prior','catboost_cell_cooccurrence'])
            exported = joblib.load(root/result['model_file'])['model']
            np.testing.assert_array_equal(exported.predict(rows),y)
            self.assertEqual(sha(root/'incumbent.joblib'),selection['model_sha256'])

    def test_no_installed_weights_and_task_restrictions(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(list(resources()), [SCRATCH])
        self.assertTrue(permitted(capability(SCRATCH), ['no_pretrained_assets']))
        self.assertFalse(permitted(capability('a'*64), ['no_pretrained_assets']))
        self.assertFalse(permitted(capability(SCRATCH), ['linear_only']))


if __name__=='__main__':
    unittest.main()
