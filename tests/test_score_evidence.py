"""Score identity counterexamples: labels alone are insufficient for AUC."""
from copy import deepcopy
import unittest

import numpy as np
from superagent.ab_catalog import capabilities_for,untried_candidates
from superagent.ab_scheduler import selection_evidence,ROW_SCOPE
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import make_prediction_record
from superagent.score_evidence import make_score_record,checked_score_record,matrix_sha256
from tests.test_ab_scheduler import recorded


class ScoreEvidenceTests(unittest.TestCase):
    def fixture(self):
        state=recorded();state['task']['metric']['name']='roc_auc'
        metric=MetricSpec.from_payload({'metric_spec':state['task']['metric']})
        caps=capabilities_for(state['task'],train_rows=256)
        state.update(history=[],prediction_evidence=[])
        trusted={'primary':dict(row_ids=[0,1,2,3],labels=[0,0,1,1],evaluation_payload_sha256='a'*64)}
        matrix=np.array([[.9,.1],[.8,.2],[.7,.3],[.6,.4]])
        for trial,(name,cap) in enumerate(list(caps.items())[:2],1):
            values=matrix if trial==1 else matrix[::-1]
            pred=[0]*4;record=make_prediction_record(pred,trusted['primary']['labels'],'a'*64)
            parent=dict(trial=trial,candidate_id=name,split_id='primary',row_ids=[0,1,2,3],row_id_scope=ROW_SCOPE,
                **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in record.items() if k!='schema_version'})
            parent['score_record']=make_score_record(record,values,[0,1],metric,name)
            state['prediction_evidence'].append(parent)
            state['history'].append(dict(trial=trial,candidate=name,params=cap['params'],split_id='primary',
                status='completed',primary_metric='roc_auc',metric_spec=metric.record(),
                search_score=metric.evaluate(trusted['primary']['labels'],pred,values,[0,1])['score'],
                search_accuracy=.5,classifier_fits=1))
        state['available_candidates']=untried_candidates(caps,state['history'])
        return state,metric,trusted

    def test_same_hard_predictions_choose_by_auc(self):
        state,_,trusted=self.fixture()
        self.assertEqual(state['prediction_evidence'][0]['predictions'],state['prediction_evidence'][1]['predictions'])
        review=selection_evidence(state,trusted_partitions=trusted)
        self.assertEqual(review['incumbent'],state['history'][0]['candidate'])
        state['history'].reverse()  # Trial identity, not incidental list order, decides chronology.
        self.assertEqual(selection_evidence(state,trusted_partitions=trusted),review)

    def test_scores_cannot_be_confirmed_without_trusted_labels(self):
        state,_,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'trusted local labels'):selection_evidence(state)

    def test_mutated_identity_or_matrix_rejected(self):
        state,_,trusted=self.fixture()
        changes={'scores':[[.5,.5]]*4,'class_order':[1,0],'candidate_id':'different',
            'scores_sha256':'b'*64,'class_order_sha256':'b'*64,'label_sha256':'b'*64,
            'prediction_sha256':'b'*64,'evaluation_payload_sha256':'b'*64,'metric_spec':None}
        for key,value in changes.items():
            bad=deepcopy(state);bad['prediction_evidence'][0]['score_record'][key]=value
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError,TypeError)):
                selection_evidence(bad,trusted_partitions=trusted)
        bad=deepcopy(state);del bad['prediction_evidence'][0]['score_record']
        with self.assertRaises(ValueError):selection_evidence(bad,trusted_partitions=trusted)

    def test_rehashed_scores_still_must_reproduce_recorded_auc(self):
        state,_,trusted=self.fixture();item=state['prediction_evidence'][0]['score_record']
        item['scores']=item['scores'][::-1];item['scores_sha256']=matrix_sha256(item['scores'])
        with self.assertRaisesRegex(ValueError,'Stored score differs'):
            selection_evidence(state,trusted_partitions=trusted)

    def test_synchronous_column_reordering_is_valid(self):
        state,metric,trusted=self.fixture();parent=state['prediction_evidence'][0]
        matrix=np.array(parent['score_record']['scores'])[:,::-1]
        parent['score_record']=make_score_record(parent,matrix,[1,0],metric,parent['candidate_id'])
        checked_score_record(parent['score_record'],parent,metric,parent['candidate_id'],trusted['primary']['labels'])
        self.assertEqual(selection_evidence(state,trusted_partitions=trusted)['incumbent'],parent['candidate_id'])

    def test_invalid_probabilities_cannot_be_signed(self):
        state,metric,_=self.fixture();parent=state['prediction_evidence'][0]
        for values,classes in (([[.5,.5]]*3,[0,1]),([[1.1,-.1]]*4,[0,1]),
                               ([[.1,.2]]*4,[0,1]),([[np.nan,.5]]*4,[0,1]),([[.5,.5]]*4,[0,0])):
            with self.subTest(classes=classes),self.assertRaises(ValueError):
                make_score_record(parent,values,classes,metric,parent['candidate_id'])


if __name__=='__main__':unittest.main()
