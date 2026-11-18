import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from superagent.prediction_evidence import make_prediction_record,compare_predictions,reference_trial,comparison_for_trial
from superagent.evidence_agent import state_for,measured_facts,latest_result
from superagent.models import search_candidates
from superagent.benchmark import run_one
from superagent.data import DevelopmentTask

PAYLOAD='a'*64


def comparison(labels,current,reference):
    return compare_predictions(make_prediction_record(current,labels,PAYLOAD),make_prediction_record(reference,labels,PAYLOAD),
        labels,2,1,'logistic_C0.25','logistic_C0.05')


class PredictionEvidenceTests(unittest.TestCase):
    def test_binary_tie_can_contain_opposing_correctness_changes(self):
        result=comparison([0,0,1,1,0,1],[0,1,1,0,1,1],[0,0,0,0,1,1])
        self.assertEqual({k:result[k] for k in ('both_correct','newly_correct','newly_wrong','both_wrong')},
            {'both_correct':2,'newly_correct':1,'newly_wrong':1,'both_wrong':2})
        self.assertEqual(result['net_correct_change'],0);self.assertEqual(result['accuracy_change'],0)
        self.assertEqual(result['prediction_disagreement_rows'],2);self.assertFalse(result['statistical_test_performed'])

    def test_multiclass_both_wrong_may_still_disagree_and_gain_is_exact_integer_count(self):
        result=comparison([0,1,2,0],[2,0,2,0],[1,1,0,0])
        self.assertEqual(result['prediction_disagreement_rows'],3)
        self.assertEqual(result['newly_correct']+result['newly_wrong'],2);self.assertEqual(result['both_wrong'],1)
        gain=comparison(['a','b','c'],['a','b','c'],['a','a','a'])
        self.assertEqual(gain['newly_correct'],2);self.assertEqual(gain['net_correct_change'],2)
        self.assertEqual(gain['accuracy_change'],2/3)

    def test_different_payloads_ordered_labels_or_corrupted_predictions_are_rejected(self):
        labels=[0,1,1];a=make_prediction_record([0,1,0],labels,PAYLOAD);b=make_prediction_record([1,1,0],labels,PAYLOAD)
        for corrupt in ('payload','predictions','rows','labels'):
            changed=copy.deepcopy(b)
            if corrupt=='payload':changed['evaluation_payload_sha256']='b'*64
            elif corrupt=='predictions':changed['predictions'][0]=0
            elif corrupt=='rows':changed['rows']=4
            else:changed['label_sha256']=make_prediction_record([0,1,0],[1,0,1],PAYLOAD)['label_sha256']
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):
                compare_predictions(a,changed,labels,2,1,'new','old')

    def test_reference_uses_best_completed_score_and_same_earliest_tie_as_latest_result(self):
        history=[{'trial':1,'candidate':'first','status':'completed','search_accuracy':.7},
            {'trial':2,'candidate':'second','status':'completed','search_accuracy':.7},
            {'trial':3,'candidate':'failed','status':'failed','search_accuracy':.9}]
        latest={'trial':4,'candidate':'current','status':'completed','search_accuracy':.71}
        self.assertEqual(reference_trial(history)['trial'],1)
        self.assertEqual(latest_result(history+[latest])['best_prior_trial'],1)
        self.assertEqual(reference_trial(history+[latest])['trial'],4)

    def test_missing_and_failed_evidence_is_explicitly_unavailable_not_zero(self):
        first={'trial':1,'candidate':'logistic_C0.05','status':'completed','search_accuracy':.5}
        current={'trial':2,'candidate':'logistic_C0.25','status':'completed','search_accuracy':.5}
        with tempfile.TemporaryDirectory() as temp:
            for row,history in ((first,[]),({**current,'status':'failed'},[first]),(current,[first])):
                result=comparison_for_trial(row,history,temp,[0,1],PAYLOAD)
                self.assertEqual(result['status'],'unavailable');self.assertNotIn('net_correct_change',result)

    def test_record_score_and_frozen_payload_mismatch_never_return_numeric_claims(self):
        first={'trial':1,'candidate':'logistic_C0.05','status':'completed','search_accuracy':.5}
        current={'trial':2,'candidate':'logistic_C0.25','status':'completed','search_accuracy':1.}
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            joblib.dump({'prediction_record':make_prediction_record([0,0],[0,1],PAYLOAD)},directory/'trial_1.joblib')
            joblib.dump({'prediction_record':make_prediction_record([0,1],[0,1],PAYLOAD)},directory/'trial_2.joblib')
            good=comparison_for_trial(current,[first],directory,[0,1],PAYLOAD)
            self.assertEqual(good['net_correct_change'],1)
            for row,digest in (({**current,'search_accuracy':.6},PAYLOAD),(current,'b'*64)):
                bad=comparison_for_trial(row,[first],directory,[0,1],digest)
                self.assertEqual(bad['status'],'invalid');self.assertNotIn('newly_correct',bad)

    def test_feedback_contains_counts_but_blind_state_contains_no_observed_comparison(self):
        paired=comparison([0,0,1,1,0,1],[0,1,1,0,1,1],[0,0,0,0,1,1])
        history=[{'trial':1,'candidate':'logistic_C0.05','params':{'C':.05},'status':'completed','search_accuracy':.5},
            {'trial':2,'candidate':'logistic_C0.25','params':{'C':.25},'status':'completed','search_accuracy':.5,'paired_error_comparison':paired}]
        visible=state_for({},history,[],3,'budget_routes');blind=state_for({},history,[],3,'evidence_blind')
        self.assertEqual(visible['latest_error_comparison']['newly_correct'],1)
        self.assertNotIn('latest_error_comparison',blind);self.assertEqual(blind['facts'],[])
        self.assertNotIn('paired_error_comparison',str(blind));self.assertNotIn('predictions',str(visible))
        fact=next(f for f in measured_facts(history) if f['id']=='t2.newly_wrong')
        self.assertEqual(fact['value'],1);self.assertEqual(fact['reference_trial'],1)

    def test_nonfinite_empty_or_non_scalar_vectors_and_invalid_trial_reference_fail(self):
        for predictions,labels in (([],[]),([[0],[1]],[0,1]),([0,float('nan')],[0,1]),([0],[0,1])):
            with self.assertRaises(ValueError):make_prediction_record(predictions,labels,PAYLOAD)
        record=make_prediction_record([0,1],[0,1],PAYLOAD)
        with self.assertRaises(ValueError):compare_predictions(record,record,[0,1],1,1,'same','same')

    def test_full_local_pipeline_persists_records_and_reconciles_feedback_without_extra_fits(self):
        x=np.asarray([[f'group_{i}',f'color_{i%2}'] for i in range(100)],dtype=object);y=np.arange(100)%2
        task=DevelopmentTask('categorical_tabular','synthetic',x,y,x[:12],y[:12],{})
        candidates=search_candidates('categorical_tabular','expanded')[:3]
        with tempfile.TemporaryDirectory() as temp,patch('superagent.benchmark.search_candidates',return_value=candidates):
            directory=Path(temp)/'run'
            result=run_one(task,'random',707,directory,None,max_trials=3,anchor_first=True,
                confirmation_repeats=3,confirmation_top_k=3,refit_full_train=True,portfolio='expanded')
            rows=json.loads((directory/'trials.json').read_text());payload=joblib.load(directory/'inner_data.joblib')
            payload_hash=hashlib.sha256((directory/'inner_data.joblib').read_bytes()).hexdigest()
            for i,row in enumerate(rows):
                artifact=joblib.load(directory/f'trial_{row["trial"]}.joblib')
                self.assertEqual(artifact['prediction_record']['evaluation_payload_sha256'],payload_hash)
                self.assertEqual(row['paired_error_comparison'],comparison_for_trial(row,rows[:i],directory,payload['y_search'],payload_hash))
        self.assertTrue(result['refit_succeeded']);self.assertTrue(result['confirmation']['valid_confirmation'])
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],10)
        self.assertEqual(result['prediction_evidence_audit'],{'comparisons_completed':2,'comparisons_unavailable':1,'invalid_comparisons':0,'integrity_valid':True})


if __name__=='__main__':unittest.main()
