"""Behavior checks for metric ownership and replaceable decision boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
import zipfile
from unittest.mock import patch

import joblib
import numpy as np

from superagent.benchmark import run_one
from superagent.benchmark import train_candidate
from superagent.contracts import TaskSpec,ExperimentProposal,ExperimentResult,DecisionState
from superagent.data import DevelopmentTask
from superagent.metrics import MetricSpec,score_of,metric_of
from superagent.models import Candidate
from superagent.prediction_evidence import make_prediction_record
from superagent.confirmation import shortlist,decide
from superagent.evidence_agent import latest_result,state_for
from superagent.hypothesis_memory import build_memory


def classification_fixture():
    y=np.asarray([0]*18+[1]*2)
    return y,np.zeros(20,dtype=int),np.asarray([0]*14+[1]*6)


def row(n,candidate,accuracy,score=None,metric='accuracy',proposal=None,status='completed'):
    return {'trial':n,'candidate':candidate,'params':{},'status':status,'primary_metric':metric,
        'search_accuracy':accuracy,'train_accuracy':accuracy,'search_score':accuracy if score is None else score,
        'proposal':proposal,'training_seconds':.1}


class ThresholdFixture:
    """Prespecified predictions for a controlled engine test; no fitting."""
    def __init__(self,minority):self.minority=minority
    def predict(self,x):return (np.asarray(x)[:,0]>=14).astype(int) if self.minority else np.zeros(len(x),dtype=int)


class ForwardFixturePolicy:
    def __init__(self,reverse=False):self.reverse=reverse;self.states=[]
    def propose(self,state):
        r=state.record();self.states.append(r)
        configuration=r['available_candidates'][-1 if self.reverse else 0]
        return ExperimentProposal.from_decision(r['next_trial'],configuration,'Controlled policy fixture',None,r['history'])


def simulated_worker(payload_path,candidate,destination,timeout):
    """Only controls predictions; these invocations are not classifier fits."""
    payload=joblib.load(payload_path);model=ThresholdFixture(candidate.endswith('C1.0'))
    metric=MetricSpec.from_payload(payload)
    if payload.get('mode')=='refit':
        record={'model':model,'metrics':{'training_rows':len(payload['y_train']),'training_seconds':.001,'classifier_fits':0}}
    else:
        predicted=model.predict(payload['X_search']);a=metric.evaluate(payload['y_search'],predicted)
        b=metric.evaluate(payload['y_train'],model.predict(payload['X_train']))
        metrics={'primary_metric':metric.name,'search_score':a['score'],'train_score':b['score'],
            'search_accuracy':a['accuracy'],'train_accuracy':b['accuracy'],'score_gap':b['score']-a['score'],
            'generalization_gap':b['accuracy']-a['accuracy'],'training_seconds':.001,'classifier_fits':0}
        record={'model':model,'search_predictions':predicted,'metrics':metrics,
            'prediction_record':make_prediction_record(predicted,payload['y_search'],hashlib.sha256(Path(payload_path).read_bytes()).hexdigest())}
    joblib.dump(record,destination)
    return model,{'status':'completed',**record['metrics']}


class MetricContractTests(unittest.TestCase):
    def test_metric_counterexample_reverses_search_and_confirmation(self):
        labels,a,b=classification_fixture()
        for name,winner in [('accuracy','word_C0.1'),('macro_f1','word_C1.0')]:
            metric=MetricSpec(name,(0,1));aa=metric.evaluate(labels,a);bb=metric.evaluate(labels,b)
            history=[row(1,'word_C0.1',aa['accuracy'],aa['score'],name),row(2,'word_C1.0',bb['accuracy'],bb['score'],name)]
            self.assertEqual(shortlist(history)[0]['candidate'],winner)
            measures={r['candidate']:[dict(r,seed=s) for s in (1,2,3)] for r in history}
            confirmed=decide(shortlist(history),measures,[1,2,3],'exact_rows')
            self.assertTrue(confirmed['valid_confirmation']);self.assertEqual(confirmed['selected_candidate'],winner)
        self.assertGreater(np.mean(a==labels),np.mean(b==labels))
        self.assertLess(MetricSpec('macro_f1',(0,1)).evaluate(labels,a)['score'],MetricSpec('macro_f1',(0,1)).evaluate(labels,b)['score'])

    def test_macro_domain_is_fixed_even_when_one_class_is_absent(self):
        measured=MetricSpec('macro_f1',('a','b','c')).evaluate(['a','b'],['a','b'])
        self.assertEqual(measured['score'],2/3);self.assertEqual(measured['accuracy'],1.)

    def test_bad_metric_vectors_domain_and_missing_score_fail(self):
        for name in ('f1','rmse',None,True):
            with self.assertRaises(ValueError):MetricSpec(name,(0,1))
        metric=MetricSpec('macro_f1',(0,1))
        for labels,preds in [([],[]),([0],[0,1]),([[0]],[0]),([0],[2]),([2],[0]),([0],[float('nan')])]:
            with self.assertRaises(ValueError):metric.evaluate(labels,preds)
        self.assertIsNone(score_of({'primary_metric':'macro_f1','search_accuracy':.99}))
        with self.assertRaises(ValueError):metric_of([{'primary_metric':'macro_f1'},{}])

    def test_confirmation_incomplete_f1_never_substitutes_accuracy(self):
        primary=[row(1,'word_C0.1',.9,.45,'macro_f1')]
        measures={'word_C0.1':[dict(primary[0],seed=1),{'seed':2,'status':'completed','primary_metric':'macro_f1','search_accuracy':1.},dict(primary[0],seed=3)]}
        confirmed=decide(primary,measures,[1,2,3],'exact_rows')
        self.assertFalse(confirmed['valid_confirmation']);self.assertIsNone(confirmed['candidates'][0]['mean_score'])

    def test_feedback_ranks_f1_and_retains_honest_accuracy_diagnostic(self):
        history=[row(1,'word_C0.1',.9,.47,'macro_f1'),row(2,'word_C1.0',.8,.69,'macro_f1')]
        result=latest_result(history)
        self.assertEqual(result['reading'],'improved');self.assertEqual(result['latest_search_accuracy'],.8)
        self.assertEqual(result['latest_search_score'],.69)
        visible=state_for({'primary_metric':'macro_f1'},history,[{'id':'word_C3.0','params':{'C':3.}}],3,'budget_routes')
        self.assertIn('macro_f1',visible['selection_objective']);self.assertEqual(visible['routes'][0]['best_trial'],2)

    def test_initial_empty_memory_routes_and_latest_result_share_task_metric(self):
        available=[{'id':'word_C0.1','params':{'C':.1}}]
        state=DecisionState.build({'primary_metric':'macro_f1'},[],available,3).record()
        self.assertEqual(state['memory']['primary_metric'],'macro_f1')
        self.assertTrue(all(r['primary_metric']=='macro_f1' for r in state['memory']['routes']))
        visible=state_for({'primary_metric':'macro_f1'},[],available,3,'budget_routes')
        self.assertEqual(visible['latest_result']['primary_metric'],'macro_f1')
        self.assertEqual(visible['hypothesis_memory']['primary_metric'],'macro_f1')
        self.assertTrue(all(r['primary_metric']=='macro_f1' for r in visible['routes']))
        with self.assertRaisesRegex(ValueError,'disagrees'):
            build_memory([row(1,'word_C0.1',.5)],available,'macro_f1')
        task=TaskSpec('numeric_tabular',MetricSpec('macro_f1',(0,1)),{})
        explicit=DecisionState.build({},[],available,3,task_contract=task).record()
        self.assertEqual(explicit['profile']['primary_metric'],'macro_f1')
        self.assertEqual(explicit['memory']['primary_metric'],'macro_f1')
        with self.assertRaisesRegex(ValueError,'disagrees'):
            DecisionState.build({'primary_metric':'accuracy'},[],available,3,task_contract=task)

    def test_real_worker_uses_frozen_macro_class_domain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);x=np.concatenate([np.full((30,1),-3.),np.full((30,1),3.)]);y=np.asarray([0]*30+[1]*30)
            payload={'kind':'numeric_tabular','X_train':x,'y_train':y,'X_search':x[:40],'y_search':y[:40],
                'metric_spec':MetricSpec('macro_f1',(0,1,2)).record()}
            path=root/'payload.joblib';joblib.dump(payload,path)
            model,metrics=train_candidate(path,'numeric_tabular_logistic_C1.0',root/'model.joblib',30)
            self.assertEqual(metrics['status'],'completed')
            expected=MetricSpec('macro_f1',(0,1,2)).evaluate(y[:40],model.predict(x[:40]))
            self.assertEqual(metrics['search_score'],expected['score']);self.assertEqual(metrics['search_accuracy'],1.)
            self.assertEqual(metrics['search_score'],2/3)

    def test_unsupported_manifest_is_rejected_before_client_or_credential(self):
        from superagent.cli import build_parser,_run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'task.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('manifest.json',json.dumps({'schema_version':1,'modality':'tabular','metric':'rmse'}))
            args=build_parser().parse_args(['run',str(path),'--planner','llm','--run-dir',str(root/'run')])
            with patch('superagent.cli._validate',return_value=None),patch('superagent.cli.ExperimentLLM') as client:
                with self.assertRaisesRegex(ValueError,'primary metrics'):_run(args)
                client.assert_not_called()
            self.assertFalse((root/'run').exists())


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.history=[row(1,'word_C0.1',.7)]
        self.available=[{'id':'word_C1.0','params':{'C':1.}}]

    def test_proposal_rejects_future_references_modified_config_and_repetition(self):
        base={'candidate':'word_C1.0','hypothesis':'Check a parameter','evidence_trials':[1],
            'expected_change':{'metric':'search_score','reference_trial':1,'direction':'increase','min_delta':.01}}
        p=ExperimentProposal.from_decision(2,self.available[0],'Experiment',base,self.history)
        p.validate_available(self.available,self.history)
        for change in ('future','metric','bool','negative','duplicate'):
            bad=deepcopy(base)
            if change=='future':bad['evidence_trials']=[2]
            if change=='metric':bad['expected_change']['metric']='official_validation_accuracy'
            if change=='bool':bad['expected_change']['reference_trial']=True
            if change=='negative':bad['expected_change']['min_delta']=-.01
            if change=='duplicate':bad['evidence_trials']=[1,1]
            with self.subTest(change=change),self.assertRaises(ValueError):ExperimentProposal.from_decision(2,self.available[0],'Experiment',bad,self.history)
        with self.assertRaises(ValueError):p.validate_available([{'id':'word_C1.0','params':{'C':2.}}],self.history)
        with self.assertRaises(ValueError):ExperimentProposal.from_decision(2,{'id':'word_C0.1','params':{}},'Again',None,self.history)

    def test_state_and_records_are_isolated_and_blind(self):
        history=deepcopy(self.history);history[0]['search_score']=.918273645
        history[0]['official_validation_accuracy']=.99987123
        state=DecisionState.build({'task_name':'PRIVATE_TASK','primary_metric':'accuracy'},history,self.available,3,False)
        record=state.record();self.assertIsNone(record['memory'])
        text=json.dumps(record)
        for sentinel in ('0.918273645','0.99987123','PRIVATE_TASK','search_score'):self.assertNotIn(sentinel,text)
        record['available_candidates'][0]['params']['C']=99
        self.assertEqual(state.record()['available_candidates'][0]['params']['C'],1.)
        self.assertEqual(self.available[0]['params']['C'],1.)

    def test_result_must_match_predeclared_configuration_and_failures_have_no_score(self):
        declared=ExperimentProposal.from_decision(2,self.available[0],'Try',None,self.history)
        measured=row(2,'word_C1.0',None,status='timeout');measured['params']={'C':1.}
        result=ExperimentResult.from_trial(measured,declared).record()
        self.assertFalse(result['score_available']);self.assertIsNone(result['selection_score'])
        measured['params']['C']=2.
        with self.assertRaises(ValueError):ExperimentResult.from_trial(measured,declared)

    def test_result_rejects_worker_metric_identity_mismatch(self):
        declared=ExperimentProposal.from_decision(2,self.available[0],'Try',None,self.history)
        measured=row(2,'word_C1.0',.9);measured['params']={'C':1.}
        with self.assertRaisesRegex(ValueError,'different task metric'):
            ExperimentResult.from_trial(measured,declared,MetricSpec('macro_f1',(0,1)))

    def test_memory_keeps_failed_hypothesis_and_old_route(self):
        proposal={'hypothesis':'Increasing C may help','evidence_trials':[1],
            'expected_change':{'metric':'search_score','reference_trial':1,'direction':'increase','min_delta':.01},
            'route_plan':{'next_check':'Confirm on another internal split','revisit_previous_if':'Return if the alternative fails'}}
        h=[row(1,'word_C0.1',.8),row(2,'word_char_C1.0',.79,proposal=proposal)]
        h[1]['hypothesis_check']={'status':'not_supported','metric':'search_score','observed_delta':-.01}
        memory=build_memory(h,[{'id':'word_C1.0','params':{'C':1.}},{'id':'word_char_C3.0','params':{'C':3.}}])
        self.assertEqual(memory['hypotheses'][0]['status'],'measurement_not_supported')
        old=next(r for r in memory['routes'] if r['route']=='word');self.assertEqual(old['best_trial'],1)
        self.assertEqual(old['remaining_configurations'],1)
        self.assertEqual(old['resume_if'],'Return if the alternative fails')
        self.assertFalse(memory['hypotheses'][0]['causal_explanation_verified'])


class EngineCounterexampleTests(unittest.TestCase):
    def test_two_decision_implementations_use_same_executor_and_metric_contract(self):
        # Four simulated engines: no model fitting, API or Released data.
        labels,_,_=classification_fixture();x=np.tile(np.arange(20),4).reshape(-1,1).astype(float)
        y=np.tile(labels,4);val_x=np.arange(20).reshape(-1,1).astype(float)
        catalog=[Candidate('numeric_tabular_logistic_C0.1',{},None),Candidate('numeric_tabular_logistic_C1.0',{},None)]
        for name,winner in [('accuracy','numeric_tabular_logistic_C0.1'),('macro_f1','numeric_tabular_logistic_C1.0')]:
            for reverse in (False,True):
                with self.subTest(metric=name,reverse=reverse),tempfile.TemporaryDirectory() as temp:
                    directory=Path(temp)/'run';policy=ForwardFixturePolicy(reverse)
                    task=DevelopmentTask('numeric_tabular','SIMULATED',x,y,val_x,labels,{'metric':name})
                    with patch('superagent.benchmark.search_candidates',return_value=catalog),patch('superagent.benchmark.train_candidate',side_effect=simulated_worker),patch('superagent.benchmark.inner_split',return_value=(np.arange(60),np.arange(60,80))),patch('superagent.confirmation.inner_split',return_value=(np.arange(60),np.arange(60,80))):
                        result=run_one(task,'custom_fixture',7,directory,None,max_trials=2,confirmation_repeats=3,
                            confirmation_top_k=2,refit_full_train=True,decision_policy=policy)
                    self.assertEqual(result['selected_candidate'],winner);self.assertTrue(result['refit_succeeded'])
                    self.assertEqual(result['fallback_count'],0);self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],0)
                    measured=MetricSpec(name,(0,1)).evaluate(labels,ThresholdFixture(name=='macro_f1').predict(val_x))
                    self.assertEqual(result['official_validation_score'],measured['score'])
                    self.assertEqual(result['official_validation_accuracy'],measured['accuracy'])
                    self.assertEqual(len(policy.states),2)
                    proposals=json.loads((directory/'experiment_proposals.json').read_text())
                    results=json.loads((directory/'experiment_results.json').read_text())
                    self.assertEqual([r['candidate'] for r in proposals],[r['candidate'] for r in results])
                    final=json.loads((directory/'final_selection.json').read_text())
                    self.assertEqual(final['primary_metric'],name)
                    events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()]
                    kinds=[r['kind'] for r in events]
                    self.assertLess(kinds.index('experiment_proposal_frozen'),kinds.index('trial_result'))
                    self.assertLess(kinds.index('final_model_frozen'),kinds.index('validation_scored'))


if __name__=='__main__':unittest.main()
