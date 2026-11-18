"""Architecture invariants and failures, without live model requests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from superagent.contracts import DecisionState, TaskSpec
from superagent.decision_architecture import ArchitecturePolicy, wire_state
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.metrics import MetricSpec
from superagent.task_budget import TaskBudgetAccount, TaskBudgetLedger
from superagent.hypothesis_memory import build_memory
from superagent.cli import build_parser, _run
from superagent.benchmark import run_one
from superagent.data import DevelopmentTask
from superagent.models import Candidate
from test_architecture_contracts import simulated_worker


def state_fixture(metric='macro_f1'):
    history=[{'trial':1,'candidate':'word_C0.1','params':{'C':.1},'status':'completed',
              'primary_metric':metric,'search_score':.77,'search_accuracy':.88,
              'train_score':.82,'train_accuracy':.9,'training_seconds':.1,'proposal':None}]
    available=[{'id':'word_C1.0','params':{'C':1.}}, {'id':'char_C1.0','params':{'C':1.}}]
    return DecisionState.build({'primary_metric':metric},history,available,3,
        task_contract=TaskSpec('pair_text',MetricSpec(metric,(0,1)),{}))


def proposal(candidate='word_C1.0', trial=1):
    return {'candidate':candidate,'reason':'Check this configuration under the fixed objective.',
            'hypothesis':'This configuration may improve primary score.',
            'review':'One previous configuration does not settle the best route.',
            'evidence_trials':[trial],
            'expected_change':{'metric':'search_score','reference_trial':trial,'direction':'increase','min_delta':.001},
            'assessment':{'fact_ids':[f't{trial}.search_score'],
                          'interpretation':'Compare a different configuration.', 'uncertainty':'No causal proof.'},
            'route_plan':{'next_check':'Compare primary score on the fixed internal split.',
                          'revisit_previous_if':'Revisit if this alternative fails to improve primary score.'}}


def review(replacement=None):
    return {'action':'keep' if replacement is None else 'revise','reason':'Check an alternative without deleting the incumbent.',
            'fact_ids':['t1.search_score'],'replacement':replacement}


class ScriptClient:
    def __init__(self, answers):self.answers=list(answers);self.calls=[]
    def ask(self,payload,logger,role):
        self.calls.append((deepcopy(payload),role))
        answer=self.answers.pop(0)
        if isinstance(answer,Exception):raise answer
        return deepcopy(answer)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.logger=RunLogger(self.root);self.logger.initialize_usage_ledger()
        self.state=state_fixture()

    def test_three_arms_same_planner_and_evidence_both_review_modes_keep(self):
        first=[]
        for mode in ('single','self_review','reviewer'):
            client=ScriptClient([proposal(),review()]);policy=ArchitecturePolicy(client,self.logger,mode)
            chosen=policy.propose(self.state);first.append(client.calls[0])
            self.assertEqual(chosen.candidate,'word_C1.0')
            self.assertEqual(len(client.calls),1 if mode=='single' else 2)
            self.assertTrue(policy.summary()['all_requested_decisions_valid'])
            if mode!='single':
                a={k:v for k,v in client.calls[0][0].items() if k!='instruction'}
                b={k:v for k,v in client.calls[1][0].items() if k not in ('instruction','proposed_experiment','proposed_diagnostics')}
                self.assertEqual(a,b)
                self.assertEqual(policy.records[0]['original_proposal'],policy.records[0]['chosen_proposal'])
        self.assertEqual(first[0],first[1]);self.assertEqual(first[1],first[2])

    def test_revise_retains_original_old_route_and_incumbent(self):
        original=self.state.record()
        policy=ArchitecturePolicy(ScriptClient([proposal(),review(proposal('char_C1.0'))]),self.logger,'reviewer')
        chosen=policy.propose(self.state);self.assertEqual(chosen.candidate,'char_C1.0')
        self.assertEqual(policy.records[0]['original_proposal']['candidate'],'word_C1.0')
        self.assertEqual(self.state.record(),original)
        history=original['history']+[{'trial':2,'candidate':chosen.candidate,'params':chosen.params,
            'status':'completed','primary_metric':'macro_f1','search_score':.7,'search_accuracy':.9,
            'proposal':chosen.raw_proposal}]
        memory=build_memory(history,[original['available_candidates'][0]],'macro_f1')
        prior=next(r for r in memory['routes'] if r['route']=='word')
        self.assertEqual(prior['best_trial'],1);self.assertEqual(prior['remaining_configurations'],1)
        self.assertIn('Revisit',prior['resume_if'])
        self.assertFalse(policy.summary()['freeform_explanations_verified'])

    def test_invalid_review_matrix_retains_valid_original_and_blocks_new_calls(self):
        cases=[]
        for key,value in [('fact_ids',['t99.search_score']),('fact_ids',['t1.search_score']*2),
                          ('fact_ids',[]),('action','run_extra_experiment'),('reason',''),('replacement',proposal())]:
            cases.append({**review(),key:value})
        for mutation in ('future','params','metric','tried','fabricated','extra_budget'):
            answer=proposal('char_C1.0')
            if mutation=='future':answer['evidence_trials']=[2]
            if mutation=='params':answer['params']={'C':999}
            if mutation=='metric':answer['expected_change']['metric']='official_validation_accuracy'
            if mutation=='tried':answer['candidate']='word_C0.1'
            if mutation=='fabricated':answer['assessment']['fact_ids']=['invented']
            if mutation=='extra_budget':answer['trials']=50
            cases.append(review(answer))
        cases.extend([review(None)|{'action':'revise'},RuntimeError('Synthetic review failure')])
        for bad in cases:
            with self.subTest(answer=bad):
                client=ScriptClient([proposal(),bad]);policy=ArchitecturePolicy(client,self.logger,'reviewer')
                self.assertEqual(policy.propose(self.state).candidate,'word_C1.0')
                self.assertEqual(policy.summary()['review_fallback_count'],1)
                self.assertFalse(policy.summary()['all_requested_decisions_valid'])
                with self.assertRaisesRegex(RuntimeError,'disabled'):policy.propose(self.state)
                self.assertEqual(len(client.calls),2)

    def test_bad_planner_never_calls_review_or_retries(self):
        for answer in ({},proposal('word_C0.1'),RuntimeError('Synthetic planning failure')):
            client=ScriptClient([answer]);policy=ArchitecturePolicy(client,self.logger,'reviewer')
            with self.assertRaises((ValueError,RuntimeError)):policy.propose(self.state)
            self.assertEqual(len(client.calls),1);self.assertTrue(policy.blocked)
            self.assertEqual(policy.records[0]['status'],'planner_failed')
            self.assertIsNone(policy.records[0]['chosen_proposal'])

    def test_mutating_payload_does_not_mutate_state_or_original(self):
        expected=self.state.record()
        class Mutator(ScriptClient):
            def ask(self,payload,logger,role):
                answer=super().ask(payload,logger,role)
                payload['available_candidates'][0]['params']['C']=999
                if 'proposed_experiment' in payload:payload['proposed_experiment']['params']['C']=999
                return answer
        policy=ArchitecturePolicy(Mutator([proposal(),review()]),self.logger,'reviewer')
        chosen=policy.propose(self.state)
        self.assertEqual(chosen.params,{'C':1.});self.assertEqual(self.state.record(),expected)
        self.assertEqual(policy.records[0]['original_proposal']['params'],{'C':1.})

    def test_blind_or_exhausted_state_fails_before_any_call(self):
        for change in ('blind','exhausted'):
            record=self.state.record()
            if change=='blind':record['feedback_visible']=False
            else:record['trial_budget']['remaining']=0
            client=ScriptClient([]);policy=ArchitecturePolicy(client,self.logger,'reviewer')
            with self.assertRaises(ValueError):policy.propose(DecisionState(record))
            self.assertEqual(client.calls,[])

    def test_local_review_rejected_before_loading_data_or_client(self):
        for mode in ('self_review','reviewer'):
            args=build_parser().parse_args(['run','absent.zip','--architecture',mode])
            with patch('superagent.cli.load_development_task') as data,patch('superagent.cli.ExperimentLLM') as client:
                with self.assertRaisesRegex(ValueError,'explicit --planner llm'):_run(args)
                data.assert_not_called();client.assert_not_called()

    def test_last_step_failed_review_is_impure_even_with_model_and_no_random_fallback(self):
        x=np.tile(np.arange(20),4).reshape(-1,1).astype(float);y=np.tile(np.asarray([0]*18+[1]*2),4)
        task=DevelopmentTask('numeric_tabular','SYNTHETIC',x,y,x[:20],y[:20],{'metric':'accuracy'})
        a='numeric_tabular_logistic_C0.1';b='numeric_tabular_logistic_C1.0'
        catalog=[Candidate(a,{},None),Candidate(b,{},None)]
        client=ScriptClient([proposal(b),RuntimeError('Review failed on final step')])
        with patch('superagent.benchmark.search_candidates',return_value=catalog),patch('superagent.benchmark.train_candidate',side_effect=simulated_worker),patch('superagent.benchmark.inner_split',return_value=(np.arange(60),np.arange(60,80))),patch('superagent.confirmation.inner_split',return_value=(np.arange(60),np.arange(60,80))):
            result=run_one(task,'budget_routes',707,self.root/'run',client,max_trials=2,anchor_first=True,
                confirmation_repeats=3,confirmation_top_k=2,refit_full_train=True,decision_architecture='reviewer')
        self.assertTrue(result['completed']);self.assertTrue(result['refit_succeeded'])
        self.assertFalse(result['valid_policy_run']);self.assertEqual(result['fallback_count'],0)
        self.assertEqual(result['decision_architecture']['review_fallback_count'],1)
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],0)


class ArchitectureBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.key=self.root/'dummy_key';self.key.write_text('SYNTHETIC-NONSECRET')
        self.logger=RunLogger(self.root);self.logger.initialize_usage_ledger()
        self.ledger=TaskBudgetLedger.initialize(self.root/'cost.sqlite',new_project=True)
        self.account=TaskBudgetAccount(self.ledger,'synthetic','unit-test')

    def client(self,requests=2,budget=.2):
        return ExperimentLLM('https://example.invalid',str(self.key),'deepseek-flash',budget,
            max_requests=requests,account=self.account,allowed_served_models={'deepseek-flash'})

    def response(self,answer,usage=None):
        from unittest.mock import Mock
        result=Mock();result.ok=True
        result.json.return_value={'model':'deepseek-flash','usage':usage if usage is not None else
            {'prompt_tokens':100,'completion_tokens':20},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(answer)}}]}
        return result

    def test_two_calls_settle_same_account_and_only_one_trial_proposal(self):
        client=self.client();policy=ArchitecturePolicy(client,self.logger,'reviewer')
        with patch('superagent.experiment_llm.requests.post',side_effect=[self.response(proposal()),self.response(review())]) as post:
            chosen=policy.propose(state_fixture())
        self.assertEqual(chosen.trial,2);self.assertEqual(post.call_count,2)
        state=self.ledger.snapshot('synthetic')
        self.assertAlmostEqual(state['known_task_cost_rmb'],.00036)
        self.assertEqual(state['pending_requests'],0);self.assertEqual(state['unresolved_requests'],0)
        self.assertTrue(policy.summary()['all_requested_decisions_valid'])

    def test_review_call_cap_consumes_no_second_transport_or_retry(self):
        client=self.client(requests=1);policy=ArchitecturePolicy(client,self.logger,'reviewer')
        with patch('superagent.experiment_llm.requests.post',return_value=self.response(proposal())) as post:
            chosen=policy.propose(state_fixture())
        self.assertEqual(chosen.candidate,'word_C1.0');self.assertEqual(post.call_count,1)
        self.assertEqual(client.request_count,1);self.assertEqual(policy.ask_invocations['review'],1)
        self.assertEqual(policy.summary()['review_fallback_count'],1)
        self.assertAlmostEqual(self.ledger.snapshot('synthetic')['known_task_cost_rmb'],.00018)

    def test_insufficient_run_allowance_for_review_no_second_transport(self):
        client=self.client(budget=.025);policy=ArchitecturePolicy(client,self.logger,'reviewer')
        with patch('superagent.experiment_llm.requests.post',return_value=self.response(proposal(),
            {'prompt_tokens':15000,'completion_tokens':20})) as post:
            chosen=policy.propose(state_fixture())
        self.assertEqual(chosen.candidate,'word_C1.0');self.assertEqual(post.call_count,1)
        self.assertEqual(policy.summary()['review_fallback_count'],1)
        self.assertIn('cost ceiling',policy.records[0]['error'])
        self.assertAlmostEqual(client.spent,.01508)

    def test_unknown_review_usage_stays_unknown_and_opens_both_circuits(self):
        client=self.client();policy=ArchitecturePolicy(client,self.logger,'reviewer')
        with patch('superagent.experiment_llm.requests.post',side_effect=[self.response(proposal()),self.response(review(),{})]) as post:
            self.assertEqual(policy.propose(state_fixture()).candidate,'word_C1.0')
            with self.assertRaises(RuntimeError):policy.propose(state_fixture())
        state=self.ledger.snapshot('synthetic')
        self.assertEqual(post.call_count,2);self.assertTrue(client.blocked);self.assertTrue(policy.blocked)
        self.assertEqual(state['unresolved_requests'],1);self.assertIsNone(state['remaining_course_allowance_rmb'])
        rows=[json.loads(line) for line in self.logger.usage_path.read_text().splitlines()]
        self.assertFalse(rows[-1]['usage_known']);self.assertIsNone(rows[-1]['course_cost_rmb'])

    def test_same_deadline_stops_review_before_transport(self):
        client=self.client();policy=ArchitecturePolicy(client,self.logger,'reviewer')
        client.set_deadline(time.monotonic()+100)
        def first(*args,**kwargs):
            client.set_deadline(time.monotonic()-1)
            return self.response(proposal())
        with patch('superagent.experiment_llm.requests.post',side_effect=first) as post:
            self.assertEqual(policy.propose(state_fixture()).candidate,'word_C1.0')
        self.assertEqual(post.call_count,1);self.assertEqual(policy.summary()['review_fallback_count'],1)
        self.assertEqual(policy.records[0]['error_type'],'PlanningTimeExhausted')


if __name__=='__main__':unittest.main()
