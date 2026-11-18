from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from superagent.generated_pipeline import (checked_factory,classifier_inventory,materialize_generated,
    fit_generated,resolve_generated_registry)
from superagent.research_registry import ResearchRegistry
from superagent.web_research import LiteratureSearch
from superagent.data import DevelopmentTask
from superagent.metrics import MetricSpec
from superagent.models import search_candidates,predict_with_model
from superagent.benchmark import run_one
from superagent.events import RunLogger
from superagent.cli import build_parser,_validate


CODE="from sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import SGDClassifier\ndef build_model(kind):\n    return Pipeline([('features', make_preprocessor(kind)), ('classifier', SGDClassifier(loss='log_loss', alpha=0.001, max_iter=300, random_state=42, n_jobs=1))])\n"


class FixtureSearch:
    def __init__(self,fail=False):self.calls=[];self.records=[];self.fail=fail
    def search(self,query):
        self.calls.append(query)
        if self.fail:raise RuntimeError('Declared simulated retrieval failure')
        result={'query':query,'sources':[{'source_id':'fixture_method','title':'Local synthetic evidence fixture',
            'summary':'Only a test of source/decision plumbing; not an actual retrieved paper.',
            'url':'https://example.invalid/fixture','provider':'fixture','untrusted_content':True}],
            'mock':True,'network_requests':0,'scope':'Synthetic metadata fixture, no actual network'}
        self.records.append(result);return result


class ScriptClient:
    def __init__(self,skip=False,code=CODE):self.calls=[];self.skip=skip;self.code=code;self.deadlines=[]
    def set_deadline(self,value):self.deadlines.append(value)
    def ask(self,payload,logger,role):
        self.calls.append({'role':role,'payload':deepcopy(payload)})
        logger.event('mock_model_request',role=role,payload=deepcopy(payload),real_api_request=False)
        if role=='method_researcher':
            answer={'action':'skip' if self.skip else 'search','query':None if self.skip else 'generic sparse text classification linear optimization',
                'reason':'Synthetic scripted action for integration testing.','evidence_trials':[payload['observations'][-1]['trial']]}
        elif role=='method_codegen':
            answer={'code':self.code,'hypothesis':'The new installed classifier can be measured against the retained incumbent.',
                'reason':'Synthetic scripted proposal; no actual scientific support is claimed.',
                'source_ids':['fixture_method'],'evidence_trials':[payload['observations'][-1]['trial']]}
        else:
            available=payload['available_candidates'];latest=payload['latest_result']['latest_trial']
            generated=[r['id'] for r in available if r['id'].startswith('generated_')]
            preferred=['word_char_stats_C0.3','char_C0.3']
            candidate=generated[0] if generated else next((name for name in preferred if any(r['id']==name for r in available)),available[0]['id'])
            answer={'candidate':candidate,'reason':f'Use observed trial{latest} and measure the next available route.',
                'hypothesis':'The next candidate may preserve or improve the measured score.',
                'review':'Retain the old incumbent when the new candidate loses.',
                'evidence_trials':[latest],'expected_change':{'metric':'search_score','reference_trial':latest,'direction':'increase','min_delta':0.},
                'assessment':{'fact_ids':[f't{latest}.search_score'],
                    'interpretation':'This scripted integration decision uses executor reading '+payload['latest_result']['reading']+'.','uncertainty':'Mock planner; no causal or scientific verification.'},
                'route_plan':{'next_check':'Measure next candidate on the same internal partition.','revisit_previous_if':'Retain confirmed incumbent if the next candidate fails or scores lower.'}}
        logger.event('mock_model_response',role=role,answer=deepcopy(answer),real_api_request=False)
        return answer


def pair_task():
    x=[(f'warm bright sun row{i}' if i%2 else f'cold dark rain row{i}',
        f'cold dark rain other{i}' if i%2 else f'warm bright sun other{i}') for i in range(130)]
    y=np.arange(130)%2
    return DevelopmentTask('pair_text','synthetic',x[:100],y[:100],x[100:],y[100:],{'metric':'accuracy'})


class WebGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def run_mock(self,name,client=None,searcher=None):
        client=client or ScriptClient();searcher=searcher or FixtureSearch()
        result=run_one(pair_task(),'budget_routes',707,self.root/name,client,max_trials=4,
            anchor_first=True,refit_full_train=True,confirmation_repeats=3,confirmation_top_k=3,
            confirmation_grouping='shared_text',search_grouping='shared_text',portfolio='expanded',
            refinement='research',research_searcher=searcher,time_limit=180,trial_timeout=30)
        return result,client,searcher

    def test_runtime_inventory_and_unregistered_constructor(self):
        self.assertGreater(len(classifier_inventory()),20)
        candidate=materialize_generated('pair_text',CODE)
        self.assertNotIn(candidate.family,[c.family for c in search_candidates('pair_text','all')])
        self.assertEqual(candidate.params['classifier'],'SGDClassifier')

    def test_generated_model_serializes_without_importing_generated_module(self):
        task=pair_task();model,_=fit_generated(task.kind,CODE,task.X_train,task.y_train,task.X_val)
        path=self.root/'model.joblib';joblib.dump(model,path)
        restored=joblib.load(path)
        np.testing.assert_array_equal(model.predict(task.X_val),predict_with_model(task.kind,restored,task.X_val))
        p=restored.predict_proba(task.X_val)[:,1];q=restored.predict_proba([(b,a) for a,b in task.X_val])[:,1]
        np.testing.assert_allclose(p+q,1.,atol=1e-12)
        self.assertEqual(restored.classifier_fits,1)

    def test_forbidden_calls_statements_attributes_and_generics_are_rejected(self):
        samples=[CODE+'\nopen("/tmp/anything")',CODE.replace('make_preprocessor(kind)','open("secret")'),
            CODE.replace('SGDClassifier(loss=', 'SGDClassifier.__call__(loss='),
            CODE.replace('def build_model(kind):','@print\ndef build_model(kind):'),
            CODE.replace('def build_model(kind):','def build_model[T](kind):'),
            CODE.replace('return Pipeline','x = 1\n    return Pipeline'),
            CODE.replace('from sklearn.linear_model import SGDClassifier','import os')]
        for code in samples:
            with self.subTest(code=code[:60]),self.assertRaises((ValueError,SyntaxError)):checked_factory(code,'pair_text')

    def test_resource_limits_seed_and_implicit_cv_are_checked(self):
        samples=[CODE.replace('max_iter=300','max_iter=10000'),CODE.replace('n_jobs=1','n_jobs=-1'),
            CODE.replace('random_state=42','random_state=None'),CODE.replace('SGDClassifier','LogisticRegressionCV')]
        for code in samples:
            with self.assertRaises(ValueError):checked_factory(code,'single_text')

    def test_registry_hash_kind_and_code_tampering_rejected(self):
        candidate=materialize_generated('pair_text',CODE)
        entry={'spec':{'kind':'pair_text','code':CODE},'params':candidate.params,'origin':{'source_ids':['fixture_method']}}
        doc={'schema_version':3,'kind':'pair_text','portfolio':'expanded','entries':{candidate.family:entry}}
        self.assertEqual(resolve_generated_registry('pair_text',candidate.family,doc).params,candidate.params)
        bad=deepcopy(doc);bad['entries'][candidate.family]['spec']['code']=CODE.replace('alpha=0.001','alpha=0.002')
        with self.assertRaises(ValueError):resolve_generated_registry('pair_text',candidate.family,bad)
        with self.assertRaises(ValueError):resolve_generated_registry('single_text',candidate.family,doc)

    def test_mock_search_generation_worker_feedback_confirmation_refit(self):
        result,client,searcher=self.run_mock('complete')
        self.assertTrue(result['valid_policy_run']);self.assertTrue(result['refit_succeeded'])
        self.assertEqual(result['refinement']['generated_candidates'],1)
        self.assertEqual(result['refinement']['generated_candidates_completed'],1)
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],11)
        self.assertEqual([c['role'] for c in client.calls],['evidence_planner','method_researcher','method_codegen','evidence_planner','evidence_planner'])
        directory=self.root/'complete';rows=json.loads((directory/'trials.json').read_text())
        self.assertTrue(rows[2]['candidate'].startswith('generated_'))
        subsequent=client.calls[-1]['payload']
        self.assertEqual(subsequent['latest_result']['latest_trial'],3)
        self.assertTrue(any(p['candidate'].startswith('generated_') for p in subsequent['plans']))
        self.assertEqual(len(searcher.calls),1)
        self.assertFalse(result['refinement']['current_proposal'] is None)

    def test_retrieval_failure_preserves_prior_candidates_and_marks_research_incomplete(self):
        result,client,searcher=self.run_mock('search_failure',searcher=FixtureSearch(fail=True))
        self.assertFalse(result['valid_policy_run']);self.assertTrue(result['completed'])
        self.assertEqual(result['refinement']['status'],'research_failed_incumbent_retained')
        self.assertFalse(any(c['role']=='method_codegen' for c in client.calls))
        self.assertEqual(result['refinement']['generated_candidates'],0)

    def test_bad_generated_code_is_preserved_without_running_or_retrying_it(self):
        result,client,searcher=self.run_mock('code_failure',client=ScriptClient(code=CODE+'\nopen("/secret")'))
        self.assertFalse(result['valid_policy_run']);self.assertEqual(result['refinement']['generated_candidates'],0)
        trace=json.loads((self.root/'code_failure/research_trace.json').read_text())
        self.assertTrue(any(r['phase']=='code_proposed' for r in trace))
        self.assertEqual(sum(c['role']=='method_codegen' for c in client.calls),1)

    def test_researcher_can_skip_without_forced_search_or_revision(self):
        result,client,searcher=self.run_mock('skip',client=ScriptClient(skip=True))
        self.assertTrue(result['valid_policy_run']);self.assertEqual(result['refinement']['status'],'skipped_with_reason')
        self.assertEqual(searcher.calls,[]);self.assertFalse(any(c['role']=='method_codegen' for c in client.calls))

    def test_arxiv_response_and_crossref_fallback_are_distinguished(self):
        xml=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1234.5678</id><title>Mock method</title><summary>Mock abstract</summary><published>2020-01-01</published></entry></feed>'
        searcher=LiteratureSearch()
        with patch.object(searcher,'_fetch',return_value=(xml,{'sha256':hashlib.sha256(xml).hexdigest()})) as fetched:
            result=searcher.search('generic text classification')
        self.assertEqual(fetched.call_args.args[1]['search_query'],'all:text AND all:classification')
        self.assertEqual(result['network_requests'],1);self.assertEqual(result['sources'][0]['provider'],'arxiv')
        body=json.dumps({'message':{'items':[{'DOI':'10.fixture/example','title':['Mock crossref'],'URL':'https://doi.org/10.fixture/example'}]}}).encode()
        with patch.object(searcher,'_fetch',side_effect=[RuntimeError('mock transport'),(body,{'sha256':hashlib.sha256(body).hexdigest()})]):
            result=searcher.search('generic text classification')
        self.assertEqual(result['network_requests'],2);self.assertEqual(result['sources'][0]['provider'],'crossref')

    def test_search_failure_and_invalid_queries_do_not_invent_sources(self):
        searcher=LiteratureSearch()
        for query in ('','x','a\nbbbbbbbbb'):
            with self.assertRaises(ValueError):searcher.search(query)
        with patch.object(searcher,'_fetch',side_effect=RuntimeError('mock offline')):
            with self.assertRaises(RuntimeError):searcher.search('generic classification method')
        self.assertEqual(searcher.records[-1]['sources'],[]);self.assertEqual(searcher.records[-1]['network_requests'],2)

    def test_cli_refuses_unbudgeted_research_modes_before_credentials(self):
        for argv in [['--planner','local'],['--planner','llm','--architecture','reviewer'],['--planner','llm','--trials','2']]:
            args=build_parser().parse_args(['run','notread.zip','--run-dir','notcreated','--refinement','research',*argv])
            with self.assertRaisesRegex(ValueError,'Research requires'):_validate(args)

    def test_public_fetch_uses_separate_credential_free_session_and_saves_raw_hash(self):
        class Response:
            status_code=200;url='https://export.arxiv.org/api/query?search_query=generic'
            def raise_for_status(self):pass
            def iter_content(self,size):yield b'public metadata'
        class Session:
            def __init__(self):self.trust_env=True;self.record=None
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def get(self,*args,**kwargs):self.record=kwargs;return Response()
        session=Session();searcher=LiteratureSearch(self.root/'downloads')
        with patch('superagent.web_research.requests.Session',return_value=session):
            body,receipt=searcher._fetch('https://export.arxiv.org/api/query',{'search_query':'generic'})
        self.assertFalse(session.trust_env);self.assertNotIn('Authorization',session.record['headers'])
        self.assertFalse(session.record['allow_redirects']);self.assertFalse(receipt['authorization_header_sent'])
        self.assertEqual(Path(receipt['raw_body_path']).read_bytes(),body)
        self.assertEqual(receipt['sha256'],hashlib.sha256(body).hexdigest())

    def test_ordinary_and_research_roles_share_request_and_course_budget(self):
        from superagent.experiment_llm import ExperimentLLM
        from superagent.task_budget import TaskBudgetLedger,TaskBudgetAccount
        ledger=TaskBudgetLedger.initialize(self.root/'budget.sqlite3',new_project=True)
        account=TaskBudgetAccount(ledger,'unit_research',str(self.root/'unit_run'))
        key=self.root/'fake_key.txt';key.write_text('synthetic-unit-key')
        client=ExperimentLLM('https://school.example.invalid/chat',str(key),'DeepSeek-V4.1-Flash',.10,
            max_requests=2,allowed_served_models={'DeepSeek-V4.1-Flash'},account=account)
        (self.root/'logs').mkdir();logger=RunLogger(self.root/'logs');logger.initialize_usage_ledger()
        class Response:
            ok=True;status_code=200
            def raise_for_status(self):pass
            def json(self):return {'model':'DeepSeek-V4.1-Flash','usage':{'prompt_tokens':100,'completion_tokens':10},
                'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]}
        with patch('superagent.experiment_llm.requests.post',return_value=Response()) as post:
            client.ask({},logger,'evidence_planner');client.ask({},logger,'method_researcher')
            with self.assertRaisesRegex(RuntimeError,'request-count'):client.ask({},logger,'method_codegen')
        self.assertEqual(post.call_count,2);self.assertEqual(ledger.project_summary()['requests'],2)
        self.assertAlmostEqual(ledger.project_summary()['known_course_cost_rmb'],.00028)


if __name__=='__main__':unittest.main()
