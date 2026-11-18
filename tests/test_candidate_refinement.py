import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from superagent.candidate_refinement import CandidateRegistry,make_spec,materialize,resolve_registered,MAX_GENERATED
from superagent.models import search_candidates,predict_with_model
from superagent.evidence_agent import route_of
from superagent.local_search import LocalSearchPolicy
from superagent.benchmark import run_one,train_candidate
from superagent.data import DevelopmentTask
from superagent.cli import build_parser,_validate


def observation(candidate,trial,score,status='completed'):
    return {'trial':trial,'candidate':candidate.family,'params':candidate.params,'status':status,'search_accuracy':score}


class CandidateRefinementTests(unittest.TestCase):
    def test_materialized_candidate_changes_only_c_and_id_is_canonical(self):
        spec=make_spec('pair_text','expanded','word_char_C0.3',math.sqrt(.1*.3))
        candidate=materialize('pair_text',spec)
        self.assertEqual(spec['template_id'],'word_char_C0.05')
        self.assertEqual(candidate.params['C'],candidate.fit_predict.keywords['c'])
        self.assertEqual(candidate.params['C'],float(f'{math.sqrt(.03):.12g}'))
        self.assertEqual(route_of(candidate.family),'word_char')
        self.assertEqual(candidate.family,materialize('pair_text',copy.deepcopy(spec)).family)
        template=next(c for c in search_candidates('pair_text','expanded') if c.family==spec['template_id'])
        self.assertEqual({k:v for k,v in template.params.items() if k!='C'},{k:v for k,v in candidate.params.items() if k!='C'})

    def test_invalid_templates_values_schema_and_registry_tampering_fail_before_fitting(self):
        for c in (0,-1,True,float('nan'),float('inf'),.05,10.,.3,100.):
            with self.subTest(c=c),self.assertRaises(ValueError):make_spec('pair_text','expanded','word_C0.05',c)
        with self.assertRaises(ValueError):make_spec('numeric_tabular','expanded','word_C0.05',.2)
        with self.assertRaises(ValueError):make_spec('pair_text','research','word_C0.05',.2)
        spec=make_spec('pair_text','expanded','word_C0.05',.2)
        with self.assertRaises(ValueError):materialize('pair_text',{**spec,'python':'arbitrary code'})
        with self.assertRaises(ValueError):materialize('single_text',spec)
        candidate=materialize('pair_text',spec)
        document={'schema_version':1,'kind':'pair_text','portfolio':'expanded','entries':{
            candidate.family:{'spec':spec,'params':candidate.params,'origin':{}}}}
        self.assertEqual(resolve_registered('pair_text',candidate.family,document).params,candidate.params)
        changed=copy.deepcopy(document);changed['entries'][candidate.family]['params']['C']=.25
        with self.assertRaises(ValueError):resolve_registered('pair_text',candidate.family,changed)

    def test_generation_requires_two_completed_same_route_observations_and_does_not_remove_old_routes(self):
        catalog={c.family:c for c in search_candidates('pair_text','expanded')};a=catalog['word_char_C0.1'];b=catalog['word_char_C0.3']
        with tempfile.TemporaryDirectory() as temp:
            registry=CandidateRegistry('pair_text','expanded',Path(temp)/'registry.json')
            self.assertEqual(registry.propose([observation(a,1,.65)]),[])
            self.assertEqual(registry.propose([observation(a,1,.65),observation(b,2,None,'failed')]),[])
            history=[observation(a,1,.65),observation(b,2,.67)]
            added=registry.propose(history);self.assertEqual(len(added),1)
            origin=added[0]['origin'];self.assertEqual(origin['reference_trial'],2);self.assertEqual(origin['neighbor_trial'],1)
            self.assertEqual(origin['C_endpoints'],[.3,.1]);self.assertEqual(registry.propose(history),[])
            self.assertEqual(len(registry.catalog),22);self.assertEqual(set(c.family for c in registry.catalog),set(catalog))

    def test_caps_and_generated_history_validation_prevent_unbounded_or_unregistered_refinement(self):
        catalog={c.family:c for c in search_candidates('pair_text','expanded')}
        history=[observation(catalog['word_char_C0.1'],1,.65),observation(catalog['word_char_C0.3'],2,.67)]
        with tempfile.TemporaryDirectory() as temp:
            registry=CandidateRegistry('pair_text','expanded',Path(temp)/'registry.json')
            for step in range(6):
                added=registry.propose(history);self.assertLessEqual(len(added),2)
                for entry in added:
                    c=materialize('pair_text',entry['spec']);history.append(observation(c,len(history)+1,.68+.01*step))
            self.assertEqual(len(registry.document['entries']),MAX_GENERATED)
            self.assertEqual(registry.propose(history),[])
        with tempfile.TemporaryDirectory() as temp:
            registry=CandidateRegistry('pair_text','expanded',Path(temp)/'registry.json')
            bad=copy.deepcopy(history[:2]);bad[0]['params']['C']=.12345
            with self.assertRaises(ValueError):registry.propose(bad)

    def test_policy_catalog_expansion_preserves_rng_state_old_coordinates_and_available_routes(self):
        catalog=search_candidates('pair_text','expanded')
        metadata=[{'id':c.family,'route':route_of(c.family),'params':c.params} for c in catalog]
        policy=LocalSearchPolicy('greedy_routes',metadata,707);policy.rng.random();before=policy.rng.getstate();coordinates=copy.deepcopy(policy.coordinates)
        candidate=materialize('pair_text',make_spec('pair_text','expanded','word_C0.05',.2))
        item={'id':candidate.family,'route':route_of(candidate.family),'params':candidate.params}
        policy.add_candidates([item]);self.assertEqual(before,policy.rng.getstate())
        self.assertEqual(coordinates,{k:policy.coordinates[k] for k in coordinates});self.assertEqual(len(policy.catalog),23)
        with self.assertRaises(ValueError):policy.add_candidates([item])

    def test_blind_policy_is_rejected_before_data_or_credentials_are_touched(self):
        args=build_parser().parse_args(['run','/nonexistent-task','--refinement','midpoints'])
        with self.assertRaisesRegex(ValueError,'feedback local policy'):_validate(args)
        for policy in ('random','local_balanced_blind','evidence_blind'):
            with self.subTest(policy=policy),self.assertRaises(ValueError):
                run_one(None,policy,707,Path('/unused'),None,refinement='midpoints',portfolio='expanded')

    def test_worker_confirmation_and_explicit_refit_resolve_identical_generated_configuration(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(120,4));y=(x[:,0]+.4*x[:,1]>0).astype(int)
        task=DevelopmentTask('numeric_tabular','synthetic',x,y,x[:18],y[:18],{})
        catalog=search_candidates('numeric_tabular','expanded')[:3]
        with tempfile.TemporaryDirectory() as temp,patch('superagent.benchmark.search_candidates',return_value=catalog):
            directory=Path(temp)/'run'
            result=run_one(task,'local_greedy_routes',707,directory,None,max_trials=3,anchor_first=True,
                confirmation_repeats=3,confirmation_top_k=3,refit_full_train=True,portfolio='expanded',refinement='midpoints')
            rows=json.loads((directory/'trials.json').read_text());generated=[r for r in rows if '_refined_' in r['candidate']]
            self.assertEqual(len(generated),1);row=generated[0]
            self.assertEqual(row['resolved_configuration']['origin'],'registry')
            model=joblib.load(directory/f'trial_{row["trial"]}.joblib')['model']
            self.assertEqual(model.named_steps['classifier'].C,row['params']['C'])
            for index in (1,2):
                metrics=joblib.load(directory/f'confirmation_{index}_{row["trial"]}.joblib')['metrics']
                self.assertEqual(metrics['resolved_configuration']['params'],row['params'])
                self.assertEqual(metrics['resolved_configuration']['origin'],'registry')
            payload=directory/'extra_refit_data.joblib'
            joblib.dump({'kind':task.kind,'mode':'refit','X_train':x,'y_train':y,'X_search':x[:1]},payload)
            fitted,metrics=train_candidate(payload,row['candidate'],directory/'extra_refit.joblib',60)
            self.assertEqual(metrics['status'],'completed');self.assertEqual(metrics['resolved_configuration']['params'],row['params'])
            self.assertEqual(fitted.named_steps['classifier'].C,row['params']['C'])
            (directory/'candidate_registry.json').rename(directory/'registry_hidden.json')
            with patch.object(LogisticRegression,'fit',side_effect=AssertionError('Prediction cannot refit')):
                pred=predict_with_model(task.kind,fitted,x[:18]);self.assertEqual(pred.shape,(18,))
        self.assertTrue(result['confirmation']['valid_confirmation']);self.assertTrue(result['refit_succeeded'])
        self.assertTrue(result['prediction_evidence_audit']['integrity_valid'])
        self.assertEqual(result['classifier_fit_accounting']['completed_classifier_fits'],10)
        self.assertEqual(result['refinement']['generated_candidates'],1)


if __name__=='__main__':unittest.main()
