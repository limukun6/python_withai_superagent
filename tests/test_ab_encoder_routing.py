import unittest
from unittest.mock import patch

import numpy as np

from superagent.ab_encoder_routing import ConfidenceRoutedPreferenceVote, assess, select_threshold


class EncoderRoutingTests(unittest.TestCase):
    def test_switch_defaults_off_and_rejects_unmatched_scheduler_before_reading_data(self):
        from superagent.cli import build_parser, _run
        args=build_parser().parse_args(['run','missing.zip'])
        self.assertFalse(args.confidence_cascade)
        args=build_parser().parse_args(['run','missing.zip','--confidence-cascade'])
        with self.assertRaisesRegex(ValueError,'matched preference scheduler'):
            _run(args)

    def test_confirmation_uses_only_groups_absent_from_primary(self):
        matched={'promoted':True,'members':['a','b']}
        caps={k:{'resource':'local_frozen_encoder_cpu'} for k in matched['members']}
        trusted={'primary':{'groups':['seen']},'confirmation_1':{'groups':['seen','fresh']}}
        def columns(state, trusted, split, metric):
            y=np.array([1]) if split=='primary' else np.array([0,1])
            p=np.array([.9]) if split=='primary' else np.array([.1,.9])
            return y,np.ones(len(y),dtype=bool),{k:(y,p,1) for k in ('a','b')}
        with patch('superagent.ab_encoder_routing.split_columns',side_effect=columns):
            result=assess(matched,{},trusted,caps)
        self.assertEqual(result['confirmation_rows_before_filter'],2)
        self.assertEqual(result['confirmation']['rows'],1)
        self.assertEqual(result['confirmation']['correct'],1)

    def test_confirm_can_reject_and_choose_two_models(self):
        primary=({'a':np.array([.9,.1,.51,.49]),'b':np.array([.85,.15,.85,.15])},
                 np.array([1,0,1,0]),np.ones(4,dtype=bool))
        different=({'a':np.array([.55,.45,.55,.45]),'b':np.array([.05,.95,.05,.95])},
                   np.array([0,1,0,1]),np.ones(4,dtype=bool))
        self.assertEqual(select_threshold(primary,primary)['threshold'],0.)
        chosen=select_threshold(primary,different)
        self.assertEqual(chosen['first'],'a')
        self.assertFalse(chosen['adopted'])
        self.assertEqual(chosen['threshold'],1.)

    def test_preconditions_preserve_a_single_or_an_unconfirmed_pair(self):
        self.assertEqual(assess({'promoted':False},{},{},{}),{'status':'single_selected','adopted':False})
        matched={'promoted':True,'members':['a','b']}
        self.assertEqual(assess(matched,{},{},{'a':{'resource':'cpu'}})['status'],'pair_without_two_frozen_encoders')
        caps={k:{'resource':'local_frozen_encoder_cpu'} for k in matched['members']}
        same={'primary':{'groups':['x']},'confirmation_1':{'groups':['x']}}
        self.assertEqual(assess(matched,{},same,caps)['status'],'no_fresh_confirmation_groups')

    def test_physical_second_stage_handles_only_uncertain_rows_and_swap(self):
        calls=[]
        def probabilities(model,rows):
            calls.append((model,list(rows)))
            by_text={'first':{'a':.9,'b':.55,'c':.5},'second':{'b':.1,'c':.5}}
            p=np.array([by_text[model][a] for a,_ in rows])
            return np.array([0,1]),np.column_stack((1-p,p))
        model=ConfidenceRoutedPreferenceVote('pair_text',['second','first'],np.array([0,1]),1,.2)
        examples=[('a','z'),('b','y'),('c','c')]
        with patch('superagent.ab_encoder_routing.canonical_probabilities',side_effect=probabilities):
            original=model.predict_proba(examples)
            reversed_=model.predict_proba([(b,a) for a,b in examples])
            self.assertEqual(model.predict_proba([]).shape,(0,2))
        self.assertEqual(calls[1],('second',examples[1:]))
        self.assertEqual(calls[3],('second',examples[1:]))
        np.testing.assert_allclose(original,reversed_[:,::-1])
        np.testing.assert_allclose(original[:,1],[.9,.325,.5])


if __name__=='__main__':
    unittest.main()
