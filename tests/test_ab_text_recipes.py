import copy
import unittest

from superagent import ab_text_recipes as recipes
from superagent.models import search_candidates


def observation(i, analyzer, score, train, strength=.3):
    return dict(candidate='observed_'+str(i),trial=i,status='completed',search_score=score,
                params={'C':strength,'representation':analyzer},train_metrics={'score':train})


class TestBoundedCompositions(unittest.TestCase):
    def setUp(self):
        self.menu = {c.family:dict(id=c.family,kind='single_text',params=c.params) for c in recipes.candidates('single_text')}
        self.ids = tuple(self.menu)

    def test_existing_factory_produces_eight_unique_single_fit_combinations(self):
        for kind in ('single_text','pair_text'):
            choices = recipes.candidates(kind)
            self.assertEqual(len({c.family for c in choices}),8)
            full = [c.family for c in search_candidates(kind,'all')]
            self.assertEqual(len(full),len(set(full)))
            for c in choices:
                self.assertEqual(c.params['classifier'],'logistic')
                self.assertEqual(c.params['dimensions'],64)
        self.assertEqual(recipes.candidates('image'),[])

    def test_same_menu_different_prior_observations_change_the_composed_spec(self):
        observations = [observation(1,'word',.72,.94),observation(2,'word_char',.7,.85)]
        before = copy.deepcopy((observations,self.menu))
        a = recipes.propose(self.ids,observations,self.menu)
        self.assertEqual((a['chosen_spec']['analyzer'],a['chosen_spec']['representation']),('char_wb','latent'))
        self.assertEqual(before,(observations,self.menu))
        other = [observation(1,'word',.9,.93,1.),observation(2,'char',.87,.93)]
        b = recipes.propose(self.ids,other,self.menu)
        self.assertEqual((b['chosen_spec']['analyzer'],b['chosen_spec']['representation'],b['chosen_spec']['C']),('word','hybrid',1.))
        self.assertEqual(b['evidence_trials'],[1,2])
        self.assertNotEqual(a['chosen_id'],b['chosen_id'])

    def test_unaffordable_or_tried_targets_only_fall_back_within_the_supplied_list(self):
        history = [observation(1,'word',.7,.75),observation(2,'combined',.76,.77)]
        original = recipes.propose(self.ids,history,self.menu)
        allowed = tuple(n for n in self.ids if n!=original['chosen_id'])
        fallback = recipes.propose(allowed,history,self.menu)
        self.assertFalse(fallback['target_eligible'])
        self.assertEqual(fallback['target_id'],original['chosen_id'])
        self.assertIn(fallback['chosen_id'],allowed)
        self.assertEqual(fallback['chosen_spec']['analyzer'],original['chosen_spec']['analyzer'])
        self.assertIsNone(recipes.propose(self.ids,history[:1],self.menu))
        self.assertIsNone(recipes.propose(('an_existing_route',*self.ids),history,self.menu))
