import unittest

import numpy as np

from superagent.method_choice import PAIR_ITEM, PAIR_OPPONENT, TEXT_MINILM, select_action
from superagent.sentence_interactions import interaction_features


class SentenceInteractionTests(unittest.TestCase):
    def test_pair_swap_reverses_sign_and_control_has_itemwise_scores(self):
        rng=np.random.default_rng(12)
        a,b,c=(rng.normal(size=(6,16)) for _ in range(3))
        projections=tuple(rng.normal(size=(16,8)) for _ in range(2))
        for opponent in (False,True):
            ab=interaction_features(a,b,projections,opponent)
            np.testing.assert_allclose(ab,-interaction_features(b,a,projections,opponent),atol=1e-6)
            np.testing.assert_array_equal(interaction_features(a,a,projections,opponent),0)
            cycle=ab+interaction_features(b,c,projections,opponent)+interaction_features(c,a,projections,opponent)
            if opponent:self.assertGreater(np.max(np.abs(cycle[:,16:])),1.)
            else:np.testing.assert_allclose(cycle,0,atol=1e-5)

    def test_paired_control_gets_same_opportunity(self):
        ev=dict(kind="pair_text",semantic="x",shared_errors=100,required_shared_errors=20,low_support_shared_errors=0,
                short_latin_share=0.,primary_metric="accuracy",corrected=33,lost=22,search_rows=1000,residual_shared_errors=90)
        eligible={TEXT_MINILM,PAIR_ITEM,PAIR_OPPONENT}
        for mode,second in (("opponent",PAIR_OPPONENT),("item-control",PAIR_ITEM)):
            self.assertEqual(select_action(ev,eligible,mode,[])[0],TEXT_MINILM)
            self.assertEqual(select_action(ev,eligible,mode,[TEXT_MINILM])[0],second)
            self.assertIsNone(select_action(ev|{"residual_shared_errors":9},eligible,mode,[TEXT_MINILM])[0])
