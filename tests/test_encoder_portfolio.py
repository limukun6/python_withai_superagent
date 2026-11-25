import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from scripts.unified_encoder_portfolio import training_roles
from tools.encoder_assets import BGE, MPNET, ROBERTA
from tools.feedback_policy_pool import evidence_pool, grouped_error
from tools.portfolio_encoders import PortfolioFeatures, pair_key
from tools.preference_portfolio import choose, outputs, panel, probabilities, weights


class EncoderPortfolioTests(unittest.TestCase):
    def test_learn_then_confirm_can_accept_or_retain_the_pair(self):
        rng = np.random.default_rng(993)
        name = ["original one", "original two", "alternative alpha", "alternative beta"]
        p = rng.uniform(.05, .95, (420, len(name)))
        pairs = [("an earlier headline", "a different wording")] * len(p)
        y = (p[:, 3] > .5).astype(int)
        learning = panel(p[:220], pairs[:220])
        confirm = panel(p[220:], pairs[220:])
        selected = choose(name, learning, y[:220], confirm, y[220:])
        self.assertEqual(selected["chosen"], "single")
        self.assertEqual(selected["rules"]["single"]["indices"], [3])
        np.testing.assert_array_equal(outputs(selected, confirm)["agent"], y[220:])
        rejected = choose(name, learning, y[:220], confirm, confirm["base"])
        self.assertEqual(rejected["chosen"], "baseline")
        np.testing.assert_array_equal(outputs(rejected, confirm)["agent"], confirm["base"])

    def test_probability_weights_are_normalized_and_do_not_depend_on_side_order(self):
        r = np.random.default_rng(7802)
        p = r.uniform(.02,.98,(300,6))
        rows = [("A shorter title", "Second and longer title") if i%3 else ("Zero", "One") for i in range(300)]
        y = (p[np.arange(300), np.where(np.abs(p[:, 2]-.5)>.25, 2, 5)]>.5).astype(int)
        data = panel(p, rows)
        choice = choose(list("abcdef"),data,y,data,y)
        flipped = panel(1-p,[(b,a) for a,b in rows])
        policies = choice["rules"]
        for rule in policies.values():
            w = weights(rule,data)
            self.assertTrue((w>=0).all())
            np.testing.assert_allclose(w.sum(1),1)
            np.testing.assert_allclose(w,weights(rule,flipped),atol=1e-10)
            np.testing.assert_allclose(probabilities(rule,data),1-probabilities(rule,flipped),atol=1e-10)
        self.assertGreater(np.max(weights(policies["contextual"],data).std(0)), .01)

    def test_joint_text_features_are_antisymmetric_including_ties(self):
        rows = [("one","two"),("two","one"),("one","one")]
        def embedded(name, keys):
            self.assertEqual(name, ROBERTA)
            self.assertEqual(keys[0],keys[1])
            return np.ones((len(keys),768), dtype=np.float32)
        with patch("tools.portfolio_encoders.frozen_vectors",side_effect=embedded):
            out,_ = PortfolioFeatures(ROBERTA).semantic_blocks(rows,[])
        v = out[0].toarray()
        np.testing.assert_array_equal(v[0],-v[1])
        np.testing.assert_array_equal(v[2],0)
        self.assertNotEqual(pair_key('a,b','c'),pair_key('a','b,c'))
        for member in (BGE, MPNET):
            d = PortfolioFeatures(member)
            with tempfile.TemporaryDirectory() as temp:
                artifact = Path(temp)/"encoder.joblib"
                joblib.dump(d,artifact)
                self.assertEqual(joblib.load(artifact).get_params()["name"],member)

    def test_agent_actually_picks_contextual_fusion_for_complementary_specialists(self):
        rng = np.random.default_rng(2424)
        count = 720
        y = rng.integers(0, 2, count)
        related = np.arange(count)%2 == 0
        rows = [("local museum opens today", "local museum opens tomorrow") if same
                else ("city vote", "festival winter") for same in related]
        right = np.where(y==1, .8, .2)
        a = np.where(related, right, 1-right)
        p = np.column_stack((rng.uniform(.38,.62,count), rng.uniform(.38,.62,count), a, 1-a))
        fit, check = panel(p[:400],rows[:400]), panel(p[400:],rows[400:])
        choice = choose(["incumbent-A","incumbent-B","expert-A","expert-B"],fit,y[:400],check,y[400:])
        self.assertEqual(choice["chosen"],"contextual")
        adopted = outputs(choice,check)["agent"]
        self.assertGreater(np.mean(adopted==y[400:]),.9)

    def test_cold_roles_exclude_the_entire_audit_and_each_others_groups(self):
        groups = np.asarray([f"headline-{i//2}" for i in range(160)])
        labels = np.arange(len(groups))%2
        permitted = np.arange(20,len(groups))
        roles = training_roles(groups,labels,permitted)
        self.assertEqual(sorted(np.concatenate(list(roles.values())).tolist()),permitted.tolist())
        used = [set(groups[v]) for v in roles.values()]
        self.assertEqual(sum(map(len,used)),len(set.union(*used)))
        self.assertTrue(set(groups[:20]).isdisjoint(set.union(*used)))
        self.assertIsNone(training_roles(groups,labels,np.arange(8)))

    def test_grouped_feedback_does_not_reward_duplicate_or_unhelpful_models(self):
        effect = np.asarray([.1,-.1,.2,-.2])
        self.assertAlmostEqual(grouped_error(effect,np.arange(4)),
                               grouped_error(np.repeat(effect,3),np.repeat(np.arange(4),3)))
        p = np.tile([.75,.75,.75,.75],(80,1))
        x = panel(p,[("a","b")]*len(p))
        y = x["base"]
        choice = choose(list("abcd"),x,y,x,y)
        pool = evidence_pool(choice,x,y,np.arange(80),x,y,np.arange(80))
        self.assertFalse(pool["adopted"])
        self.assertEqual(pool["deployed_weights"],{"baseline":1.})
