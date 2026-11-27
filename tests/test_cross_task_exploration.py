import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import unified_agent
from scripts.replay_planner_review_archive import propose
from scripts.verify_planner_lineage import verdict
from superagent.experiment_policy import identity
from superagent.exploration_rule import apply_rule, forecast, load_rule, snapshot


class CrossTaskExplorationTests(unittest.TestCase):
    def test_identity_and_cold_start_preserve_ties_and_all_existing_choices(self):
        menu = [dict(action='confirm', split='confirmation_1', utility=.68, members=['x'], evidence={}),
                dict(action='explore', split='primary', utility=.67, members=['y'],
                     evidence=dict(expected=.65, novelty=.8, reserved_refits=1))]
        self.assertIs(apply_rule(menu, [0, 0, 0], 4), menu)
        self.assertIs(apply_rule(menu, [.1, 0, 0], 0), menu)
        new = apply_rule(menu, [.1, 0, 0], 4)
        self.assertEqual([r['members'] for r in new], [['y'], ['x']])
        self.assertEqual(new[0]['evidence']['reserved_refits'], 1)
        self.assertIs(new[1], menu[0])
        self.assertNotIn('inherited_forecast_adjustment', menu[1]['evidence'])

    def test_proposal_is_bounded_and_depends_on_its_parent(self):
        trace = {'observations': [(2, dict(expected=.50, novelty=.75), .85, 3)]}
        memory = {'one': trace}
        root = np.zeros(3)
        child = propose(root, memory, .6)
        descendant = propose(child, memory, .75)
        self.assertFalse(np.array_equal(child, descendant))
        self.assertFalse(np.array_equal(propose(root, memory, .75), descendant))
        self.assertTrue(np.all(np.abs(descendant) <= .12))
        self.assertAlmostEqual(forecast(np.full(3, .12), 1, {'expected': .99, 'novelty': .8}), 1.)

    def test_course_entry_snapshots_the_rule_before_launch(self):
        with tempfile.TemporaryDirectory() as dirname:
            folder = Path(dirname)
            path = folder/'input.json'
            expected = snapshot([-.02, 0, 0])
            path.write_text(json.dumps(expected))
            args = unified_agent.prepare_training(folder/'train.zip', folder/'run', 16, 600, 'protected', path)
            path.write_text(json.dumps(snapshot([.1, 0, 0])))
            self.assertEqual(load_rule(args.exploration_rule), expected)
            protocol = json.loads((folder/'run/training_protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(protocol['exploration_rule_digest'], identity(expected))

    def test_rejects_task_lookup_fields_nonfinite_and_invalid_weights(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname)/'rule.json'
            valid = snapshot([0, 0, 0])
            for invalid in [dict(valid, model='x'), dict(valid, format='other'), dict(valid, weights={}),
                            *[dict(valid, weights=v) for v in [[True,0,0], [0,0], [.121,0,0], [float('nan'),0,0]]]]:
                with self.subTest(value=invalid):
                    path.write_text(json.dumps(invalid))
                    with self.assertRaises(ValueError):
                        load_rule(path)

    def test_adoption_requires_measured_path_change_and_accounts_for_regressions(self):
        def grade(score, count, changed=True):
            return {'parent': {'score': .70, 'fits': 10}, 'child': {'score': score, 'fits': count,
                                                                    'changed_path': changed}}
        self.assertFalse(verdict({'a': grade(.72,10,False), 'b': grade(.70,10,False)})['promote'])
        self.assertFalse(verdict({'a': grade(.75,8), 'b': grade(.68,8)})['promote'])
        self.assertTrue(verdict({'a': grade(.72,10), 'b': grade(.70,10)})['promote'])
        self.assertTrue(verdict({'a': grade(.70,8), 'b': grade(.70,9)})['cost_gate'])
