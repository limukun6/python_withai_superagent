import unittest

from tools.ab_deployment_criteria import assess,ranking_key


def pair(task,score,seconds=10,split=1):
    return dict(task=task,split=split,metric='accuracy',baseline_score=.5,score=score,
                baseline_seconds=10,seconds=seconds,requests=0)


class TestDeploymentCriteria(unittest.TestCase):
    def test_quality_thresholds_are_inclusive_and_use_task_means(self):
        result=assess([pair('a',.502),pair('b',.504),pair('b',.5,split=2),pair('c',.495)])
        self.assertTrue(result['quality_condition'])
        self.assertEqual(result['tasks_gaining_at_least_0_2pp'],2)
        self.assertFalse(assess([pair('a',.504),pair('b',.5),pair('c',.505),pair('d',.4949)])['quality_condition'])

    def test_one_large_gain_or_a_quick_but_regressing_run_does_not_qualify(self):
        self.assertFalse(assess([pair('one',.9),pair('other',.5)])['meets_measured_criteria'])
        bad=assess([pair('one',.499,7),pair('other',.4979,7)])
        self.assertFalse(bad['efficiency_condition'])
        self.assertTrue(assess([pair('one',.498,8),pair('other',.5,8)])['efficiency_condition'])

    def test_risks_and_time_tie_break(self):
        fast=assess([pair('first',.47,8),pair('second',.6,8)])
        slow=assess([pair('first',.47,9),pair('second',.6,9)])
        self.assertEqual([r['task'] for r in fast['risk_splits']],['first'])
        self.assertLess(ranking_key(fast),ranking_key(slow))
        with self.assertRaisesRegex(ValueError,'Repeated'):
            assess([pair('x',.51),pair('x',.52)])
