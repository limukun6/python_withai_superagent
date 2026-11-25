import copy
import unittest

from superagent.planner_review import PlannerReview


def option(kind='explore',value=.5):
    return dict(action=kind, members=['model'],split='primary' if kind=='explore' else None,
                utility=value,evidence={'novelty':.5,'expected':.5} if kind=='explore' else {})


class PlannerReviewTests(unittest.TestCase):
    def observe(self,review,trial,value):
        review.before_fit(trial,option())
        review.observed(trial,value)

    def test_two_future_results_precede_adoption_and_successor_can_propose(self):
        review=PlannerReview()
        for trial in (1,2):
            self.observe(review,trial,.8)
        self.assertEqual(review.generation,0)
        self.assertEqual(review.pending['fit_trials'],[1,2])
        for trial in (3,4):
            review.before_fit(trial,option())
            ticket=copy.deepcopy(review.ticket)
            self.assertEqual(ticket['parent'],0)
            self.assertGreater(ticket['child_forecast'],ticket['parent_forecast'])
            review.observed(trial,.8)
        self.assertEqual(review.generation,1)
        self.assertEqual(review.versions[0]['status'],'accepted')
        self.assertEqual(review.pending['parent'],1)
        self.assertGreater(review.pending['proposal_step'],review.versions[0]['proposal_step'])
        for version in review.versions:
            self.assertTrue(all(row['trial']>max(version['fit_trials']) for row in version['evaluation']))

    def test_a_misleading_successor_is_rejected_and_cannot_change_actions(self):
        review=PlannerReview()
        original=[option('stop',.54),option()]
        for trial,score in enumerate((.8,.8,.2,.2),1):
            self.observe(review,trial,score)
        self.assertEqual(review.generation,0)
        self.assertEqual(review.versions[0]['status'],'rejected')
        self.assertIs(review.menu(original),original)
        self.assertEqual(review.pending['parent'],0)

    def test_admitted_menu_and_base_evidence_remain_intact(self):
        review=PlannerReview()
        for trial in range(1,5):
            self.observe(review,trial,.8)
        original=[option('stop',.54),option()]
        updated=review.menu(original)
        self.assertEqual(updated[0]['action'],'explore')
        self.assertEqual(sorted(a['action'] for a in original),sorted(a['action'] for a in updated))
        self.assertEqual(original[1]['utility'],.5)
        review.before_fit(5,option('confirm'))
        review.observed(5,.9)
        self.assertEqual(len(review.observations),4)
        self.assertEqual(PlannerReview().record()['generation'],0)
