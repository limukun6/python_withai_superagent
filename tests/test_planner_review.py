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

    def test_active_credit_leaves_room_for_another_exploration_confirmation_and_refit(self):
        review = PlannerReview()
        for trial in (1, 2):
            self.observe(review, trial, .8)
        menu = []
        for kind, name, value in [('confirm', 'leader', .515), ('explore', 'first', .50),
                                  ('explore', 'second', .49), ('stop', 'leader', .48)]:
            row = option(kind, value)
            row['members'] = [name]
            row['evidence']['reserved_refits'] = 2
            menu.append(row)
        caps = {name: {'classifier_fits': 1} for name in ('leader', 'first', 'second')}
        self.assertEqual(review.seek_evidence(menu, 4, caps)[0]['action'], 'confirm')
        active = review.seek_evidence(menu, 5, caps)
        self.assertEqual(active[0]['members'], ['first'])
        info = active[0]['evidence']['information_value']
        self.assertEqual(info['required_fits_including_followup_and_refit'], 5)
        self.assertEqual(info['followup_action'], 'confirm')
        self.assertGreater(info['credit'], 0)
        self.assertNotIn('information_value', menu[1]['evidence'])
        self.assertEqual(len(active), len(menu))
        caps['first']['classifier_fits'] = 3
        self.assertEqual(review.seek_evidence(menu, 5, caps)[0]['action'], 'confirm')

    def test_no_information_credit_without_disagreement_or_a_distinct_followup(self):
        review = PlannerReview()
        for trial in (1, 2):
            self.observe(review, trial, .5)
        rows = [option('stop', .51)]
        for name in ('one', 'two', 'three'):
            row = option()
            row['members'] = [name]
            row['evidence']['reserved_refits'] = 1
            rows.append(row)
        caps = {r['members'][0]: {'classifier_fits': 1} for r in rows}
        ranked = review.seek_evidence(rows, 16, caps)
        self.assertEqual(ranked[0]['action'], 'stop')
        self.assertEqual(ranked[1]['evidence']['information_value']['credit'], 0)
        self.assertEqual(review.seek_evidence(rows[:2], 16, caps), rows[:2])

    def test_passive_and_active_modes_round_trip_both_command_parsers(self):
        from superagent.cli import build_parser
        from superagent.planner_review import review_flags
        for flag, expected in [(None, False), ('--self-review', True), ('--active-self-review', 'active'),
                               ('--protected-self-review', 'protected')]:
            args = build_parser().parse_args(['unified', 'input.zip', '--run-dir', 'fresh', *([flag] if flag else [])])
            self.assertEqual(args.self_review, expected)
            self.assertEqual(review_flags(expected), [flag] if flag else [])
