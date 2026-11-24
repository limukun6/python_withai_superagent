import unittest

from superagent.plan_fragment import (fragment_status, next_step, validate_fragment)

MENU = [{'id': 'word_char_C0.05', 'params': {}}, {'id': 'word_char_C0.1', 'params': {}}]
STATE = {
    'history': [
        {'trial': 1, 'candidate': 'word_C0.05', 'status': 'completed', 'search_score': 0.6228},
        {'trial': 2, 'candidate': 'word_char_C0.05', 'status': 'completed', 'search_score': 0.6520},
    ],
    'available_candidates': MENU,
}
EXPLORE = {'action': 'explore', 'candidate_id': 'word_char_C0.05'}


def fragment(*steps):
    return {'steps': list(steps)}


class ValidationTests(unittest.TestCase):
    def test_a_two_step_fragment_is_accepted(self):
        normalised, errors = validate_fragment(
            fragment({'candidate': 'word_char_C0.05', 'requires': {'best_completed_trial': 1}},
                     {'candidate': 'word_char_C0.1', 'requires': {'best_completed_trial': 2}}),
            {c['id'] for c in MENU})
        self.assertEqual(errors, [])
        self.assertEqual([s['step'] for s in normalised['steps']], [1, 2])

    def test_more_than_two_steps_is_rejected(self):
        _, errors = validate_fragment(
            fragment(*[{'candidate': 'word_char_C0.05', 'requires': {'minimum_best_score': .5}}] * 3),
            {c['id'] for c in MENU})
        self.assertTrue(any('1..2 steps' in e for e in errors))

    def test_free_text_conditions_are_rejected(self):
        _, errors = validate_fragment(
            fragment({'candidate': 'word_char_C0.05', 'requires': {'because': 'it looked better'}}),
            {c['id'] for c in MENU})
        self.assertTrue(any('unsupported conditions' in e for e in errors))

    def test_candidates_outside_the_menu_are_rejected(self):
        _, errors = validate_fragment(
            fragment({'candidate': 'catboost_depth8', 'requires': {'minimum_best_score': .5}}),
            {c['id'] for c in MENU})
        self.assertTrue(any('eligible menu' in e for e in errors))

    def test_duplicate_candidates_are_rejected(self):
        _, errors = validate_fragment(
            fragment({'candidate': 'word_char_C0.05', 'requires': {'minimum_best_score': .5}},
                     {'candidate': 'word_char_C0.05', 'requires': {'best_completed_trial': 1}}),
            {c['id'] for c in MENU})
        self.assertTrue(any('repeats' in e for e in errors))

    def test_an_empty_condition_set_is_rejected(self):
        _, errors = validate_fragment(
            fragment({'candidate': 'word_char_C0.05', 'requires': {}}), {c['id'] for c in MENU})
        self.assertTrue(any('nonempty' in e for e in errors))


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.good, errors = validate_fragment(
            fragment({'candidate': 'word_char_C0.05', 'requires': {'best_completed_trial': 2}},
                     {'candidate': 'word_char_C0.1', 'requires': {'best_completed_trial': 2}}),
            {c['id'] for c in MENU})
        self.assertEqual(errors, [])

    def test_a_valid_step_names_its_candidate(self):
        report = fragment_status(self.good, STATE, EXPLORE, 0)
        self.assertEqual(report['status'], 'valid')
        self.assertEqual(report['candidate'], 'word_char_C0.05')
        self.assertTrue(all(c['status'] == 'holds' for c in report['checks']))

    def test_a_scheduler_stop_supersedes_the_fragment(self):
        report = fragment_status(self.good, STATE, {'action': 'stop'}, 0)
        self.assertEqual(report['status'], 'superseded_by_scheduler')
        self.assertIsNone(report['candidate'])

    def test_a_scheduler_confirm_supersedes_the_fragment(self):
        self.assertEqual(fragment_status(self.good, STATE, {'action': 'confirm'}, 0)['status'],
                         'superseded_by_scheduler')

    def test_new_evidence_invalidates_the_remaining_step(self):
        moved = {'history': STATE['history'] + [
            {'trial': 3, 'candidate': 'word_char_C0.1', 'status': 'completed', 'search_score': 0.71}],
            'available_candidates': MENU}
        report = fragment_status(self.good, moved, EXPLORE, 1)
        self.assertEqual(report['status'], 'invalidated')
        self.assertIn('best_completed_trial', report['reason'])

    def test_a_score_floor_is_satisfied_not_violated_by_better_evidence(self):
        floor, _ = validate_fragment(fragment({'candidate': 'word_char_C0.05',
                                                'requires': {'minimum_best_score': 0.6}}),
                                    {c['id'] for c in MENU})
        self.assertEqual(fragment_status(floor, STATE, EXPLORE, 0)['status'], 'valid')
        below, _ = validate_fragment(fragment({'candidate': 'word_char_C0.05',
                                             'requires': {'minimum_best_score': 0.9}}),
                                   {c['id'] for c in MENU})
        self.assertEqual(fragment_status(below, STATE, EXPLORE, 0)['status'], 'invalidated')

    def test_a_candidate_leaving_the_menu_invalidates_the_step(self):
        shrunk = {'history': STATE['history'], 'available_candidates': [MENU[1]]}
        self.assertEqual(fragment_status(self.good, shrunk, EXPLORE, 0)['status'], 'invalidated')

    def test_a_spent_fragment_is_consumed(self):
        self.assertEqual(fragment_status(self.good, STATE, EXPLORE, 2)['status'], 'consumed')

    def test_next_step_returns_none_when_superseded(self):
        self.assertIsNone(next_step(self.good, STATE, {'action': 'stop'}, 0))
        self.assertEqual(next_step(self.good, STATE, EXPLORE, 0), 'word_char_C0.05')


if __name__ == '__main__':
    unittest.main()
