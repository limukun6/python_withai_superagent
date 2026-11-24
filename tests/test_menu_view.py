import json
import unittest

from superagent.ab_flash import context_for
from superagent.cli import build_parser
from superagent.menu_view import compact_menu, decode_menu, recorded_size


def menu_payload(count=29):
    """A payload shaped like the failing A151 request: a wide candidate menu."""
    def candidate(index):
        return {'id': f'word_char_{index}', 'params': {'C': [0.1, 0.3, 0.7][index % 3],
                                                       'representation': 'word_char',
                                                       'swap_augmentation': True,
                                                       'max_features': 40000}}
    candidates = [candidate(i) for i in range(count)]
    facts = {row['id']: {'classifier': 'logistic', 'source': 'registered_training_function',
                         'training_function': '_word_char_difference_text'} for row in candidates}
    return {'context_version': 'ab-history-calibrated-v2', 'task_spec': {'kind': 'pair_text'},
            'history': [{'trial': 1, 'candidate': 'word_C0.05', 'search_score': 0.6397}],
            'available_candidates': candidates, 'candidate_facts': facts,
            'budget': {'fit_limit': 3}, 'instruction': 'Return JSON.'}


class RoundTripTests(unittest.TestCase):
    def test_decoding_restores_the_original_payload_exactly(self):
        original = menu_payload()
        self.assertEqual(decode_menu(compact_menu(original)), original)

    def test_every_candidate_and_parameter_survives(self):
        original = menu_payload()
        restored = decode_menu(compact_menu(original))
        self.assertEqual([c['id'] for c in restored['available_candidates']],
                         [c['id'] for c in original['available_candidates']])
        self.assertEqual([c['params'] for c in restored['available_candidates']],
                         [c['params'] for c in original['available_candidates']])

    def test_omitted_and_null_values_stay_distinct(self):
        payload = {'available_candidates': [{'id': 'a', 'params': {'C': None}},
                                            {'id': 'b', 'params': {}},
                                            {'id': 'c', 'params': {'C': 0.3}}],
                   'candidate_facts': {'a': {'classifier': None}, 'b': {'classifier': 'logistic'},
                                       'c': {'classifier': 'logistic'}}}
        self.assertEqual(decode_menu(compact_menu(payload)), payload)

    def test_reserved_at_keys_are_escaped_not_reinterpreted(self):
        payload = {'available_candidates': [{'id': '@', 'params': {'@@': 1}}],
                   'candidate_facts': {'@': {'classifier': 'logistic'}}}
        self.assertEqual(decode_menu(compact_menu(payload)), payload)

    def test_an_unknown_encoding_version_is_rejected(self):
        compact = compact_menu(menu_payload())
        compact['menu_encoding']['version'] = 'something-else'
        with self.assertRaisesRegex(ValueError, 'Unknown menu encoding'):
            decode_menu(compact)
        with self.assertRaisesRegex(ValueError, 'no menu encoding'):
            decode_menu({'available_candidates': []})

    def test_the_compact_view_is_smaller_on_a_wide_menu(self):
        original = menu_payload()
        self.assertLess(recorded_size(compact_menu(original)), recorded_size(original))


class ContextWiringTests(unittest.TestCase):
    def state(self):
        payload = menu_payload()
        return {'task': {'kind': 'pair_text', 'metric': {'name': 'accuracy', 'labels': [0, 1]}},
                'history': [], 'available_candidates': payload['available_candidates'],
                'budget': {'fit_limit': 3, 'attempted_classifier_fits': 0,
                           'completed_classifier_fits': 0, 'remaining_fit_allowance': 3,
                           'reserved_final_refit': 1, 'llm_request_limit': 3, 'used_llm_requests': 0}}

    def capabilities(self):
        return {row['id']: {'estimator': 'logistic', 'params': row['params']}
                for row in menu_payload()['available_candidates']}

    def test_default_context_is_unchanged(self):
        state = self.state()
        ids = [c['id'] for c in state['available_candidates']]
        full = context_for(state, self.capabilities(), ids, True, 'calibrated-v2', 'full')
        explicit = context_for(state, self.capabilities(), ids, True, 'calibrated-v2', 'full',
                               menu_view='full')
        self.assertEqual(full, explicit)
        self.assertNotIn('menu_encoding', full)

    def test_compact_context_round_trips_back_to_the_full_view(self):
        state = self.state()
        ids = [c['id'] for c in state['available_candidates']]
        full = context_for(state, self.capabilities(), ids, True, 'calibrated-v2', 'full')
        compact = context_for(state, self.capabilities(), ids, True, 'calibrated-v2', 'full',
                              menu_view='compact')
        self.assertIn('menu_encoding', compact)
        self.assertEqual(decode_menu(compact), full)

    def test_an_unknown_menu_view_is_rejected(self):
        state = self.state()
        ids = [c['id'] for c in state['available_candidates']]
        with self.assertRaisesRegex(ValueError, 'Unknown menu view'):
            context_for(state, self.capabilities(), ids, True, 'calibrated-v2', 'full',
                        menu_view='abbreviated')


class CliTests(unittest.TestCase):
    def test_menu_view_defaults_to_full(self):
        self.assertEqual(build_parser().parse_args(['run', 'missing.zip']).menu_view, 'full')

    def test_compact_view_is_selectable(self):
        args = build_parser().parse_args(['run', 'missing.zip', '--scheduler', 'ab-v1',
                                          '--menu-view', 'compact'])
        self.assertEqual(args.menu_view, 'compact')


if __name__ == '__main__':
    unittest.main()
