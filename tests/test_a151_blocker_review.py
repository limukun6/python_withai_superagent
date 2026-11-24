import unittest
from scripts.review_a151_blocker2 import compact_menu, expand_menu, encode_tree, decode_tree, transport_review
from superagent.experiment_llm import read_chat_stream


class OfflineBlockerTests(unittest.TestCase):
    def test_missing_null_and_nested_values_roundtrip(self):
        value = [{'id': 'a', 'params': {'C': 0.5, 'x': None}},
                 {'id': 'b', 'params': {}, 'extra': [True, '中文', None]}, {}]
        self.assertEqual(decode_tree(encode_tree(value)), value)

    def test_encoding_does_not_change_other_context_or_input(self):
        value = dict(available_candidates=[{'id': 'a', 'params': {'C': 1}}],
                     candidate_facts={'a': {'classifier': 'logistic'}},
                     history=[{'score': 0.3}], checks=[{'unknown': True}],
                     memory={'failure': 'preserve'}, instruction='Keep grounded assertions')
        encoded = compact_menu(value)
        self.assertEqual(expand_menu(encoded), value)
        for key in ('history', 'checks', 'memory', 'instruction'):
            self.assertEqual(encoded[key], value[key])
        self.assertNotIn('menu_encoding', value)

    def test_current_requests_defaults_do_not_retry_post(self):
        result = transport_review()
        self.assertEqual(result['total_retries'], 0)
        self.assertEqual(set(result['error_checks'].values()), {'raises_without_retry'})

    def test_shared_values_preserve_nested_and_reserved_dictionary_keys(self):
        params = {'name': 'repeated ' * 20, 'values': [0.1, None, True], '@': 3}
        payload = {'available_candidates': [{'id': 'a', 'params': params},
                                           {'id': 'b', 'params': params}],
                   'candidate_facts': {'a': {'@@': [['unexpected', 1]]}}}
        self.assertEqual(expand_menu(compact_menu(payload)), payload)

    def test_existing_empty_line_counter_cannot_prove_no_bytes(self):
        class Response:
            def iter_lines(self): return iter([b'', b'', b''])
        progress = {'data_events': 0}
        read_chat_stream(Response(), None, 0, progress)
        # Documents the present observability limitation, not the desired future counter.
        self.assertEqual(progress, {'data_events': 0})


if __name__ == '__main__': unittest.main()
