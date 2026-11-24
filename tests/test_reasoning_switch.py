import io
import json
import unittest

from superagent.ab_flash import context_for
from superagent.cli import build_parser
from superagent.experiment_llm import read_chat_stream


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self):
        for line in self.lines:
            yield line


class StreamAccountingTests(unittest.TestCase):
    def test_keepalive_lines_are_counted_so_a_silent_failure_is_diagnosable(self):
        progress = {'data_events': 0}
        lines = [b': keepalive', b'', b': ping', b'data: {"choices": [{"delta": {"content": "ok"}}]}',
                 b'data: [DONE]']
        body = read_chat_stream(FakeResponse(lines), None, 0.0, progress)
        self.assertEqual(body['choices'][0]['message']['content'], 'ok')
        self.assertEqual(progress['data_events'], 1)
        self.assertEqual(progress['other_events'], 2)
        self.assertIn('first_other_seconds', progress)

    def test_a_stream_with_no_data_lines_is_distinguishable_from_no_bytes(self):
        silent = {'data_events': 0}
        read_chat_stream(FakeResponse([]), None, 0.0, silent)
        self.assertNotIn('other_events', silent)
        keepalive_only = {'data_events': 0}
        read_chat_stream(FakeResponse([b': keepalive'] * 3), None, 0.0, keepalive_only)
        self.assertEqual(keepalive_only['data_events'], 0)
        self.assertEqual(keepalive_only['other_events'], 3)


class ReasoningFlagTests(unittest.TestCase):
    def test_cli_accepts_an_explicit_thinking_and_reasoning_switch(self):
        args = build_parser().parse_args(['run', 'missing.zip', '--scheduler', 'ab-v1',
                                          '--thinking', 'disabled', '--reasoning-effort', 'none'])
        self.assertEqual(args.thinking_mode, 'disabled')
        self.assertEqual(args.reasoning_effort_level, 'none')

    def test_flags_default_to_none_so_old_requests_are_unchanged(self):
        args = build_parser().parse_args(['run', 'missing.zip', '--scheduler', 'ab-v1'])
        self.assertIsNone(args.thinking_mode)
        self.assertIsNone(args.reasoning_effort_level)


if __name__ == '__main__':
    unittest.main()
