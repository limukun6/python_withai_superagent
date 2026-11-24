import unittest

from superagent.ab_run import validate_options
from superagent.cli import build_parser

BASE = ['run', 'task.zip', '--planner', 'checked-history', '--scheduler', 'ab-v1',
        '--portfolio', 'ab-v1', '--catalog-scope', 'full', '--preference-catalog', 'standard',
        '--parameter-semantics', 'off', '--history-policy', 'incremental-research',
        '--task-budget-id', 'a151_cap_test']


def parsed(*extra):
    return build_parser().parse_args([*BASE, *extra])


class RequestLimitTests(unittest.TestCase):
    def test_the_default_is_three(self):
        self.assertEqual(parsed().llm_request_limit, 3)

    def test_a_larger_cap_is_accepted(self):
        validate_options(parsed('--llm-request-limit', '6'))

    def test_zero_and_negative_are_rejected(self):
        for value in ('0', '-1'):
            with self.assertRaises(ValueError):
                validate_options(parsed('--llm-request-limit', value))

    def test_more_than_twelve_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_options(parsed('--llm-request-limit', '13'))

    def test_the_default_command_still_validates(self):
        validate_options(parsed())


if __name__ == '__main__':
    unittest.main()
