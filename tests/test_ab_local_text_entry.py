import unittest
from superagent.ab_run import validate_options
from superagent.cli import build_parser


class LocalTextEntryTests(unittest.TestCase):
    def args(self,*extra):
        return build_parser().parse_args(['run','anonymous.zip','--scheduler','ab-v1','--portfolio','ab-v1',*extra])

    def test_explicit_mode_keeps_same_fit_limit_and_default_off(self):
        self.assertEqual(self.args().local_text_proposals,'off')
        args=self.args('--local-text-proposals','evidence')
        self.assertEqual(validate_options(args),11)

    def test_incompatible_modes_refuse_before_data_or_network(self):
        for extra in (['--planner','raw-history'],['--planner','checked-history'],
                      ['--scheduler','ab-fixed'],['--scheduler','ab-finalists'],['--catalog-scope','baseline']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError,'Evidence text proposals require'):
                validate_options(self.args('--local-text-proposals','evidence',*extra))

    def test_non_ab_entry_refuses_ignored_option(self):
        args=build_parser().parse_args(['run','missing.zip','--local-text-proposals','evidence'])
        with self.assertRaisesRegex(ValueError,'AB fit-limit/portfolio'):
            args.handler(args)
