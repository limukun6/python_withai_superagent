import unittest
from pathlib import Path
from unittest.mock import patch

from tools.ab_registered_run import run
from superagent import ab_run
from superagent.cli import build_parser


class RegisteredOptionsTests(unittest.TestCase):
    def test_new_partition_option_is_forwarded_without_training(self):
        task=object();directory=Path('anonymous_run');metric=object();selector=object()
        observed=[]
        def freeze(current,where,seed,scoring,**options):
            self.assertIs(current,task);self.assertIs(scoring,metric)
            self.assertEqual(options,{'preference_vote':True});observed.append(options)
            return 'registered'
        def execute(args):
            return ab_run.freeze_partitions(task,directory,707,metric,preference_vote=True)
        with patch.object(ab_run,'freeze_partitions',side_effect=freeze), \
             patch('tools.ab_registered_run.registered_partitioner',return_value=selector), \
             patch('superagent.cli._run',side_effect=execute):
            for scheduler in ('ab-v1','ab-adaptive-complement'):
                self.assertEqual(run(Path('rows.json'),'digest',['run','anonymous.zip','--scheduler',scheduler,'--portfolio','ab-v1']),'registered')
        self.assertEqual(len(observed),2)

    def test_new_options_are_explicit_and_reject_other_schedulers(self):
        from superagent.cli import _run
        for flag in ('--cover-untried-encoder','--resolve-single-conflict'):
            args=build_parser().parse_args(['run','anonymous.zip',flag])
            with self.assertRaisesRegex(ValueError,'[Aa]daptive'):_run(args)


if __name__=='__main__':unittest.main()
