import unittest

from superagent.candidate_coverage import project_routes
from superagent.ab_run import validate_options
from superagent.cli import build_parser, _run


class CoverageTests(unittest.TestCase):
    def fixture(self, kinds=4):
        caps = {f'{route}_{c}':dict(id=f'{route}_{c}',route=f'route{route}')
                for route in range(kinds) for c in range(2)}
        return caps, dict(available_candidates=list(caps.values()),history=[])

    def row(self, name, split='primary', status='completed'):
        return dict(candidate=name,split_id=split,status=status)

    def test_anchor_unmodified_and_primary_routes_precede_more_tuning(self):
        caps,state=self.fixture();ids=tuple(caps)
        self.assertEqual(project_routes(state,caps,ids)[0],ids)
        state['history']=[self.row('0_0')]
        kept,audit=project_routes(state,caps,ids)
        self.assertEqual(kept,tuple(name for name in ids if not name.startswith('0_')))
        self.assertEqual(audit['reason'],'prefer_unseen_feasible_routes')

    def test_full_goal_restores_original_order(self):
        caps,state=self.fixture(5);ids=tuple(reversed(caps))
        state['history']=[self.row(f'{route}_0') for route in range(4)]
        self.assertEqual(project_routes(state,caps,ids)[0],ids)

    def test_confirmation_and_failed_rows_do_not_count_as_route_coverage(self):
        caps,state=self.fixture()
        state['history']=[self.row('0_0'),self.row('1_0','confirmation_1'),self.row('2_0',status='failed')]
        kept,audit=project_routes(state,caps,tuple(caps))
        self.assertEqual(audit['observed_primary_routes'],['route0'])
        self.assertIn('1_0',kept);self.assertIn('2_0',kept)

    def test_respects_feasibility_and_small_route_catalogue(self):
        caps,state=self.fixture(2);state['history']=[self.row('0_0')]
        ids=('0_1',)
        kept,audit=project_routes(state,caps,ids)
        self.assertEqual(kept,ids)
        self.assertEqual(audit['reason'],'no_unseen_route_in_feasible_menu')
        self.assertEqual(audit['target_routes'],2)
        self.assertEqual(project_routes(state,caps,('0_1','1_1'))[0],('1_1',))

    def test_unknown_duplicate_or_expanded_ids_fail_before_model_request(self):
        caps,state=self.fixture()
        for ids in ((),('0_0','0_0'),('unknown',)):
            with self.subTest(ids=ids),self.assertRaises(ValueError):project_routes(state,caps,ids)
        state['available_candidates']=[]
        with self.assertRaises(ValueError):project_routes(state,caps,('0_0',))

    def test_untrusted_route_or_history_is_not_guessed(self):
        caps,state=self.fixture();caps['0_0']['route']=None
        with self.assertRaises(ValueError):project_routes(state,caps,('1_0',))
        caps,state=self.fixture();state['history']=[self.row('unknown')]
        with self.assertRaises(ValueError):project_routes(state,caps,('1_0',))

    def test_cli_is_explicit_and_legacy_local_reject_silent_ignoring(self):
        parser=build_parser()
        self.assertEqual(parser.parse_args(['run','unused.zip']).candidate_coverage,'off')
        local=parser.parse_args(['run','unused.zip','--scheduler','ab-v1','--portfolio','ab-v1','--candidate-coverage','routes'])
        with self.assertRaises(ValueError):validate_options(local)
        legacy=parser.parse_args(['run','unused.zip','--candidate-coverage','routes'])
        with self.assertRaises(ValueError):_run(legacy)
        flash=parser.parse_args(['run','unused.zip','--scheduler','ab-fixed','--portfolio','ab-v1','--planner','raw-history','--candidate-coverage','routes'])
        self.assertEqual(validate_options(flash),11)

    def test_coverage_does_not_modify_state_or_registry(self):
        from copy import deepcopy
        caps,state=self.fixture();state['history']=[self.row('0_0')]
        before=deepcopy((caps,state))
        project_routes(state,caps,tuple(caps))
        self.assertEqual((caps,state),before)


if __name__=='__main__':unittest.main()
