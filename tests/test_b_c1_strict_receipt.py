import unittest

from scripts.run_b_c1_matrix import verify_strict_history


class FormalBudgetReceiptTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'history_policy':'strict'}
        self.project=dict(unresolved_requests=0,pending_requests=0,complete_course_cost_rmb=0.023)
        self.paid=dict(history_policy='strict',course_budget_compliance='verified_local_ledger',
                       valid_for_budgeted_final_report=None, project_usage_after_run=self.project,
                       complete_project_course_cost_rmb=0.023)

    def test_distinguishes_local_strict_and_paid_strict(self):
        local=dict(history_policy='strict',course_budget_compliance='no_new_llm_calls')
        verify_strict_history(self.manifest,local,'local')
        verify_strict_history(self.manifest,self.paid,'raw-history')
        with self.assertRaisesRegex(ValueError,'history policy'):
            verify_strict_history(self.manifest,local,'checked-history')

    def test_rejects_incremental_research_and_inconsistent_or_incomplete_project_totals(self):
        for changes in ({'history_policy':'incremental-research'}, {'course_budget_compliance':'unverified_history'},
                        {'valid_for_budgeted_final_report':False}, {'complete_project_course_cost_rmb':0.0},
                        {'project_usage_after_run':None}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                verify_strict_history(self.manifest,dict(self.paid,**changes),'checked-history')
        for changes in ({'unresolved_requests':1},{'pending_requests':1},{'complete_course_cost_rmb':None},
                        {'complete_course_cost_rmb':float('nan')}):
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'project budget receipt'):
                verify_strict_history(self.manifest,dict(self.paid,project_usage_after_run=dict(self.project,**changes)),
                                      'raw-history')
