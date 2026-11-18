import copy
import json
from pathlib import Path
import unittest

from superagent.evidence_agent import EvidenceAgent, latest_result, parse_evidence, state_for


FIXTURES = json.loads((Path(__file__).parent / "fixtures/v03_failed_readings.json").read_text())


class ExecutorReadingTests(unittest.TestCase):
    def test_real_failures_replay_without_rewriting_their_original_records(self):
        for fixture in FIXTURES:
            with self.subTest(run=fixture["original_run"]):
                original = copy.deepcopy(fixture["original_answer"])
                with self.assertRaisesRegex(ValueError, "Reading contradicts"):
                    parse_evidence(original, fixture["history"], fixture["available"], "evidence_routes")
                # This is an explicit offline schema migration, not a new LLM call.
                adapted = copy.deepcopy(original)
                adapted["assessment"].pop("reading")
                result = parse_evidence(adapted, fixture["history"], fixture["available"], "budget_routes")
                self.assertEqual(result["candidate"], original["candidate"])
                self.assertEqual(result["assessment"]["reading"], latest_result(fixture["history"])["reading"])
                self.assertEqual(result["assessment"]["reading_source"], "executor")
                self.assertEqual(fixture["original_answer"], original)

    def test_original_redundant_llm_label_is_not_silently_corrected(self):
        fixture = FIXTURES[0]
        with self.assertRaisesRegex(ValueError, "Incorrect assessment fields"):
            parse_evidence(fixture["original_answer"], fixture["history"], fixture["available"], "budget_routes")

    def test_reference_validation_is_not_weakened(self):
        fixture = FIXTURES[0]
        adapted = copy.deepcopy(fixture["original_answer"])
        adapted["assessment"].pop("reading")
        adapted["assessment"]["fact_ids"] = ["t999.search_accuracy"]
        with self.assertRaisesRegex(ValueError, "fabricated fact"):
            parse_evidence(adapted, fixture["history"], fixture["available"], "budget_routes")

    def test_latest_is_compared_to_best_earlier_not_immediate_predecessor(self):
        history = [{"trial": i, "status": "completed", "search_accuracy": score}
                   for i,score in enumerate((.8,.6,.7), 1)]
        result = latest_result(history)
        self.assertEqual(result["reading"], "lower")
        self.assertEqual(result["best_prior_trial"], 1)
        self.assertAlmostEqual(result["delta_vs_prior_best"], -.1)

    def test_first_missing_and_failed_outcomes_are_not_made_into_numeric_gains(self):
        self.assertEqual(latest_result([])["reading"], "no_comparison")
        for status, expected in (("timeout", "execution_failure"), ("completed", "not_evaluable")):
            result = latest_result([{"trial": 1, "status": status, "search_accuracy": None}])
            self.assertEqual(result["reading"], expected)
            self.assertIsNone(result["latest_search_accuracy"])
            self.assertIsNone(result["delta_vs_prior_best"])

    def test_planner_receives_executor_comparison_with_new_schema(self):
        fixture = FIXTURES[0]
        adapted = copy.deepcopy(fixture["original_answer"])
        adapted["assessment"].pop("reading")
        class LocalClient:
            def ask(self, payload, logger, role):
                self.payload = payload
                return adapted
        client = LocalClient()
        agent = EvidenceAgent(client, None, 8, "budget_routes")
        candidate, reason = agent.choose({}, fixture["history"], fixture["available"])
        self.assertEqual(client.payload["latest_result"]["reading"], "lower")
        self.assertIn("No reading field", client.payload["instruction"])
        self.assertEqual(candidate, adapted["candidate"])
        self.assertNotIn("latest_result", state_for({}, fixture["history"], fixture["available"], 8, "evidence_routes"))


if __name__ == "__main__":
    unittest.main()
