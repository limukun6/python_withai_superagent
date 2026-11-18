import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from sklearn.dummy import DummyClassifier

from superagent.benchmark import run_one
from superagent.benchmark import main as benchmark_main
from superagent.data import DevelopmentTask
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM, RunAllowance
from superagent.hypothesis_agent import build_state, check_expectation, parse_proposal
from superagent.predict import create_predictions


def answer(candidate="b", reference=1, observed=True):
    return {
        "candidate": candidate,
        "reason": "Test a different regularization setting.",
        "hypothesis": "The next score will increase by at least 0.01.",
        "review": "Use prior measurements." if observed else "Outcomes are hidden; the earlier prediction is unverified.",
        "evidence_trials": [reference] if observed and reference is not None else [],
        "expected_change": None if reference is None else {
            "metric": "search_accuracy", "reference_trial": reference,
            "direction": "increase", "min_delta": 0.01,
        },
    }


class ScriptedClient:
    """A schema/transport fixture, not evidence of real LLM decision quality."""
    def __init__(self):
        self.requests = []

    def ask(self, payload, logger, role):
        self.requests.append(copy.deepcopy(payload))
        plans = payload["plans"]
        reference = plans[-1]["trial"] if plans else None
        return answer(payload["available_candidates"][0]["id"], reference, payload["feedback_visible"])


class HypothesisTests(unittest.TestCase):
    def setUp(self):
        self.history = [{"trial": 1, "candidate": "a", "params": {}, "status": "completed",
                         "train_accuracy": .9, "search_accuracy": .6, "generalization_gap": .3,
                         "training_seconds": 4.0, "official_validation_accuracy": .999731,
                         "hypothesis_check": {"status": "not_stated"}}]
        self.available = [{"id": "b", "params": {}}, {"id": "c", "params": {}}]

    def test_feedback_state_retains_results_but_never_final_scores_or_identity(self):
        state = build_state({"task_kind": "categorical_tabular", "task_name": "secret_name"},
                            self.history, self.available, 3)
        self.assertEqual(state["observations"][0]["search_accuracy"], .6)
        self.assertNotIn("official_validation_accuracy", json.dumps(state))
        self.assertNotIn("secret_name", json.dumps(state))

    def test_no_feedback_state_hides_scores_failures_and_checks(self):
        history = copy.deepcopy(self.history)
        history[0]["error"] = "PRIVATE_DIAGNOSTIC"
        history[0]["hypothesis_check"] = {"status": "supported", "observed_value": .999731}
        state = build_state({}, history, self.available, 3, False)
        self.assertEqual(state["observations"], [])
        self.assertEqual(state["plans"][0]["candidate"], "a")
        for hidden in ("PRIVATE_DIAGNOSTIC", "hypothesis_check", "0.999731"):
            self.assertNotIn(hidden, json.dumps(state))
        self.assertNotIn("search_accuracy", state["plans"][0])

    def test_state_is_a_snapshot(self):
        state = build_state({}, self.history, self.available, 3)
        self.history[0]["search_accuracy"] = 0
        self.available[0]["params"]["C"] = 100
        self.assertEqual(state["observations"][0]["search_accuracy"], .6)
        self.assertEqual(state["available_candidates"][0]["params"], {})

    def test_valid_proposal_and_prior_reference(self):
        proposal = parse_proposal(answer(), self.history, self.available)
        self.assertEqual(proposal.candidate, "b")
        self.assertEqual(proposal.expected_change["reference_trial"], 1)

    def test_first_round_can_have_no_reference(self):
        self.assertIsNone(parse_proposal(answer(reference=None), [], self.available).expected_change)

    def test_rejects_unavailable_candidate_future_evidence_and_nonfinite_delta(self):
        variants = []
        for key, value in (("candidate", "a"), ("evidence_trials", [2]), ("evidence_trials", [True])):
            bad = answer()
            bad[key] = value
            variants.append(bad)
        for key, value in (("reference_trial", 2), ("reference_trial", True), ("min_delta", float("nan")),
                           ("min_delta", -1), ("metric", "official_validation_accuracy")):
            bad = answer()
            bad["expected_change"][key] = value
            variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_proposal(bad, self.history, self.available)

    def test_no_feedback_may_predict_relative_change_but_cannot_cite_results(self):
        self.assertIsNotNone(parse_proposal(answer(observed=False), self.history, self.available, False))
        with self.assertRaises(ValueError):
            parse_proposal(answer(), self.history, self.available, False)

    def test_expected_gain_supported_or_not_supported(self):
        proposal = answer()
        row = {"status": "completed", "search_accuracy": .62}
        self.assertEqual(check_expectation(proposal, row, self.history)["status"], "supported")
        row["search_accuracy"] = .59
        self.assertEqual(check_expectation(proposal, row, self.history)["status"], "not_supported")

    def test_expected_gap_decrease(self):
        proposal = answer()
        proposal["expected_change"] = {"metric": "generalization_gap", "reference_trial": 1,
                                       "direction": "decrease", "min_delta": .05}
        checked = check_expectation(proposal, {"status": "completed", "generalization_gap": .2}, self.history)
        self.assertEqual(checked["status"], "supported")

    def test_failure_missing_measurement_and_no_expectation_are_not_successes(self):
        for row in ({"status": "timeout"}, {"status": "completed"},
                    {"status": "completed", "search_accuracy": float("nan")}):
            self.assertEqual(check_expectation(answer(), row, self.history)["status"], "not_evaluable")
        self.assertEqual(check_expectation(None, {}, self.history)["status"], "not_stated")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        x = np.asarray([[str(i % 4), str(i)] for i in range(100)])
        y = np.asarray([i % 2 for i in range(100)])
        self.task = DevelopmentTask("categorical_tabular", "synthetic", x, y, x[:10], y[:10], {})

    def train(self, payload, candidate, destination, timeout):
        model = DummyClassifier().fit(np.zeros((10, 1)), np.tile([0, 1], 5))
        number = int(destination.stem.split("_")[-1])
        score = (.6, .65, .64)[number - 1]
        return model, {"status": "completed", "search_accuracy": score, "train_accuracy": .9,
                       "generalization_gap": .9 - score, "training_seconds": 1.0}

    def test_anchor_and_equal_call_counts_in_feedback_and_blind_arms(self):
        arms = []
        for strategy in ("reflective", "reflective_no_feedback"):
            client = ScriptedClient()
            directory = self.root / strategy
            with patch("superagent.benchmark.train_candidate", side_effect=self.train):
                summary = run_one(self.task, strategy, 11, directory, client, anchor_first=True)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(summary["trials_attempted"], 3)
            self.assertTrue(summary["valid_policy_run"])
            rows = json.loads((directory / "trials.json").read_text())
            self.assertEqual(rows[1]["hypothesis_check"]["status"], "supported")
            self.assertEqual(rows[2]["hypothesis_check"]["status"], "not_supported")
            if strategy == "reflective":
                self.assertEqual(client.requests[1]["observations"][1]["hypothesis_check"]["status"], "supported")
            else:
                self.assertTrue(all(request["observations"] == [] for request in client.requests))
            arms.append(rows)
        self.assertEqual(arms[0][0]["candidate"], arms[1][0]["candidate"])

    def test_bad_schema_is_recorded_as_recovery_not_pure_policy(self):
        client = ScriptedClient()
        client.ask = lambda *args: {"candidate": "does_not_exist"}
        with patch("superagent.benchmark.train_candidate", side_effect=self.train):
            summary = run_one(self.task, "reflective", 11, self.root / "failed_policy", client, anchor_first=True)
        self.assertTrue(summary["completed"])
        self.assertFalse(summary["valid_policy_run"])
        self.assertEqual(summary["fallback_count"], 2)

    def test_oneshot_anchor_counts_toward_the_existing_trial_limit(self):
        requests = []
        class OneShotClient:
            def ask(self, payload, logger, role):
                requests.append(copy.deepcopy(payload))
                return {"plan": [{"candidate": item["id"], "reason": "Common anchor is already completed."}
                                 for item in payload["available_candidates"][:payload["max_trials"]]]}
        with patch("superagent.benchmark.train_candidate", side_effect=self.train):
            summary = run_one(self.task, "oneshot", 11, self.root / "oneshot", OneShotClient(), anchor_first=True)
        self.assertEqual(summary["trials_attempted"], 3)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_trials"], 2)
        self.assertNotIn("completed_experiments", requests[0])

    def test_cli_rejects_an_underfunded_comparison_before_any_api_setup(self):
        argv = ["benchmark", "not-loaded.zip", "--strategies", "reflective", "reflective_no_feedback",
                "--seeds", "11", "--per-run-budget-rmb", ".6", "--batch-budget-rmb", "1"]
        with patch("sys.argv", argv), patch("superagent.benchmark.ExperimentLLM") as client:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                benchmark_main()
        self.assertEqual(result.exception.code, 2)
        client.assert_not_called()

    def test_benchmark_bundle_is_accepted_by_existing_predict_command(self):
        directory = self.root / "export"
        with patch("superagent.benchmark.train_candidate", side_effect=self.train):
            run_one(self.task, "reflective", 11, directory, ScriptedClient(), anchor_first=True)
        import io
        stream = io.BytesIO()
        np.save(stream, np.asarray([["new", "sample"], ["new", "second"]]))
        package = self.root / "prediction.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("C_test.npy", stream.getvalue())
            archive.writestr("y_test.npy", b"Invalid sentinel: never load")
        destination = create_predictions(str(package), directory, self.root / "predictions.csv")
        self.assertEqual(len(destination.read_text().splitlines()), 3)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        key = self.root / "key.txt"
        key.write_text("mock-not-a-real-key")
        self.client = ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)
        self.logger = RunLogger(self.root)
        self.logger.initialize_usage_ledger()

    def test_per_run_exhaustion_does_not_spend_another_runs_allowance(self):
        first = RunAllowance(self.client, .02)
        self.client.spent = .02
        with patch("superagent.experiment_llm.requests.post") as post:
            with self.assertRaises(RuntimeError):
                first.ask({}, self.logger, "planner")
            post.assert_not_called()
        self.assertFalse(self.client.blocked)
        self.assertAlmostEqual(RunAllowance(self.client, .02).cost_ceiling, .04)

    def test_ambiguous_usage_keeps_circuit_open_across_run_wrappers(self):
        with patch("superagent.experiment_llm.requests.post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {}
            with self.assertRaises(RuntimeError):
                RunAllowance(self.client, .2).ask({}, self.logger, "planner")
            with self.assertRaises(RuntimeError):
                RunAllowance(self.client, .2).ask({}, self.logger, "planner")
            self.assertEqual(post.call_count, 1)

    def test_metered_overrun_is_logged_and_blocks_subsequent_calls(self):
        with patch("superagent.experiment_llm.requests.post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1},
                                                    "choices": [{"message": {"content": "{}"}}]}
            with self.assertRaises(RuntimeError):
                self.client.ask({}, self.logger, "planner")
            self.assertTrue(self.client.blocked)
            self.assertIn('"kind": "budget_overrun"', self.logger.events_path.read_text())


if __name__ == "__main__":
    unittest.main()
