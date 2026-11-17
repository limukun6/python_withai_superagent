import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from superagent import benchmark, cli, runtime
from superagent.acceptance import accept_artifact, digest, load_accepted_model, check_predictions
from superagent.contracts import build_contract, describe_candidate
from superagent.data import DevelopmentTask
from superagent.decisions import validate_scientific_decision, hypothesis_outcome
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.models import search_candidates


PAIR_INSTRUCTIONS = "The target `label_pairwise` is `1` if A had the higher rate and `0` if B did."


def toy_task(kind="categorical_tabular", instructions=""):
    x = np.asarray([[str(i), str(i % 3)] for i in range(80)])
    if kind == "pair_text":
        x = [(f"left {i}", f"right {i}") for i in range(80)]
    return DevelopmentTask(kind, "test", x, np.tile([0, 1], 40), x[:10], np.tile([0, 1], 5),
                           {"instructions": instructions, "instruction_source": "fixture/README.md"})


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cli_and_benchmark_use_same_runtime(self):
        self.assertIs(cli.run_task, runtime.run_task)
        self.assertIs(benchmark.run_one, runtime.run_task)

    def test_pair_semantics_must_be_explicit_not_inferred_from_columns(self):
        for text in ("", "Predict whether two headlines are semantically similar."):
            with self.assertRaisesRegex(ValueError, "semantics are unverified"):
                build_contract(toy_task("pair_text", text))
        contract = build_contract(toy_task("pair_text", PAIR_INSTRUCTIONS))
        self.assertTrue(contract.swap_inverts_label)
        self.assertEqual(contract.pair_relation, "ordered_comparison")

    def test_multiclass_not_mislabeled_supported(self):
        task = toy_task()
        task.y_train[0] = 2
        with self.assertRaisesRegex(ValueError, "multiclass"):
            build_contract(task)

    def test_registry_describes_real_estimator_and_representation(self):
        specs = [describe_candidate("pair_text", c) for c in search_candidates("pair_text")]
        self.assertTrue(all(s["estimator"] == "LogisticRegression" for s in specs))
        self.assertTrue(all("ngrams(1,2)" in s["representation"] for s in specs))
        self.assertTrue(any("char_wb" in s["representation"] for s in specs))

    def decision(self):
        history = [{"trial": 1, "candidate": "a", "status": "completed", "search_accuracy": .69,
                    "train_accuracy": .98}]
        decision = {"candidate": "b", "reason": "Test stronger regularization",
                    "observations": [{"evidence_id": "trial:1", "field": "search_accuracy", "value": .69}],
                    "hypothesis": "Regularization may help", "alternative_explanation": "Representation is insufficient",
                    "changed_factor": "C", "parent_candidate": "a",
                    "prediction": {"metric": "search_accuracy", "direction": "increase"}}
        return decision, history

    def test_fabricated_fact_or_evidence_reference_is_rejected(self):
        decision, history = self.decision()
        self.assertEqual(validate_scientific_decision(decision, history), decision)
        for field, value in (("value", .99), ("evidence_id", "trial:999"), ("value", True)):
            bad = copy.deepcopy(decision)
            bad["observations"][0][field] = value
            with self.assertRaises(ValueError):
                validate_scientific_decision(bad, history)

    def test_smaller_gap_does_not_count_as_accuracy_improvement(self):
        decision, history = self.decision()
        result = {"status": "completed", "search_accuracy": .68, "train_accuracy": .80}
        outcome = hypothesis_outcome(decision, result, history)
        self.assertEqual(outcome["state"], "not_supported")
        self.assertFalse(outcome["causal_claim"])
        decision["prediction"] = {"metric": "generalization_gap", "direction": "decrease"}
        with self.assertRaises(ValueError):
            validate_scientific_decision(decision, history)

    def test_execution_failure_does_not_disprove_hypothesis(self):
        decision, history = self.decision()
        outcome = hypothesis_outcome(decision, {"status": "timeout"}, history)
        self.assertEqual(outcome["state"], "execution_failed")

    def export_fixture(self):
        contract = build_contract(toy_task()).to_dict()
        (self.root / "task_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        model = DummyClassifier(strategy="most_frequent").fit(np.zeros((3, 2)), [0, 0, 1])
        joblib.dump({"kind": "categorical_tabular", "candidate": "fixture", "model": model},
                    self.root / "selected_model.joblib")
        manifest = {"schema_version": 1, "kind": "categorical_tabular", "candidate": "fixture", "labels": [0, 1],
                    "model_file": "selected_model.joblib", "model_sha256": digest(self.root / "selected_model.joblib"),
                    "contract_sha256": digest(self.root / "task_contract.json")}
        (self.root / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return np.zeros((3, 2))

    def test_independent_reload_passes_then_model_tamper_is_rejected(self):
        features = self.export_fixture()
        report = accept_artifact(self.root, features, np.zeros(3, dtype=int))
        self.assertTrue(report["passed"], report)
        self.assertEqual(load_accepted_model(self.root)[0]["candidate"], "fixture")
        with (self.root / "selected_model.joblib").open("ab") as f:
            f.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_accepted_model(self.root)

    def test_bad_expected_predictions_fail_independent_acceptance(self):
        features = self.export_fixture()
        report = accept_artifact(self.root, features, np.ones(3, dtype=int))
        self.assertFalse(report["passed"])
        with self.assertRaises(ValueError):
            load_accepted_model(self.root)

    def test_unknown_labels_and_bad_row_count_rejected(self):
        for pred in ([0, 2], [0], [[0], [1]]):
            with self.assertRaises(ValueError):
                check_predictions(pred, 2, [0, 1])

    def test_failed_acceptance_blocks_official_validation(self):
        model = DummyClassifier().fit(np.zeros((3, 2)), [0, 0, 1])
        with patch("superagent.runtime.train_candidate", return_value=(model, {"status": "completed", "search_accuracy": .5})), \
             patch("superagent.runtime.accept_artifact", return_value={"passed": False, "error": "test failure"}), \
             patch("superagent.runtime.accuracy_score") as scorer:
            result = runtime.run_task(toy_task(), "deterministic", 11, self.root / "run", None, max_trials=1)
        self.assertFalse(result["completed"])
        self.assertFalse(result["valid_policy_run"])
        scorer.assert_not_called()

    def test_scientist_invalid_claim_degrades_and_is_not_a_valid_policy_run(self):
        class BadClient:
            def ask(self, payload, logger, role):
                return {"candidate": payload["available_candidates"][0]["id"], "reason": "Trust me"}
        model = DummyClassifier().fit(np.zeros((3, 2)), [0, 0, 1])
        with patch("superagent.runtime.train_candidate", return_value=(model, {"status": "completed", "search_accuracy": .5})):
            result = runtime.run_task(toy_task(), "scientist", 11, self.root / "run", BadClient(), max_trials=1)
        self.assertTrue(result["completed"])
        self.assertFalse(result["valid_policy_run"])
        self.assertEqual(result["fallback_count"], 1)

    def test_api_deadline_blocks_request_before_network(self):
        key = self.root / "key.txt"
        key.write_text("fake-key", encoding="utf-8")
        client = ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)
        client.deadline = time.monotonic() - 1
        logger = RunLogger(self.root)
        logger.initialize_usage_ledger()
        with patch.object(client.session, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                client.ask({}, logger, "planner")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
