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
from superagent.diagnostics import summarize, claim_facts
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.runtime import Policy
from superagent.runtime import confirmation_split
from superagent.selection import confirm_once
from superagent.models import search_candidates
from superagent.replay_confirmation import replay


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

    def test_aggregate_diagnostics_expose_exact_slice_facts_without_raw_text(self):
        fit = [("short", "x"), ("one two three four", "x")] * 40
        search = [("tiny", "x")] * 50 + [("one two three four five", "x")] * 50
        labels = np.asarray([0] * 50 + [1] * 50)
        joblib.dump({"search_predictions": np.asarray([0] * 50 + [0] * 50)}, self.root / "trial_1.joblib")
        report = summarize("pair_text", fit, search, labels,
                           [{"trial": 1, "candidate": "word_C0.3", "status": "completed"}], self.root)
        self.assertEqual(report["scope"], "inner_search_only")
        self.assertNotIn("tiny", json.dumps(report))
        facts = claim_facts(report)
        short = next(f for f in facts if f["evidence_id"].endswith("short_or_equal"))
        self.assertEqual(short["field"], "accuracy")
        self.assertEqual(short["value"], 1.0)

    def test_diagnostic_fact_must_match_exactly(self):
        history = [{"trial": 1, "candidate": "a", "status": "completed", "search_accuracy": .69}]
        facts = [{"evidence_id": "diagnostic:trial:1:short", "field": "error_rate", "value": .3}]
        decision = {"candidate": "b", "reason": "Target short-text errors", "observations": [
            {"evidence_id": "diagnostic:trial:1:short", "field": "error_rate", "value": .3}],
            "hypothesis": "Character features may improve short texts", "alternative_explanation": "Noise",
            "changed_factor": "representation", "parent_candidate": "a",
            "prediction": {"metric": "search_accuracy", "direction": "increase"}}
        validate_scientific_decision(decision, history, facts)
        decision["observations"][0]["value"] = .2
        with self.assertRaises(ValueError):
            validate_scientific_decision(decision, history, facts)

    def test_scientist_can_use_exact_diagnostic_fact(self):
        class Client:
            def ask(self, payload, logger, role):
                self.payload = payload
                return {"candidate": "b", "reason": "Test character features", "observations": [
                    {"evidence_id": "diagnostic:trial:1:short", "field": "error_rate", "value": .3}],
                    "hypothesis": "Character features may help", "alternative_explanation": "Split noise",
                    "changed_factor": "representation", "parent_candidate": "a",
                    "prediction": {"metric": "search_accuracy", "direction": "increase"}}
        client = Client()
        logger = RunLogger(self.root)
        logger.initialize_usage_ledger()
        policy = Policy("scientist", 1, client, logger, 2)
        history = [{"trial": 1, "candidate": "a", "status": "completed", "search_accuracy": .69}]
        diagnostics = {"trial_reports": [], "pairwise_predictions": []}
        with patch("superagent.runtime.diagnostic_claim_facts", return_value=[
            {"evidence_id": "diagnostic:trial:1:short", "field": "error_rate", "value": .3}]):
            chosen, _ = policy.choose({}, history, [{"id": "b"}], diagnostics=diagnostics)
        self.assertEqual(chosen, "b")
        self.assertEqual(client.payload["tool_facts"][-1]["value"], .3)

    def test_confirmation_split_keeps_duplicates_in_one_partition(self):
        x = np.repeat(np.asarray([[str(i), str(i % 4)] for i in range(50)]), 2, axis=0)
        task = DevelopmentTask("categorical_tabular", "", x, np.tile([0, 1], 50), x[:4], np.array([0, 1, 0, 1]), {})
        fit, search, confirm = confirmation_split(task, 11)
        self.assertFalse(set(fit) & set(search) or set(fit) & set(confirm) or set(search) & set(confirm))
        partitions = [set(map(tuple, x[idx])) for idx in (fit, search, confirm)]
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])

    def test_confirmation_can_switch_only_between_frozen_top_two(self):
        x = np.asarray([["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]])
        y = np.asarray([1, 1, 1, 1])
        zero = DummyClassifier(strategy="constant", constant=0).fit(np.zeros((2, 1)), [0, 1])
        one = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, 1)), [0, 1])
        joblib.dump({"model": zero}, self.root / "trial_1.joblib")
        joblib.dump({"model": one}, self.root / "trial_2.joblib")
        history = [
            {"trial": 1, "candidate": "logistic_C0.05", "status": "completed", "search_accuracy": .80},
            {"trial": 2, "candidate": "logistic_C0.25", "status": "completed", "search_accuracy": .79},
            {"trial": 3, "candidate": "logistic_C1.0", "status": "completed", "search_accuracy": .10},
        ]
        selected, _, report = confirm_once("categorical_tabular", history, self.root, x, y, True)
        self.assertEqual(selected["candidate"], "logistic_C0.25")
        self.assertTrue(report["switched"])
        self.assertEqual([x["candidate"] for x in report["shortlist"]], ["logistic_C0.05", "logistic_C0.25"])
        selected, _, report = confirm_once("categorical_tabular", history, self.root, None, None, False)
        self.assertEqual(selected["candidate"], "logistic_C0.05")
        self.assertFalse(report["switched"])

    def test_confirmation_rows_are_never_written_to_worker_payload(self):
        seen_payloads = []

        def fake_train(payload_path, candidate, destination, timeout):
            payload = joblib.load(payload_path)
            seen_payloads.append(payload)
            model = DummyClassifier(strategy="most_frequent").fit(np.zeros((4, 1)), [0, 0, 1, 1])
            joblib.dump({"model": model}, destination)
            return model, {"status": "completed", "search_accuracy": .50}

        with patch("superagent.runtime.train_candidate", side_effect=fake_train):
            result = runtime.run_task(toy_task(), "deterministic", 11, self.root / "run",
                                      None, max_trials=2, confirmation=True)
        self.assertTrue(result["completed"])
        self.assertEqual(len(seen_payloads), 2)
        self.assertTrue(all("confirmation" not in key for p in seen_payloads for key in p))
        self.assertTrue(all(len(p["X_train"]) + len(p["X_search"]) < 80 for p in seen_payloads))

    def test_reserved_confirmation_replays_frozen_models_and_detects_tampering(self):
        task = toy_task()
        task.y_val[:] = 1
        fit, search, sealed = np.arange(72), np.arange(72, 80, 2), np.arange(73, 80, 2)
        models = [DummyClassifier(strategy="constant", constant=label).fit(np.zeros((2, 1)), [0, 1])
                  for label in (0, 1)]
        run = self.root / "reserved"

        def fake_train(_, candidate, destination, timeout):
            trial = int(destination.stem.split("_")[1])
            model = models[trial - 1]
            joblib.dump({"model": model}, destination)
            return model, {"status": "completed", "search_accuracy": .9 - trial / 10}

        with patch("superagent.runtime.confirmation_split", return_value=(fit, search, sealed)), \
             patch("superagent.runtime.train_candidate", side_effect=fake_train):
            summary = runtime.run_task(task, "deterministic", 11, run, None, max_trials=2,
                                       reserve_confirmation=True)
        source_hash = json.loads((run / "runtime_manifest.json").read_text())["source_sha256"]
        self.assertEqual(summary["official_validation_accuracy"], 0.0)
        result = replay(task, run, summary, source_hash)
        self.assertTrue(result["switched"])
        self.assertEqual(result["c_on_validation"], 1.0)
        with (run / "trial_2.joblib").open("ab") as file:
            file.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen candidate artifact changed"):
            replay(task, run, summary, source_hash)

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
