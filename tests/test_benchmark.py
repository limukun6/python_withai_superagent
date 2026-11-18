import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import requests
from sklearn.dummy import DummyClassifier

from superagent.benchmark import Policy, inner_split, run_one
from superagent.data import DevelopmentTask
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.models import predict_with_model


class FakeClient:
    def __init__(self):
        self.requests = []

    def ask(self, payload, logger, role):
        self.requests.append((role, payload))
        ids = [c["id"] for c in payload["available_candidates"]]
        if role == "initial_plan":
            return {"plan": [{"candidate": i, "reason": "Test plan"} for i in ids[:payload["max_trials"]]]}
        return {"candidate": ids[-1] if role == "reviewer" else ids[0], "reason": "Test hypothesis"}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.logger = RunLogger(self.root)
        self.logger.initialize_usage_ledger()

    def test_oneshot_calls_once_and_never_receives_feedback(self):
        client = FakeClient()
        policy = Policy("oneshot", 1, client, self.logger, 2)
        candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        self.assertEqual(policy.choose({}, [], candidates)[0], "a")
        self.assertEqual(policy.choose({}, [{"search_accuracy": .9}], candidates[1:])[0], "b")
        self.assertEqual(len(client.requests), 1)
        self.assertNotIn("completed_experiments", client.requests[0][1])

    def test_feedback_and_review_receive_metrics(self):
        for strategy, count in (("feedback", 1), ("reviewed", 2)):
            client = FakeClient()
            history = [{"search_accuracy": .7, "status": "completed"}]
            result = Policy(strategy, 1, client, self.logger, 2).choose({}, history, [{"id": "a"}, {"id": "b"}])
            self.assertEqual(len(client.requests), count)
            self.assertEqual(client.requests[0][1]["completed_experiments"], history)
            self.assertEqual(result[0], "b" if strategy == "reviewed" else "a")

    def test_v2_one_shot_plans_once_after_shared_baselines(self):
        class Client:
            def __init__(self):
                self.calls = []

            def ask(self, payload, logger, role):
                self.calls.append((role, payload))
                return {"plan": [
                    {"candidate": "a", "reason": "first"},
                    {"candidate": "b", "reason": "second"},
                ]}

        client = Client()
        policy = Policy("o", 1, client, self.logger, 2)
        profile = {"task_kind": "pair_text"}
        warm_history = [{"trial": 1, "status": "completed", "search_accuracy": .6}]
        self.assertEqual(policy.choose(profile, warm_history, [{"id": "a"}, {"id": "b"}])[0], "a")
        self.assertEqual(policy.choose(profile, warm_history, [{"id": "b"}])[0], "b")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][0], "v2_one_shot_plan")

    def test_v2_rich_feedback_receives_diagnostics_scalar_does_not(self):
        class Client:
            def __init__(self):
                self.payloads = []

            def ask(self, payload, logger, role):
                self.payloads.append((role, payload))
                return {"candidate": "a", "reason": "test"}

        diagnostics = {"scope": "inner_search_only", "trial_reports": []}
        profile = {"task_kind": "pair_text"}
        history = [{"trial": 1, "status": "completed", "search_accuracy": .6}]
        for strategy, expected_role in (("f", "v2_scalar_feedback"), ("r", "v2_rich_feedback")):
            client = Client()
            Policy(strategy, 1, client, self.logger, 2).choose(profile, history, [{"id": "a"}], diagnostics)
            role, payload = client.payloads[0]
            self.assertEqual(role, expected_role)
            self.assertEqual("diagnostics" in payload, strategy == "r")

    def test_v2_gate_checks_stagnation_against_best_and_requires_two_trials(self):
        policy = Policy("g", 1, None, self.logger, 4)
        policy.plan = []
        history = [{"trial": i, "status": "completed", "search_accuracy": score}
                   for i, score in enumerate((.80, .72, .75), start=1)]
        self.assertIsNone(policy._gated_trigger(history[:2], {}))
        self.assertEqual(policy._gated_trigger(history, {}), "stagnation:trial:2:3")
        history[-1]["search_accuracy"] = .805
        self.assertIsNone(policy._gated_trigger(history, {}))

    def test_v2_gate_requires_same_eligible_slice_in_two_trials(self):
        policy = Policy("g", 1, None, self.logger, 4)
        policy.plan = []
        history = [{"trial": i, "status": "completed", "search_accuracy": score}
                   for i, score in enumerate((.70, .72), start=1)]
        reports = [{"evidence_id": f"diagnostic:trial:{i}", "available": True, "slices": [
            {"slice_id": "short", "available": True, "error_rate": .40}]}
            for i in (1, 2)]
        diagnostics = {"trial_reports": reports}
        self.assertEqual(policy._gated_trigger(history, diagnostics),
                         "persistent_slice_error:short:trial:1:2")
        reports[-1]["slices"][0]["available"] = False
        self.assertIsNone(policy._gated_trigger(history, diagnostics))

    def test_v2_d_uses_shared_warm_start_then_frozen_coverage_order(self):
        x = np.asarray([[str(i), str(i % 3)] for i in range(80)])
        task = DevelopmentTask("categorical_tabular", "", x, np.tile([0, 1], 40), x[:10], np.tile([0, 1], 5), {})
        selected = []

        def fake_train(payload_path, candidate, output, timeout):
            selected.append(candidate)
            model = DummyClassifier().fit(np.zeros((10, 1)), np.tile([0, 1], 5))
            return model, {"status": "completed", "search_accuracy": .5}

        with patch("superagent.runtime.train_candidate", side_effect=fake_train):
            result = run_one(task, "d", 11, self.root / "v2d", None, max_trials=4)
        self.assertTrue(result["valid_policy_run"])
        self.assertEqual(selected, ["logistic_C1.0", "catboost_depth4_reg10.0",
                                    "extra_trees_leaf2", "catboost_depth6_reg3.0"])

    def test_pilot_stops_search_after_unknown_usage_and_keeps_incumbent(self):
        x = np.asarray([[str(i), str(i % 3)] for i in range(80)])
        task = DevelopmentTask("categorical_tabular", "", x, np.tile([0, 1], 40), x[:10], np.tile([0, 1], 5), {})
        client = self._client()
        model = DummyClassifier().fit(np.zeros((10, 1)), np.tile([0, 1], 5))
        with patch.object(client.session, "post", side_effect=requests.ReadTimeout("read timeout")) as post, \
             patch("superagent.runtime.train_candidate", return_value=(model, {
                 "status": "completed", "search_accuracy": .5})) as train:
            result = run_one(task, "r", 101, self.root / "stopped", client,
                             max_trials=6, stop_on_unknown_usage=True)
        self.assertEqual(train.call_count, 2)
        self.assertEqual(post.call_count, 1)
        self.assertTrue(result["acceptance_passed"])
        self.assertFalse(result["valid_policy_run"])
        self.assertFalse(result["usage_complete"])
        self.assertTrue(result["stopped_on_unknown_usage"])

    def test_split_keeps_duplicate_features_together(self):
        x = np.repeat(np.asarray([[str(i), str(i % 3)] for i in range(50)]), 2, axis=0)
        task = DevelopmentTask("categorical_tabular", "", x, np.tile([0, 1], 50), x[:4], np.array([0, 1, 0, 1]), {})
        a, b = inner_split(task, 11)
        self.assertFalse(set(map(tuple, x[a])) & set(map(tuple, x[b])))
        np.testing.assert_array_equal(a, inner_split(task, 11)[0])

    def test_trial_limit_and_holdout_only_after_selection(self):
        x = np.asarray([[str(i), str(i % 3)] for i in range(80)])
        task = DevelopmentTask("categorical_tabular", "", x, np.tile([0, 1], 40), x[:10], np.tile([0, 1], 5), {})
        directory = self.root / "run"
        def fake_train(payload_path, candidate, output, timeout):
            payload = __import__("joblib").load(payload_path)
            self.assertEqual(set(payload), {"kind", "X_train", "y_train", "X_search", "y_search", "search_indices"})
            self.assertFalse((directory / "selection.json").exists())
            model = DummyClassifier().fit(np.zeros((10, 1)), np.tile([0, 1], 5))
            return model, {"status": "completed", "search_accuracy": .5}
        def score_final(kind, model, inputs):
            self.assertTrue((directory / "selection.json").exists())
            return np.zeros(len(inputs), dtype=int)
        with patch("superagent.runtime.train_candidate", side_effect=fake_train) as train:
            with patch("superagent.runtime.predict_with_model", side_effect=score_final) as score:
                result = run_one(task, "random", 11, directory, None, max_trials=2)
        self.assertEqual(train.call_count, 2)
        self.assertEqual(score.call_count, 2)  # inner acceptance probe, then official validation
        self.assertTrue(result["valid_policy_run"])

    def test_text_dummy_reload_prediction(self):
        model = DummyClassifier().fit(np.zeros((3, 1)), [0, 0, 1])
        self.assertEqual(predict_with_model("pair_text", model, [("a", "b")]).tolist(), [0])

    def test_missing_api_usage_blocks_more_calls(self):
        key = self.root / "key.txt"
        key.write_text("test-not-a-real-key", encoding="utf-8")
        client = ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)
        with patch.object(client.session, "post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
            self.assertEqual(post.call_count, 1)
        usage = [json.loads(line) for line in self.logger.usage_path.read_text().splitlines()]
        self.assertIsNone(usage[-1]["cost_rmb"])

    def test_llm_ignores_stale_proxy_environment(self):
        key = self.root / "key.txt"
        key.write_text("test-not-a-real-key", encoding="utf-8")
        client = ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)
        self.assertFalse(client.session.trust_env)

    def test_llm_reserves_room_for_reasoning_and_final_json(self):
        key = self.root / "key.txt"
        key.write_text("test-not-a-real-key", encoding="utf-8")
        client = ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)
        with patch.object(client.session, "post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            self.assertEqual(client.ask({}, self.logger, "planner"), {})
        self.assertEqual(
            post.call_args.kwargs["json"]["max_tokens"], ExperimentLLM.max_output_tokens
        )
        # 4096 was still truncated on the longest planner call of a run
        # (finish_reason=length, content=null), so the reservation must stay
        # strictly above the value that failed.
        self.assertGreater(ExperimentLLM.max_output_tokens, 4096)

    def test_flash_uses_current_api_name_and_sixty_second_timeout(self):
        key = self.root / "key.txt"
        key.write_text("test-not-a-real-key", encoding="utf-8")
        for name in ("DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash-1017"):
            client = ExperimentLLM("https://example.invalid/api", str(key), name, 1)
            with patch.object(client.session, "post", return_value=self._good_response()) as post:
                client.ask({}, self.logger, "planner")
            self.assertEqual(post.call_args.kwargs["json"]["model"], "DeepSeek-V4.1-Flash")
            self.assertEqual(post.call_args.kwargs["timeout"], 60)
            self.assertFalse(client.session.trust_env)


    # ---- unknown usage must not trigger a second paid call -----------------

    def _client(self):
        key = self.root / "retry_key.txt"
        key.write_text("test-not-a-real-key", encoding="utf-8")
        return ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)

    @staticmethod
    def _good_response():
        response = MagicMock(ok=True)
        response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
        return response

    def test_transport_failure_stops_without_automatic_paid_retry(self):
        client = self._client()
        with patch.object(
            client.session, "post", side_effect=[requests.ConnectionError("reset"), self._good_response()]
        ) as post:
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
        self.assertEqual(post.call_count, 1)
        self.assertTrue(client.blocked)
        events = [json.loads(line) for line in (self.root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["kind"] == "llm_transport_failure" for e in events))
        usage = [json.loads(line) for line in (self.root / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()]
        # The failed attempt stays visible as unknown usage: recovery must not erase the gap.
        self.assertTrue(any(u["event"] == "unresolved_call" and u["usage_known"] is False for u in usage))

    def test_transport_failure_after_retries_opens_circuit(self):
        client = self._client()
        with patch.object(client.session, "post", side_effect=requests.ConnectionError("reset")) as post, \
                patch("superagent.experiment_llm.time.sleep"):
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
        self.assertEqual(post.call_count, ExperimentLLM.max_transport_retries + 1)
        self.assertTrue(client.blocked)
        with self.assertRaises(RuntimeError):
            client.ask({}, self.logger, "planner")
        self.assertEqual(post.call_count, ExperimentLLM.max_transport_retries + 1)

    def test_http_error_is_not_retried(self):
        client = self._client()
        bad = MagicMock(ok=False, status_code=400, text="bad request")
        bad.raise_for_status.side_effect = requests.HTTPError("400 Client Error", response=bad)
        with patch.object(client.session, "post", return_value=bad) as post, \
                patch("superagent.experiment_llm.time.sleep"):
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
        self.assertEqual(post.call_count, 1)
        self.assertTrue(client.blocked)


if __name__ == "__main__":
    unittest.main()
