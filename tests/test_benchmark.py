import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
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
            self.assertEqual(set(payload), {"kind", "X_train", "y_train", "X_search", "y_search",'metric_spec'})
            self.assertEqual(payload['metric_spec'],{'name':'accuracy','labels':[0,1],'direction':'maximize'})
            self.assertFalse((directory / "selection.json").exists())
            model = DummyClassifier().fit(np.zeros((10, 1)), np.tile([0, 1], 5))
            return model, {"status": "completed", "search_accuracy": .5}
        def score_final(kind, model, inputs):
            self.assertTrue((directory / "selection.json").exists())
            self.assertTrue((directory / "final_selection.json").exists())
            return np.zeros(len(inputs), dtype=int)
        with patch("superagent.benchmark.train_candidate", side_effect=fake_train) as train:
            with patch("superagent.finalization.predict_with_model", side_effect=score_final) as score:
                result = run_one(task, "random", 11, directory, None, max_trials=2)
        self.assertEqual(train.call_count, 2)
        self.assertEqual(score.call_count, 1)
        self.assertTrue(result["valid_policy_run"])

    def test_text_dummy_reload_prediction(self):
        model = DummyClassifier().fit(np.zeros((3, 1)), [0, 0, 1])
        self.assertEqual(predict_with_model("pair_text", model, [("a", "b")]).tolist(), [0])

    def test_missing_api_usage_blocks_more_calls(self):
        key = self.root / "key.txt"
        key.write_text("test-not-a-real-key", encoding="utf-8")
        client = ExperimentLLM("https://example.invalid/api", str(key), "DeepSeek-V4-Pro", 1)
        with patch("superagent.experiment_llm.requests.post") as post:
            post.return_value.ok = True
            post.return_value.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
            with self.assertRaises(RuntimeError):
                client.ask({}, self.logger, "planner")
            self.assertEqual(post.call_count, 1)
        usage = [json.loads(line) for line in self.logger.usage_path.read_text().splitlines()]
        self.assertIsNone(usage[-1]["cost_rmb"])


if __name__ == "__main__":
    unittest.main()
