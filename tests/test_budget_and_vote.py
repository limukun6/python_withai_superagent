import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import joblib
import numpy as np

from superagent.data import DevelopmentTask
from superagent.evidence_agent import state_for
from superagent.events import RunLogger
from superagent.finalization import finalize, propose_vote
from superagent.models import SoftVoteModel, predict_with_model


class FixedProbabilities:
    def __init__(self, p0, reversed_classes=False):
        self.classes_ = np.asarray([1, 0] if reversed_classes else [0, 1])
        probabilities = np.asarray([[p, 1-p] for p in p0])
        self.probabilities = probabilities[:, ::-1] if reversed_classes else probabilities

    def predict_proba(self, inputs):
        return self.probabilities[:len(inputs)]

    def predict(self, inputs):
        return self.classes_[self.predict_proba(inputs).argmax(axis=1)]


class BudgetAndVoteTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([[str(i)] for i in range(4)])
        self.y = np.asarray([0, 0, 1, 1])
        self.a = FixedProbabilities([.9, .49, .2, .1])
        self.b = FixedProbabilities([.49, .9, .1, .2], reversed_classes=True)
        self.history = [
            {"trial": 1, "candidate": "logistic_C0.05", "status": "completed", "search_accuracy": .75, "params": {}},
            {"trial": 2, "candidate": "extra_trees_leaf1", "status": "completed", "search_accuracy": .75, "params": {}},
        ]
        self.models = {1: self.a, 2: self.b}

    def test_horizon_and_route_memory_without_identity_or_holdout_leak(self):
        rows = copy.deepcopy(self.history)
        rows[-1]["official_validation_accuracy"] = .91234567
        available = [{"id": "catboost_depth4_reg3.0", "params": {}}, {"id": "logistic_C0.25", "params": {}}]
        state = state_for({"task_name": "SECRET_NAME"}, rows, available, 3, "budget_routes")
        self.assertEqual(state["horizon"]["followups_after_this"], 0)
        self.assertTrue(state["horizon"]["final_attempt"])
        self.assertEqual(state["horizon"]["untried_routes"], ["catboost"])
        self.assertTrue(any(r["route"] == "logistic" and r["status"] == "paused" for r in state["routes"]))
        self.assertNotIn("SECRET_NAME", json.dumps(state))
        self.assertNotIn("0.91234567", json.dumps(state))
        self.assertNotIn("horizon", state_for({}, rows, available, 3, "evidence_routes"))

    def test_vote_aligns_reversed_classes_and_survives_serialization(self):
        vote = SoftVoteModel("categorical_tabular", [self.a, self.b], np.asarray([0, 1]))
        np.testing.assert_array_equal(predict_with_model("categorical_tabular", vote, self.x), self.y)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            joblib.dump(vote, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(self.x), vote.predict_proba(self.x))
        self.b.classes_ = np.asarray([1, 2])
        with self.assertRaisesRegex(ValueError, "different fitted classes"):
            vote.predict(self.x)

    def test_one_fixed_vote_best_per_route_and_reject_equal_accuracy(self):
        rows = self.history + [{**self.history[0], "trial": 3, "candidate": "logistic_C0.25", "search_accuracy": .5}]
        def load(row):
            self.assertNotEqual(row["trial"], 3)
            return self.models[row["trial"]]
        vote, record = propose_vote("categorical_tabular", rows, load, self.x, self.y, True)
        self.assertIsNotNone(vote)
        self.assertEqual(record["evaluations"], 1)
        self.assertEqual(len(record["components"]), 2)
        self.models[2] = self.a
        vote, record = propose_vote("categorical_tabular", rows, load, self.x, self.y, True)
        self.assertIsNone(vote)
        self.assertEqual(record["gain"], 0)

    def test_finalization_cannot_choose_using_validation_and_counts_refits(self):
        # Internal vote is better but external validation is deliberately worse.
        # It must still be selected, with both scores reported honestly.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            logger = RunLogger(directory)
            task = DevelopmentTask("categorical_tabular", "fixture", self.x, self.y, self.x, 1-self.y, {})
            payload = {"X_train": self.x, "y_train": self.y, "X_search": self.x, "y_search": self.y}
            for row in self.history:
                joblib.dump({"model": self.models[row["trial"]]}, directory / f"trial_{row['trial']}.joblib")
            def train(path, candidate, output, timeout):
                contents = joblib.load(path)
                self.assertEqual(set(contents), {"mode", "kind", "X_train", "y_train", "X_search",'metric_spec'})
                self.assertEqual(contents['metric_spec']['name'],'accuracy')
                np.testing.assert_array_equal(contents["y_train"], self.y)
                model = self.a if candidate.startswith("logistic") else self.b
                return model, {"status": "completed", "training_seconds": .5}
            # Alphabetical tie-break makes extra_trees the single winner.
            result = finalize(task, self.history, self.b, "extra_trees_leaf1", .75, payload,
                              directory, logger, train, time.monotonic(), 30, 10, True, True)
            self.assertTrue(result["ensemble_selected"])
            self.assertEqual(result["official_validation_accuracy"], 0.)
            self.assertEqual(result["single_model_validation_accuracy"], .25)
            self.assertEqual(result["full_refits_completed"], 2)
            self.assertEqual(result["refit_training_seconds"], 1.)
            events = [json.loads(line)["kind"] for line in (directory / "events.jsonl").read_text().splitlines()]
            self.assertLess(events.index("selection_frozen"), events.index("final_refit"))
            self.assertLess(events.index("final_model_frozen"), events.index("validation_scored"))


if __name__ == "__main__":
    unittest.main()
