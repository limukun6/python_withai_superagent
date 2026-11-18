import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from superagent.benchmark import run_one
from superagent.data import DevelopmentTask
from superagent.evidence_agent import latest_reading, measured_facts, parse_evidence, route_board, state_for
from superagent.models import search_candidates
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM


def trial(number, candidate, score, status="completed", proposal=None):
    return {"trial": number, "candidate": candidate, "params": {}, "status": status,
            "train_accuracy": .85, "search_accuracy": score, "generalization_gap": .85-score if score else None,
            "training_seconds": 1., "proposal": proposal}


def proposal(candidate, history, routes=True, blind=False):
    from superagent.evidence_agent import route_of
    latest = history[-1]["trial"]
    selected = route_of(candidate)
    prior = {route_of(r["candidate"]) for r in history}
    last = route_of(history[-1]["candidate"])
    value = {"candidate": candidate, "reason": "Check another configuration.",
             "hypothesis": "The proposed configuration may improve the search score.",
             "review": "Outcomes unavailable." if blind else "The measured difference applies to tested configurations.",
             "evidence_trials": [] if blind else [latest], "expected_change": None,
             "assessment": {"reading": latest_reading(history, not blind),
                            "fact_ids": [] if blind else [f"t{latest}.status"],
                            "interpretation": "Possible explanation, not an established cause.",
                            "uncertainty": "Untested configurations and sampling variation remain unknown."}}
    if routes:
        value["route_plan"] = {"next_check": "Check another parameter only if the remaining budget supports it.",
                               "revisit_previous_if": "Return if the alternative does not improve the score."}
    return value


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.history = [trial(1, "word_C0.05", .70), trial(2, "word_C0.1", .698)]
        self.available = [{"id": "word_C0.3", "params": {"C": .3}}, {"id": "word_char_C0.1", "params": {"C": .1}}]

    def test_small_drop_is_a_measurement_not_route_elimination(self):
        self.assertEqual(latest_reading(self.history), "lower")
        board = route_board(self.history, self.available)
        word = next(r for r in board if r["route"] == "word")
        self.assertEqual(word["status"], "active")
        self.assertEqual(word["best_search_accuracy"], .70)
        self.assertEqual(word["remaining_configurations"], 1)

    def test_failed_execution_has_no_zero_score_and_can_be_revisited(self):
        failed = trial(3, "word_char_C0.1", None, "timeout")
        history = self.history + [failed]
        self.assertEqual(latest_reading(history), "execution_failure")
        self.assertFalse(any(f["trial"] == 3 and f["metric"] == "search_accuracy" for f in measured_facts(history)))
        answer = proposal("word_C0.3", history)
        parsed = parse_evidence(answer, history, self.available[:1], "evidence_routes")
        self.assertEqual(parsed["route_update"]["action"], "revisit")

    def test_paused_route_note_survives_and_remaining_configuration_stays_available(self):
        answer = proposal("word_char_C0.1", self.history)
        parsed = parse_evidence(answer, self.history, self.available, "evidence_routes")
        history = self.history + [trial(3, "word_char_C0.1", .72, proposal=parsed)]
        board = route_board(history, self.available[:1] + [{"id": "word_char_C0.3", "params": {"C": .3}}])
        word = next(r for r in board if r["route"] == "word")
        self.assertEqual(word["status"], "paused")
        self.assertIn("alternative", word["resume_if"])
        self.assertEqual(word["remaining_configurations"], 1)

    def test_false_reading_fabricated_fact_and_false_revisit_are_rejected(self):
        for change in ("reading", "fact", "action"):
            answer = proposal("word_char_C0.1", self.history)
            if change == "reading": answer["assessment"]["reading"] = "improved"
            if change == "fact": answer["assessment"]["fact_ids"] = ["t99.search_accuracy"]
            if change == "action": answer["route_plan"]["action"] = "revisit"
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_evidence(answer, self.history, self.available, "evidence_routes")

    def test_leaving_route_requires_condition_but_continuing_is_allowed(self):
        answer = proposal("word_char_C0.1", self.history)
        answer["route_plan"]["revisit_previous_if"] = ""
        with self.assertRaises(ValueError): parse_evidence(answer, self.history, self.available, "evidence_routes")
        self.assertEqual(parse_evidence(proposal("word_C0.3", self.history), self.history, self.available,
                                        "evidence_routes")["route_update"]["action"], "continue")

    def test_blind_arm_and_holdout_do_not_leak(self):
        history = copy.deepcopy(self.history)
        history[0]["search_accuracy"] = .918273645
        history[1]["official_validation_accuracy"] = .99987123
        state = state_for({"task_name": "PRIVATE_TASK"}, history, self.available, 5, "evidence_blind")
        self.assertEqual(state["facts"], [])
        self.assertEqual(state["observations"], [])
        for sentinel in ("0.918273645", "0.99987123", "PRIVATE_TASK"):
            self.assertNotIn(sentinel, json.dumps(state))
        parsed = parse_evidence(proposal("word_C0.3", history, routes=False, blind=True), history, self.available, "evidence_blind")
        self.assertEqual(parsed["cited_facts"], [])

    def test_parameter_definitions_match_for_all_arms(self):
        cards = [state_for({}, self.history, self.available, 5, strategy)["parameter_definitions"]
                 for strategy in ("evidence_blind", "evidence_feedback", "evidence_routes")]
        self.assertEqual(cards[0], cards[1])
        self.assertEqual(cards[1], cards[2])
        self.assertIn("STRONGER", cards[0]["C"])


class RefitIntegrationTest(unittest.TestCase):
    def test_full_refit_uses_only_train_and_reloads_identically(self):
        # Real local worker/model; validation labels are never in its payload.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.asarray([[str(i % 3), str(i)] for i in range(90)])
            y = np.asarray([i % 2 for i in range(90)])
            task = DevelopmentTask("categorical_tabular", "synthetic", x, y, x[:12], y[:12], {})
            result = run_one(task, "random", 11, root / "run", None, max_trials=1,
                             anchor_first=True, refit_full_train=True)
            self.assertTrue(result["refit_succeeded"])
            self.assertEqual(result["final_training_rows"], 90)
            payload = joblib.load(root / "run/refit_data.joblib")
            self.assertEqual(set(payload), {"mode", "kind", "X_train", "y_train", "X_search",'metric_spec'})
            self.assertEqual(payload['metric_spec']['name'],'accuracy')
            self.assertEqual(len(payload["X_search"]), 1)
            self.assertEqual(len(payload["y_train"]), 90)
            from superagent.models import predict_with_model
            model = joblib.load(root / "run/selected_model.joblib")
            prediction = predict_with_model(task.kind, model["model"], task.X_val)
            saved = json.loads((root / "run/validation_predictions.json").read_text())
            self.assertEqual(prediction.tolist(), saved)


class GatewayTests(unittest.TestCase):
    def test_reasoning_none_uses_gateway_parameter_and_truncation_is_still_billed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = root / "key"
            key.write_text("dummy-test-key")
            logger = RunLogger(root)
            logger.initialize_usage_ledger()
            client = ExperimentLLM("https://example.invalid/chat", str(key), "DeepSeek-V4.1-Flash", .1, reasoning_effort="none",
                                   disable_template_thinking=True)
            with patch("superagent.experiment_llm.requests.post") as post:
                post.return_value.ok = True
                post.return_value.json.return_value = {"usage": {"prompt_tokens": 100, "completion_tokens": 2048},
                    "choices": [{"finish_reason": "length", "message": {"content": "{\"unfinished\":"}}]}
                with self.assertRaisesRegex(ValueError, "Output token limit"):
                    client.ask({}, logger, "test")
                body = post.call_args.kwargs["json"]
                self.assertEqual(body["reasoning_effort"], "none")
                self.assertNotIn("thinking", body)
                self.assertEqual(body["chat_template_kwargs"], {"thinking": False, "enable_thinking": False})
                self.assertEqual(body["cache"], {"no-cache": True, "no-store": True})
                self.assertEqual(post.call_count, 1)
                self.assertAlmostEqual(client.spent, .008292)
                self.assertIn('"finish_reason": "length"', logger.usage_path.read_text())


if __name__ == "__main__": unittest.main()
