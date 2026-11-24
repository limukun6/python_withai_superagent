import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import joblib
import numpy as np

from scripts.unified_released_study import read, sha
from superagent.ab_complement import PreferenceProbabilityVote, canonical_probabilities
from superagent.metrics import MetricSpec
from superagent.models import SoftVoteModel
from superagent.predict import create_predictions
from superagent.prediction_evidence import fingerprint, make_prediction_record
from superagent.score_evidence import make_score_record
from tools.preference_handoff import inputs, predict
from tools.unified_handoff import FeedbackPreferenceVote, attach, frozen
from unified_agent import pair_packet, save


class FixedProbabilities(SoftVoteModel):
    def predict_proba(self, rows):
        p = np.full(len(rows), self.models[0])
        return np.column_stack([1-p, p])


class MarkerProbabilities(SoftVoteModel):
    def predict_proba(self, rows):
        p = np.asarray([.02 if a.startswith("a-low") else .42 for a, _ in rows])
        return np.column_stack([1-p, p])


class OptionalHandoffTests(unittest.TestCase):
    def make_delivery(self, agent, model):
        spec = {"input_schema": {"columns": [{"name": "a", "type": "text"}, {"name": "b", "type": "text"}],
                    "modality": "pair_text", "target": "label", "pair_semantics": "preference", "pair_label_one": "left"},
                "task_kind": "pair_text_binary", "metric": "accuracy",
                "metric_spec": {"name": "accuracy", "labels": [0, 1], "direction": "maximize"}}
        spec["input_columns"] = spec["input_schema"]["columns"]
        save(agent / "task_spec.json", spec)
        with (agent / "selected_model.joblib").open("xb") as output:
            joblib.dump({"kind": "pair_text", "candidate": "equal_soft_vote", "model": model}, output)
        save(agent / "final_selection.json", {"family": "equal_soft_vote", "ensemble_components": ["left", "right"],
             "selected_experiment": "synthetic", "model_format": "candidate_bundle_v1", "model_file": "selected_model.joblib",
             "test_accessed": False, "model_sha256": sha(agent / "selected_model.joblib"),
             "task_spec_sha256": sha(agent / "task_spec.json")})
        return spec

    def test_active_gate_survives_native_prediction_export(self):
        members = [FixedProbabilities("pair_text", [p], np.array([0, 1])) for p in (.65, .7, .1)]
        policy = {"kind": "soft", "center": [0]*6, "scale": [1]*6, "coefficients": [2, 0, 0, 0, 0, 0, 0]}
        model = FeedbackPreferenceVote("pair_text", members, np.array([0, 1]), policy)
        pairs = [("a", "z"), ("z", "a"), ("z", "z")]
        data = inputs(np.column_stack([canonical_probabilities(m, pairs)[1][:, 1] for m in members]), pairs)
        expected = predict(policy, data)
        self.assertEqual(expected.tolist(), [0, 1, 0])
        self.assertEqual(data["base"].tolist(), [1, 0, 0])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.make_delivery(root, model)
            packet = root / "features.zip"
            pair_packet(packet, SimpleNamespace(spec=spec, X_train=pairs), range(len(pairs)), training=False)
            path = create_predictions(str(packet), root, root / "out/predictions.csv")
            with path.open() as source:
                result = list(csv.DictReader(source))
            self.assertEqual([(int(r["row_id"]), int(r["prediction"])) for r in result], list(enumerate(expected)))
            self.assertEqual(read(path.parent / "prediction_manifest.json")["test_labels_read"], False)

    def test_exhausted_fit_path_uses_native_delivery_and_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = root / "agent"
            agent.mkdir()
            self.make_delivery(agent, None)
            save(agent / "classifier_fits.json", {"attempted_classifier_fits": 11, "completed_classifier_fits": 11,
                                                  "unknown_failed_attempts": 0})
            save(agent / "prediction_evidence.json", [])
            (agent / "refit_data.joblib").touch()
            decision = attach(root)
            self.assertEqual(decision["chosen"], "baseline")
            self.assertEqual(decision["extra_fits"], 0)
            self.assertEqual(frozen(root)[1], agent)
            self.assertFalse((root / "conditional_handoff/admission.json").exists())
            self.assertEqual(attach(root), decision)

    def test_native_attach_selects_and_exports_a_useful_challenger_from_training_feedback(self):
        classes = np.array([0, 1])
        members = [FixedProbabilities("pair_text", [v], classes) for v in (.55, .60)]
        members.append(MarkerProbabilities("pair_text", [], classes))
        names = ["left", "right", "candidate-from-feedback"]
        rows = [(("a-high" if i % 4 == 0 else "a-low") + str(i), "z" + str(i)) for i in range(360)]
        y = np.asarray([int(a.startswith("a-high")) for a, _ in rows])
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            agent = run / "agent"
            agent.mkdir()
            spec = self.make_delivery(agent, PreferenceProbabilityVote("pair_text", members[:2], classes))
            save(agent / "classifier_fits.json", {"attempted_classifier_fits": 3, "completed_classifier_fits": 3,
                                                  "unknown_failed_attempts": 0})
            joblib.dump({"X_train": rows, "y_train": y}, agent / "refit_data.joblib")
            metric = MetricSpec.from_payload(spec)
            evidence = []
            for offset, split in ((0, "primary"), (120, "confirmation_1")):
                ids = list(range(offset, offset + 120))
                split_rows = [rows[i] for i in ids]
                payload = agent / (split + ".joblib")
                joblib.dump({"X_search": split_rows, "y_search": y[ids], "metric_spec": metric.record()}, payload)
                save(agent / (split + "_split.json"), {"search_indices": ids, "row_ids_sha256": fingerprint(ids),
                                                       "train_indices": list(range(240, 360))})
                for index, (name, member) in enumerate(zip(names, members, strict=True), 1):
                    _, p = canonical_probabilities(member, split_rows)
                    predicted = (p[:, 1] > .5).astype(int)
                    record = make_prediction_record(predicted, y[ids], sha(payload))
                    evidence.append({**record, "candidate_id": name, "split_id": split, "row_ids": ids, "trial": index,
                         "predictions": predicted.tolist(), "score_record": make_score_record(record, p, classes, metric, name)})
                    if split == "primary" and name == names[-1]:
                        joblib.dump({"model": member, "score_classes": classes, "search_scores": p}, agent / "trial_3.joblib")
            save(agent / "prediction_evidence.json", evidence)
            save(agent / "trial_3.fits.json", {"candidate": names[-1]})
            receipt = attach(run)
            self.assertEqual((receipt["chosen"], receipt["challenger"], receipt["extra_fits"]), ("soft", names[-1], 1))
            packet = run / "features.zip"
            pair_packet(packet, SimpleNamespace(spec=spec, X_train=rows), range(240, 360), training=False)
            output = create_predictions(str(packet), frozen(run)[1], run / "out/predictions.csv")
            with output.open() as stream:
                predictions = [int(r["prediction"]) for r in csv.DictReader(stream)]
            self.assertEqual(predictions, y[240:].tolist())
            self.assertEqual(attach(run), receipt)
