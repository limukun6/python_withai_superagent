import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import numpy as np

from superagent.ab_run import confirmation_coverage
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_training_task
from superagent.metrics import SUPPORTED
from scripts import unified_encoder_portfolio as portfolio


class AnonymousGoalTests(unittest.TestCase):
    def package(self, folder, text, *, manifest=None, native="csv"):
        path = Path(folder)/"any-name.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("README.md", text)
            z.writestr("test/README.md", "Primary metric: accuracy.")
            if native == "csv":
                z.writestr("train.csv", "a,label\n1,minority\n2,usual\n3,third\n")
            else:
                import io
                buf = io.BytesIO()
                shape = (4, 3, 3) if native=="image" else (4, 2)
                np.savez(buf, X=np.ones(shape), y=np.array([0,1,0,1]))
                z.writestr("train.npz", buf.getvalue())
            if manifest is not None:
                z.writestr("manifest.json", json.dumps(manifest))
        return path

    def test_all_supported_names_and_provenance_in_native_training_and_development(self):
        for expected in SUPPORTED:
            with self.subTest(metric=expected), tempfile.TemporaryDirectory() as tmp:
                path = self.package(tmp, f"Primary metric: **{expected.replace('_', '-')}**.", native="numeric")
                for loader in (load_training_task, load_development_task):
                    spec = TaskSpec.from_task(loader(path)).record()
                    self.assertEqual(spec["metric"]["name"], expected)
                    goal = spec["goal_resolution"]["fields"]["primary_metric"]
                    self.assertEqual((goal["source"], goal["status"]), ("declared_task_readme","confirmed"))

    def test_defaults_precedence_and_primary_versus_secondary(self):
        cases = [("Primary metric: macro-F1. Also report accuracy.", "macro_f1", "declared_task_readme"),
                 ("Report weighted F1.", "weighted_f1", "declared_task_readme"),
                 ("Report test accuracy on the held-out rows.", "accuracy", "declared_task_readme"),
                 ("We might report macro-F1.", "accuracy", "default_accuracy"),
                 ("Report accuracy. Report F1.", "accuracy", "default_accuracy"),
                 ("No objective provided.", "accuracy", "default_accuracy")]
        for text, name, source in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                spec = load_training_task(self.package(tmp, text)).spec
                self.assertEqual((spec['metric'],spec['goal_sources']['primary_metric']), (name, source))
        with tempfile.TemporaryDirectory() as tmp:
            declared = dict(schema_version=1, modality="tabular", target="label", metric="accuracy",
                            columns=[dict(name="a",type="numeric")], splits={"train":{"format":"csv","path":"train.csv"}})
            spec = load_training_task(self.package(tmp, "Primary metric: macro-F1.",manifest=declared)).spec
            self.assertEqual((spec["metric"],spec["goal_sources"]["primary_metric"]),("accuracy","declared_manifest"))

    def test_invalid_primary_and_positive_class_declarations_are_not_silently_scored(self):
        cases = ["Primary metric: log loss.", "Primary metric: F1.",
                 "Primary metric: macro-F1. Primary metric: accuracy.",
                 "Primary metric: binary-F1. Positive class: absent."]
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    load_training_task(self.package(tmp, text,native="numeric"))

    def test_explicit_class_zero_in_image_npz_and_no_evaluation_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.package(tmp, "Primary metric: ROC-AUC.\nPositive class: 0.",native="image")
            original = zipfile.ZipFile.open
            def only_train(archive, name, *args, **kwargs):
                if archive.filename and Path(archive.filename).resolve() == path.resolve():
                    self.assertIn(getattr(name, "filename", name), ("train.npz","README.md"))
                return original(archive, name, *args, **kwargs)
            with patch.object(zipfile.ZipFile, "open", only_train):
                task = load_training_task(path)
            goal = TaskSpec.from_task(task).record()
            self.assertEqual((task.kind,goal["metric"]["labels"]), ("image",[1,0]))
            self.assertEqual(goal["goal_resolution"]["fields"]["positive_label"]["source"],"declared_task_readme")


class ConfirmationCoverageTests(unittest.TestCase):
    def test_distinct_partition_can_reuse_a_row_or_a_group(self):
        trusted = {"primary":{"row_ids":[0,1],"groups":["a","b"]},
                   "confirmation_1":{"row_ids":[1,2,3],"groups":["b","a","c"]}}
        coverage = confirmation_coverage(trusted, trusted, trusted)
        self.assertTrue(coverage["different_confirmation_split_used"])
        self.assertFalse(coverage["independent_confirmation_observed"])
        self.assertEqual(coverage["evaluation_coverage"]["used_for_selection"]["confirmation_1"],
                         dict(rows=3, groups=3, reused_rows=1, reused_groups=2))
        skipped = confirmation_coverage(trusted, trusted, ["primary"])
        self.assertFalse(skipped["different_confirmation_split_used"])
        self.assertIn("confirmation_1", skipped["evaluation_coverage"]["executed"])
        self.assertNotIn("confirmation_1", skipped["evaluation_coverage"]["used_for_selection"])

    def test_disjoint_groups_required_for_all_used_confirmations(self):
        trusted = {n:{"row_ids":[i],"groups":[str(i)]} for i,n in enumerate(("primary","confirmation_1","confirmation_2"))}
        self.assertTrue(confirmation_coverage(trusted,trusted,trusted)["independent_confirmation_observed"])
        trusted["confirmation_2"]["groups"] = ["1"]
        self.assertFalse(confirmation_coverage(trusted,trusted,trusted)["independent_confirmation_observed"])
        with self.assertRaises(ValueError):
            confirmation_coverage(trusted,["primary"],["confirmation_1"])


class PortfolioFallbackTests(unittest.TestCase):
    def test_incompatible_task_runs_common_trainer_and_records_reason(self):
        task = SimpleNamespace(kind="numeric_tabular", spec={}, y_train=np.array([0,1]))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"run"
            def common(package, directory):
                directory.mkdir()
                return 0
            with patch.object(portfolio,"configure"), patch.object(portfolio,"load_training_task",return_value=task), \
                    patch.object(portfolio,"unified_train",side_effect=common) as trainer:
                self.assertEqual(portfolio.train("anonymous.zip",folder,Path(tmp),None),0)
            trainer.assert_called_once_with("anonymous.zip",folder)
            self.assertEqual(json.loads((folder/"portfolio_availability.json").read_text())["reason"],"incompatible_task_contract")
