import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

import joblib
import numpy as np

from superagent.benchmark import run_one, train_candidate
from superagent.confirmation import decide, shortlist
from superagent.data import DevelopmentTask, load_training_task
from superagent.splits import inner_split, overlap_audit, sample_groups


class ConfirmationTests(unittest.TestCase):
    def test_training_loader_does_not_open_validation_or_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"task.zip"
            with zipfile.ZipFile(path,"w") as archive:
                for name,values in (("C_train.npy",np.asarray([["a"],["b"]])),("y_train.npy",np.asarray([0,1]))):
                    buffer=io.BytesIO()
                    np.save(buffer,values)
                    archive.writestr(name,buffer.getvalue())
                archive.writestr("y_val.npy",b"forbidden invalid data")
                archive.writestr("y_test.npy",b"forbidden invalid data")
            task=load_training_task(path)
            self.assertEqual(len(task.y_train),2)
            self.assertFalse(hasattr(task,"y_val"))
            self.assertFalse(hasattr(task,"X_val"))

    def test_metrics_only_worker_retains_predictions_for_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            x=np.asarray([[str(i%3),str(i)] for i in range(60)])
            y=np.asarray([i%2 for i in range(60)])
            payload=root/"data.joblib"
            joblib.dump({"kind":"categorical_tabular","X_train":x[:40],"y_train":y[:40],
                         "X_search":x[40:],"y_search":y[40:],"retain_model":False},payload)
            model,metrics=train_candidate(payload,"logistic_C0.05",root/"result.joblib",20)
            artifact=joblib.load(root/"result.joblib")
            self.assertIsNone(model)
            self.assertEqual(metrics["status"],"completed")
            self.assertEqual(float(np.mean(artifact["search_predictions"]==y[40:])),metrics["search_accuracy"])

    def test_connected_text_groups_are_transitive_and_case_normalized(self):
        pairs = []
        for i in range(30):
            pairs.extend([(f"a {i}", f"b {i}"), (f" B   {i} ", f"c {i}")])
        task = SimpleNamespace(kind="pair_text", X_train=pairs, y_train=np.tile([0,1],30))
        groups = sample_groups(task, "shared_text")
        self.assertEqual(len(set(groups)), 30)
        self.assertTrue(all(groups[i] == groups[i+1] for i in range(0,60,2)))
        train, search = inner_split(task, 11, "shared_text")
        self.assertEqual(overlap_audit(task, train, search)["search_rows_sharing_text"], 0)
        self.assertFalse(set(groups[train]) & set(groups[search]))

    def test_categorical_grouping_stays_identical(self):
        task = SimpleNamespace(kind="categorical_tabular", X_train=np.asarray([[str(i%20)] for i in range(60)]),
                               y_train=np.tile([0,1],30))
        np.testing.assert_array_equal(sample_groups(task), sample_groups(task,"shared_text"))
        train, search = inner_split(task, 22, "shared_text")
        self.assertEqual(overlap_audit(task, train, search)["search_duplicate_rows"], 0)

    def test_insufficient_components_fail_instead_of_splitting_related_rows(self):
        task = SimpleNamespace(kind="pair_text", X_train=[("common",str(i)) for i in range(20)], y_train=np.tile([0,1],10))
        with self.assertRaisesRegex(ValueError,"two independent groups"):
            inner_split(task,11,"shared_text")

    def test_fixed_shortlist_mean_ranking_and_missing_result_fallback(self):
        history = [{"candidate": name,"status":"completed","search_accuracy":score} for name,score in (("a",.8),("b",.7),("c",.6),("d",.5))]
        top = shortlist(history,3)
        self.assertEqual([r["candidate"] for r in top],["a","b","c"])
        seeds = [11,10011,20011]
        values = {name:[{"seed":seed,"status":"completed","search_accuracy":score} for seed,score in zip(seeds,scores)]
                  for name,scores in (("a",[.8,.6,.6]),("b",[.7,.8,.8]),("c",[.6,.7,.7]))}
        decision = decide(top,values,seeds,"exact_rows")
        self.assertTrue(decision["valid_confirmation"])
        self.assertEqual(decision["selected_candidate"],"b")
        values["c"][-1]={"seed":seeds[-1],"status":"timeout"}
        decision = decide(top,values,seeds,"exact_rows")
        self.assertFalse(decision["valid_confirmation"])
        self.assertEqual(decision["selected_candidate"],"a")

    def test_repeated_scores_tie_by_candidate_id_not_list_order(self):
        history = [{"candidate":name,"status":"completed","search_accuracy":.7} for name in ("b","a")]
        values = {name:[{"seed":s,"status":"completed","search_accuracy":.7} for s in (1,10001,20001)] for name in ("a","b")}
        self.assertEqual(decide(shortlist(history),values,[1,10001,20001],"exact_rows")["selected_candidate"],"a")

    def test_real_worker_confirmation_payload_has_no_external_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"run"
            x=np.asarray([[str(i%3),str(i)] for i in range(120)])
            y=np.asarray([i%2 for i in range(120)])
            task=DevelopmentTask("categorical_tabular","fixture",x,y,x[:12],1-y[:12],{})
            payloads=[]
            def checked_train(payload_path,*args):
                payload=joblib.load(payload_path)
                self.assertNotIn("X_val",payload)
                self.assertNotIn("y_val",payload)
                self.assertFalse((path/"validation_predictions.json").exists())
                payloads.append(payload)
                return train_candidate(payload_path,*args)
            with patch("superagent.benchmark.train_candidate",side_effect=checked_train):
                result=run_one(task,"random",11,path,None,max_trials=2,anchor_first=True,refit_full_train=True,
                               confirmation_repeats=3,confirmation_top_k=3)
            self.assertTrue(result["confirmation"]["valid_confirmation"])
            self.assertEqual(result["confirmation"]["new_fits_attempted"],4)
            self.assertEqual(len(payloads),7)  # 2 search + 4 confirmation + 1 full refit.
            self.assertEqual(sum(p.get("retain_model") is False for p in payloads),4)
            final=json.loads((path/"final_selection.json").read_text())
            self.assertEqual(final["selection_metric"],"confirmation_mean_accuracy")
            self.assertTrue((path/"validation_predictions.json").exists())

    def test_incompatible_selection_modes_fail_before_any_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"absent"
            task=SimpleNamespace(kind="pair_text")
            with self.assertRaisesRegex(ValueError,"cannot be combined"):
                run_one(task,"budget_routes",11,path,None,confirmation_repeats=3,ensemble_selection=True)
            self.assertFalse(path.exists())


if __name__ == "__main__": unittest.main()
