import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.method_choice import PAIR_ITEM, PAIR_OPPONENT, PAIR_POSTERIOR, TEXT_DEEP, TEXT_MINILM, TEXT_MINILM_ONLY, TEXT_REGULARIZED


class DownstreamWorkerDispatchTests(unittest.TestCase):
    def test_registered_new_methods_reach_the_real_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"pretrained_runtime.json").write_text("{}")
            (root/"method_policy.json").write_text(json.dumps({"mode":"encoder-mixture"}))
            words=["water boat","orange fruit","garden plant","blue ocean"]
            for cache in ("minilm_category_vectors.sqlite","minilm_l12_vectors.sqlite"):
                with sqlite3.connect(root/cache) as db:
                    db.execute("CREATE TABLE vectors(id TEXT PRIMARY KEY, v BLOB NOT NULL)")
                    for i,w in enumerate(words):
                        db.execute("INSERT INTO vectors VALUES (?,?)",(hashlib.sha256(w.encode()).hexdigest(),
                                   np.eye(4,384,dtype="<f4")[i].tobytes()))
            for name in (TEXT_MINILM,TEXT_MINILM_ONLY,TEXT_DEEP,PAIR_ITEM,PAIR_OPPONENT,TEXT_REGULARIZED,PAIR_POSTERIOR):
                if name==PAIR_POSTERIOR:
                    kind="categorical_tabular"
                    x=np.asarray([[f"a{i%4}",f"b{i%7}"] for i in range(180)])
                    params=dict(backbone_params=dict(depth=3,l2_leaf_reg=3.),include_pairs=True,prior_count=20)
                else:
                    kind="pair_text"
                    x=[(words[i%4],words[(i+1)%4]) for i in range(180)]
                    params=dict(dense_only=name==TEXT_MINILM_ONLY,rank=96,projection_seed=9181,opponent_context=name==PAIR_OPPONENT,C=1.2)
                y=np.asarray([i%2 for i in range(180)])
                payload=dict(kind=kind,mode="search",X_train=x[:150],y_train=y[:150],
                             X_search=x[150:],y_search=y[150:])
                joblib.dump(payload,root/"data.joblib")
                (root/f"method_{name}.json").write_text(json.dumps(dict(decision="try",candidate=name,params=params)))
                model,metrics=train_candidate(root/"data.joblib",name,root/"trial.joblib",90)
                self.assertIsNotNone(model,metrics)
                if name==TEXT_REGULARIZED:self.assertEqual(model.classifier.C, 1.2)
                self.assertEqual(metrics["classifier_fits"],1)
                self.assertEqual(metrics["resolved_configuration"]["id"],name)
                self.assertEqual(metrics["resolved_configuration"]["origin"],
                                 "training_only_method" if name==PAIR_POSTERIOR else "frozen_downstream_encoder")
                if name in (TEXT_MINILM, TEXT_DEEP):
                    primary=joblib.load(root/"trial.joblib")
                    joblib.dump(dict(payload,retain_model=False),root/"data.joblib")
                    retained,confirmation=train_candidate(root/"data.joblib",name,root/"confirmation.joblib",90)
                    self.assertIsNone(retained)
                    self.assertEqual(confirmation["status"],"completed")
                    self.assertEqual(confirmation["classifier_fits"],1)
                    checked=joblib.load(root/"confirmation.joblib")
                    np.testing.assert_array_equal(primary["search_scores"],checked["search_scores"])
                    np.testing.assert_array_equal(checked["score_classes"][checked["search_scores"].argmax(axis=1)],
                                                  checked["search_predictions"])
            self.assertFalse((root/"downstream_usage.jsonl").exists())
