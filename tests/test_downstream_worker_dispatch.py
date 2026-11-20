import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.method_choice import PAIR_POSTERIOR, TEXT_MINILM, TEXT_MINILM_ONLY


class DownstreamWorkerDispatchTests(unittest.TestCase):
    def test_registered_new_methods_reach_the_real_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"pretrained_runtime.json").write_text("{}")
            words=["water boat","orange fruit","garden plant","blue ocean"]
            with sqlite3.connect(root/"minilm_category_vectors.sqlite") as db:
                db.execute("CREATE TABLE vectors(id TEXT PRIMARY KEY, v BLOB NOT NULL)")
                for i,w in enumerate(words):
                    db.execute("INSERT INTO vectors VALUES (?,?)",(hashlib.sha256(w.encode()).hexdigest(),
                               np.eye(4,dtype="<f4")[i].tobytes()))
            for name in (TEXT_MINILM,TEXT_MINILM_ONLY,PAIR_POSTERIOR):
                if name==PAIR_POSTERIOR:
                    kind="categorical_tabular"
                    x=np.asarray([[f"a{i%4}",f"b{i%7}"] for i in range(180)])
                    params=dict(backbone_params=dict(depth=3,l2_leaf_reg=3.),include_pairs=True,prior_count=20)
                else:
                    kind="pair_text"
                    x=[(words[i%4],words[(i+1)%4]) for i in range(180)]
                    params=dict(dense_only=name==TEXT_MINILM_ONLY)
                y=np.asarray([i%2 for i in range(180)])
                joblib.dump(dict(kind=kind,mode="search",X_train=x[:150],y_train=y[:150],
                                 X_search=x[150:],y_search=y[150:]),root/"data.joblib")
                (root/f"method_{name}.json").write_text(json.dumps(dict(decision="try",candidate=name,params=params)))
                model,metrics=train_candidate(root/"data.joblib",name,root/"trial.joblib",90)
                self.assertIsNotNone(model,metrics)
                self.assertEqual(metrics["classifier_fits"],1)
                self.assertEqual(metrics["resolved_configuration"]["id"],name)
                self.assertEqual(metrics["resolved_configuration"]["origin"],
                                 "training_only_method" if name==PAIR_POSTERIOR else "frozen_downstream_encoder")
            self.assertFalse((root/"downstream_usage.jsonl").exists())
