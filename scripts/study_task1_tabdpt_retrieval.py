"""Evaluate official frozen TabDPT nearest-neighbour inference using every available fitting row."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabdpt import make_model, verify
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.models import _clean_categories

PLAN = ROOT / "research/task1_feedback_extension_1127/tabdpt_retrieval.json"
CONTEXTS = (128, 512)


class IndexedQueries:
    def __init__(self, queries, indices, count):
        self.queries, self.indices, self.count = queries, indices, count
        self.position = 0

    def get_knn_indices(self, queries, k):
        start = self.position
        self.position += len(queries)
        if k != self.count or not np.array_equal(queries, self.queries[start:self.position]):
            raise ValueError("Published inference consumed different neighbour queries")
        return self.indices[start:self.position]


def run(out, weight, device):
    digest = verify(weight)
    first = read(out / "tabdpt_election/results.json")
    require(first["winner"][0] == "onehot", "Previous one-hot representation was not nominated")
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    classes, y = np.unique(fitting["y"], return_inverse=True)
    x = _clean_categories(fitting["x"])
    panels = [_clean_categories(bound(out, s + "_features.joblib", info)) for s in ROLES]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    rows = enc.fit_transform(x)
    query = enc.transform(np.concatenate(panels))
    cuts = np.cumsum([len(p) for p in panels])[:-1]
    shared = out / "tabdpt_neighbours_02"
    shared.mkdir(exist_ok=False)
    save(shared / "launch.json", dict(protocol=sha(PLAN), runtime=sha(PLAN.with_name("tabdpt_retrieval_runtime.json")),
         inputs=sha(out / "inputs.json"),
         nomination=sha(out / "tabdpt_election/results.json"), weight=digest, encoding="onehot",
         fit_rows=len(rows), query_rows=len(query), device=device, dimensions=rows.shape[1], inf_batch_size=8))
    model = make_model(weight, device)
    model.inf_batch_size = 8
    torch.manual_seed(71321)
    model.fit(rows, y)
    save(shared / "fitted.json", dict(launch=sha(shared / "launch.json"), foundation_context_fits=1, fitted_vocabularies=1))
    processed = model.scaler.transform(model.imputer.transform(query))
    np.save(shared / "index_train.npy", model.X_train, allow_pickle=False)
    np.save(shared / "index_query.npy", processed, allow_pickle=False)
    subprocess.run([sys.executable, "-m", "scripts.tabdpt_neighbour_index", str(shared), *map(str, CONTEXTS)],
                   check=True, timeout=180)
    for count in CONTEXTS:
        start = time.monotonic()
        index_file = shared / f"index_{count}.npy"
        indices = np.load(index_file, allow_pickle=False)
        require(indices.shape == (len(query), count) and indices.min() >= 0 and indices.max() < len(rows),
                "Invalid FAISS neighbours")
        lookup = IndexedQueries(processed, indices, count)
        model.faiss_knn = lookup
        result = np.asarray(model.predict_proba(query, context_size=count), dtype=float)
        require(lookup.position == len(query), "Incomplete query coverage")
        require(result.shape == (len(query), len(classes)) and np.isfinite(result).all()
                and np.allclose(result.sum(1), 1., atol=1e-5), "Invalid retrieved probabilities")
        area = out / f"tabretrieve_k{count}"
        area.mkdir(exist_ok=False)
        signatures = {}
        for role, p in zip(ROLES, np.split(result, cuts), strict=True):
            path = area / (role + ".npy")
            np.save(path, p, allow_pickle=False)
            signatures[role] = sha(path)
        sec = round(time.monotonic() - start, 2)
        save(area / "sealed.json", dict(protocol=sha(PLAN), fitted=sha(shared / "fitted.json"), classes=classes.tolist(),
             predictions=signatures, reference_rows=count, query_contexts=len(query), index_sha256=sha(index_file),
             seconds=sec, api_requests=0))
        print(count, len(query), sec, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "grade"))
    parser.add_argument("--out", required=True, type=lambda p:Path(p).resolve())
    parser.add_argument("--weight", type=lambda p:Path(p).resolve())
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.action == "run":
        run(args.out, args.weight, args.device)
    else:
        elect(args.out, tuple(f"k{k}" for k in CONTEXTS), "tabretrieve", PLAN)
