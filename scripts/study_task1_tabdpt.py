"""Evaluate two input representations of the frozen public TabDPT 1.1 downstream classifier."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import torch

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_extensions import key
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.models import _clean_categories

PLAN = ROOT / "research/task1_feedback_extension_1127/tabdpt_model.json"
METHODS = ("ordinal", "onehot")


def make_model(weight, device):
    from tabdpt import TabDPTClassifier
    model = TabDPTClassifier(model_weight_path=str(weight), device="cpu", use_flash=False, compile=False,
                             inf_batch_size=192, verbose=False)
    if device != "cpu":
        model.to(device)
    return model


def verify(weight):
    asset = read(PLAN)["upstream"]
    require(weight.stat().st_size == asset["bytes"] and sha(weight) == asset["sha256"]
            and version("tabdpt") == "1.1.14", "Unverified frozen model or runtime")
    torch.set_num_threads(3)
    return asset["sha256"]


def smoke(out, weight, device):
    digest = verify(weight)
    rng = np.random.default_rng(381)
    x = rng.normal(size=(52, 6)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    model = make_model(weight, device)
    model.fit(x[:42], y[:42])
    probs = model.predict_proba(x[42:], context_size=42)
    require(probs.shape == (10, 2) and np.isfinite(probs).all() and np.allclose(probs.sum(1), 1., atol=1e-5),
            "Failed synthetic inference")
    save(out / "tabdpt_probe.json", dict(protocol=sha(PLAN), model_sha256=digest, device=device, rows=10,
         synthetic_context_fits=1, context_rows=42, successful=True))


def train(out, weight, device):
    digest = verify(weight)
    probe = read(out / "tabdpt_probe.json")
    require(probe["successful"] and probe["model_sha256"] == digest and probe["device"] == device, "Different probe")
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    x = _clean_categories(fitting["x"])
    classes, y = np.unique(fitting["y"], return_inverse=True)
    panels = [_clean_categories(bound(out, r + "_features.joblib", info)) for r in ROLES]
    query = np.concatenate(panels)
    sizes = np.cumsum([len(a) for a in panels])[:-1]
    contexts = [idx for idx, _ in StratifiedShuffleSplit(2, train_size=512, random_state=71813).split(x, y)]
    shared = out / "tabdpt_components"
    shared.mkdir(exist_ok=False)
    save(shared / "launch.json", dict(protocol=sha(PLAN), probe=sha(out / "tabdpt_probe.json"),
         inputs=sha(out / "inputs.json"), device=device, classes=classes.tolist(),
         context_indices=[key(ix) for ix in contexts], context_size=512, package=version("tabdpt")))
    for name in METHODS:
        folder = out / ("tabdpt_" + name)
        folder.mkdir(exist_ok=False)
        enc = (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
               if name == "ordinal" else OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))
        train_features = enc.fit_transform(x)
        query_features = enc.transform(query)
        bank = []
        for i, indices in enumerate(contexts):
            torch.manual_seed(7234 + i)
            started = time.monotonic()
            model = make_model(weight, device)
            save(folder / f"context_{i}_attempt.json", dict(launch=sha(shared / "launch.json"),
                 rows=len(indices), input_dimensions=train_features.shape[1], seed=7234 + i))
            model.fit(train_features[indices], y[indices])
            output = np.asarray(model.predict_proba(query_features, context_size=len(indices)), dtype=float)
            require(output.shape == (len(query), len(classes)) and np.isfinite(output).all()
                    and np.allclose(output.sum(1), 1., atol=1e-5), "Invalid foundation-model probability")
            path = folder / f"context_{i}.npy"
            np.save(path, output, allow_pickle=False)
            bank.append(output)
            seconds = round(time.monotonic() - started, 2)
            save(folder / f"context_{i}_done.json", dict(predictions=sha(path), seconds=seconds,
                 frozen_context_fits=1, weights_updated=False))
            print(name, i, "seconds", seconds, flush=True)
            del model
            gc.collect()
        signatures = {}
        for role, probabilities in zip(ROLES, np.split(np.mean(bank, axis=0), sizes), strict=True):
            path = folder / (role + ".npy")
            np.save(path, probabilities, allow_pickle=False)
            signatures[role] = sha(path)
        save(folder / "sealed.json", dict(protocol=sha(PLAN), launch=sha(shared / "launch.json"),
             model_sha256=digest, classes=classes.tolist(), predictions=signatures, fitted_feature_encoders=1,
             frozen_context_fits=len(bank), context_fits=[sha(folder / f"context_{i}_done.json") for i in range(len(bank))],
             model_api_requests=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    parser.add_argument("--weight", type=lambda p: Path(p).resolve())
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.action == "grade":
        elect(args.out, METHODS, "tabdpt", PLAN)
    else:
        (smoke if args.action == "smoke" else train)(args.out, args.weight, args.device)
