"""Compare two train-only support policies for the publisher's frozen Mitra core."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN = ROOT / "research/task1_feedback_extension_1127/mitra_context.json"
UPSTREAM = ("config/enums.py", "models/base.py", "models/embedding.py", "models/tab2d.py")
METHODS = ("global", "local")


def prepare(wheel, runtime):
    require(sha(wheel) == read(PLAN)["provenance"]["upstream_wheel_sha256"], "Unrecognized upstream wheel")
    # Only the publisher's frozen inference core is needed. Leave the shared AutoGluon environment untouched.
    package = runtime / "mitra_study_core" / "_internal"
    with zipfile.ZipFile(wheel) as archive:
        for relative in UPSTREAM:
            contents = archive.read("autogluon/tabular/models/mitra/_internal/" + relative)
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(target.read_bytes() == contents, "Upstream source changed")
            else:
                target.write_bytes(contents)


def setup_core(runtime):
    sys.path.insert(0, str(runtime))
    from mitra_study_core._internal.models.tab2d import Tab2D
    return Tab2D


def predictors(x, queries, seed):
    rng = np.random.default_rng(seed)
    train = np.empty(x.shape, dtype=np.float32)
    query = np.empty(queries.shape, dtype=np.float32)
    for j in range(x.shape[1]):
        levels, inverse = np.unique(x[:, j].astype(str), return_inverse=True)
        ids = rng.permutation(len(levels))
        train[:, j] = ids[inverse]
        parsed = queries[:, j].astype(str)
        ix = np.searchsorted(levels, parsed)
        ix = np.minimum(ix, len(levels) - 1)
        query[:, j] = np.where(levels[ix] == parsed, ids[ix], -1)
    return train, query


def contexts(x, y, queries, name, size=384):
    rng = np.random.default_rng(64892)
    if name == "global":
        remaining = np.arange(len(y))
        for i in range(2):
            split = StratifiedShuffleSplit(1, train_size=size, random_state=5771 + i)
            subset, rest = next(split.split(remaining, y[remaining]))
            yield remaining[subset], np.arange(len(queries))
            remaining = remaining[rest]
        return
    centers = [int(rng.integers(len(y)))]
    closest = np.ones(len(y)) * (x.shape[1] + 1)
    for _ in range(3):
        closest = np.minimum(closest, (x != x[centers[-1]]).sum(axis=1))
        centers.append(int(rng.choice(np.flatnonzero(closest == closest.max()))))
    assignments = np.argmin(np.stack([(queries != x[c]).sum(axis=1) for c in centers]), axis=0)
    for i, c in enumerate(centers):
        order = np.lexsort((rng.random(len(y)), (x != x[c]).sum(axis=1)))
        required = [order[np.flatnonzero(y[order] == k)[0]] for k in np.unique(y)]
        support = np.concatenate((required, order[~np.isin(order, required)]))[:size]
        yield support, np.flatnonzero(assignments == i)


def logits(model, x, y, q, classes, device):
    import torch
    def tensor(value, dtype):
        return torch.as_tensor(value, device=device, dtype=dtype).unsqueeze(0)
    fields = tensor(np.zeros(x.shape[1], dtype=bool), torch.bool)
    held = tensor(np.zeros(len(x), dtype=bool), torch.bool)
    predict = tensor(np.zeros(len(q), dtype=bool), torch.bool)
    with torch.inference_mode():
        values = model(tensor(x, torch.float32), tensor(y, torch.long), tensor(q, torch.float32), fields, held, predict)
        return torch.softmax(values[0, :, :classes], dim=-1).cpu().numpy()


def load_model(runtime, weight_dir):
    import torch
    origin = read(PLAN)["provenance"]
    require(sha(weight_dir / "model.safetensors") == origin["weights_sha256"], "Unverified Mitra weights")
    require(sha(weight_dir / "config.json") == origin["config_sha256"], "Unverified Mitra configuration")
    torch.set_num_threads(3)
    cls = setup_core(runtime)
    model = cls.from_pretrained(str(weight_dir), device="cpu").eval()
    devices = (["mps"] if torch.backends.mps.is_available() else []) + ["cpu"]
    errors = {}
    rng = np.random.default_rng(62753)
    for device in devices:
        model = model.to(device)
        try:
            smoke = logits(model, rng.normal(size=(20, 4)), np.tile([0, 1], 10), rng.normal(size=(5, 4)), 2, device)
            require(smoke.shape == (5, 2) and np.isfinite(smoke).all(), "Invalid synthetic smoke probabilities")
            return model, device, errors
        except (RuntimeError, NotImplementedError, ValueError) as error:
            errors[device] = str(error)[:300]
    raise RuntimeError(f"No working execution device: {errors}")


def run(out, runtime, weights):
    model, device, fallback = load_model(runtime, weights)
    info = read(out / "inputs.json")
    fit = bound(out, "fitting.joblib", info)
    x, y = fit["x"], np.asarray(fit["y"])
    classes, labels = np.unique(y, return_inverse=True)
    features = [bound(out, role + "_features.joblib", info) for role in ROLES]
    queries = np.concatenate(features)
    for name in METHODS:
        folder = out / ("mitra_" + name)
        folder.mkdir(exist_ok=True)
        if (folder / "sealed.json").exists():
            existing = read(folder / "sealed.json")
            require(existing["protocol"] == sha(PLAN) and existing["inputs"] == sha(out / "inputs.json")
                    and all(sha(folder / (r + ".npy")) == existing["predictions"][r] for r in ROLES),
                    "Unverifiable previous stage")
            continue
        counts = np.zeros(len(queries), dtype=int)
        bank = np.zeros((len(queries), len(classes)), dtype=float)
        started = time.monotonic()
        log = []
        for i, (support, qix) in enumerate(contexts(x, labels, queries, name)):
            train, query = predictors(x, queries, 887 + i)
            for begin in range(0, len(qix), 96):
                pick = qix[begin:begin + 96]
                prediction = logits(model, train[support], labels[support], query[pick], len(classes), device)
                require(np.isfinite(prediction).all() and np.allclose(prediction.sum(1), 1, atol=1e-5),
                        "Invalid prediction")
                bank[pick] += prediction
                counts[pick] += 1
            log.append(dict(context=i, support=len(support), queries=len(qix)))
            print(name, log[-1], round(time.monotonic()-started, 1), flush=True)
        require((counts == (2 if name == "global" else 1)).all(), "Incomplete query coverage")
        signatures = {}
        for role, values in zip(ROLES, np.split(bank / counts[:, None], np.cumsum([len(f) for f in features])[:-1])):
            target = folder / (role + ".npy")
            np.save(target, values, allow_pickle=False)
            signatures[role] = sha(target)
        save(folder / "sealed.json", dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"), classes=classes.tolist(),
             predictions=signatures, contexts=log, trainable_fits=0, device=device, fallback=fallback,
             seconds=round(time.monotonic()-started, 2), api_requests=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "elect"))
    for arg in ("wheel", "runtime", "weights", "out"):
        parser.add_argument("--"+arg, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.wheel, args.runtime)
    elif args.action == "elect":
        elect(args.out, METHODS, "mitra", PLAN)
    else:
        run(args.out, args.runtime, args.weights)
