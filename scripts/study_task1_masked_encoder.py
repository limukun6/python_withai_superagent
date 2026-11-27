"""Compare a task-local masked categorical encoder against the same randomly initialized architecture."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from torch import nn
from torch.nn import functional as F

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.field_interactions import choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, blocks
from superagent.models import _tabular_catboost

PLAN = ROOT / "research/task1_feedback_extension_1127/masked_table_pretraining.json"
METHODS = tuple(f"{v}_{h}" for v in ("scratch", "pretrained") for h in ("tree", "neural"))


class TableEncoder(nn.Module):
    def __init__(self, sizes, n_classes):
        super().__init__()
        self.fields = len(sizes)
        self.width = max(sizes)
        self.register_buffer("offsets", torch.tensor(np.cumsum([0] + [n + 2 for n in sizes[:-1]])))
        self.register_buffer("valid", torch.arange(self.width)[None, :] < torch.tensor(sizes)[:, None])
        self.embedding = nn.Embedding(sum(n + 2 for n in sizes), 12)
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(len(sizes) * 12, 128), nn.LayerNorm(128),
                                      nn.GELU(), nn.Dropout(.12), nn.Linear(128, 24), nn.LayerNorm(24))
        self.decode = nn.Linear(24, len(sizes) * self.width)
        self.classify = nn.Linear(24, n_classes)

    def forward(self, rows):
        return self.backbone(self.embedding(rows + self.offsets))

    def reconstruction_loss(self, rows):
        mask = torch.rand(rows.shape) < .30
        representation = self(rows.masked_fill(mask, 0))
        logits = self.decode(representation).reshape(len(rows), self.fields, self.width)
        logits = logits.masked_fill(~self.valid[None, :, :], torch.finfo(logits.dtype).min)
        return F.cross_entropy(logits[mask], rows[mask] - 2)


def optimize(model, rows, labels=None):
    unsupervised = labels is None
    epochs = 40 if unsupervised else 24
    torch.manual_seed(46105 if unsupervised else 46106)
    generator = torch.Generator().manual_seed(76211 if unsupervised else 76212)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.001)
    losses = []
    steps = 0
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator)
        total = 0.
        for indices in order.split(192):
            batch = rows[indices]
            loss = model.reconstruction_loss(batch) if unsupervised else F.cross_entropy(model.classify(model(batch)), labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            steps += 1
        losses.append(total / len(rows))
    model.eval()
    return dict(epochs=epochs, steps=steps, first_epoch_loss=losses[0], last_epoch_loss=losses[-1])


def train(out):
    torch.set_num_threads(2)
    torch.manual_seed(46101)
    inputs = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", inputs)
    x, labels = fitting["x"], fitting["y"]
    classes, y = np.unique(labels, return_inverse=True)
    panels = [bound(out, r + "_features.joblib", inputs) for r in ROLES]
    query = np.concatenate(panels)
    columns = []
    partitions = set()
    for i, col in enumerate(x.T):
        codes, levels = pd.factorize(col)
        signature = codes.tobytes()
        if 2 <= len(levels) <= 256 and signature not in partitions:
            columns.append(i)
            partitions.add(signature)
    require(len(columns) >= 2, "No applicable categorical representation")
    mapper = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1)
    train_ids = torch.from_numpy(mapper.fit_transform(x[:, columns]) + 2)
    query_ids = torch.from_numpy(mapper.transform(query[:, columns]) + 2)
    initial = TableEncoder([len(v) for v in mapper.categories_], len(classes))
    learned = deepcopy(initial)
    shared = out / "masked_components"
    shared.mkdir(exist_ok=False)
    save(shared / "launch.json", dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"),
         rows=len(x), fields=columns, classes=classes.tolist(), tensor_runtime=torch.__version__, threads=2))
    pretrain = optimize(learned, train_ids)
    torch.save(learned.state_dict(), shared / "pretrained.pt")
    print("masked training", pretrain, flush=True)
    transformer = ContextTokenTree(dict(pair_limit=3))
    transformer.profile, transformer.context = blocks(x)
    transformer.pairs = choose_pairs(rank_pairs(x, labels), 3)
    train_cats = pd.DataFrame(transformer.transform(x))
    query_cats = pd.DataFrame(transformer.transform(query))
    offsets = np.cumsum([len(a) for a in panels])[:-1]
    for variant, base in (("scratch", initial), ("pretrained", learned)):
        base.eval()
        with torch.inference_mode():
            train_vectors = base(train_ids).numpy()
            query_vectors = base(query_ids).numpy()
        for head in ("tree", "neural"):
            directory = out / f"masked_{variant}_{head}"
            directory.mkdir(exist_ok=False)
            if head == "tree":
                def frame(categories, vectors):
                    return pd.concat([categories, pd.DataFrame(vectors)], axis=1, ignore_index=True)
                model = _tabular_catboost(labels, 6, 3.)
                model.set_params(thread_count=3)
                model.fit(frame(train_cats, train_vectors), labels, cat_features=list(range(train_cats.shape[1])))
                require(np.array_equal(model.classes_, classes), "Incorrect tree label order")
                prediction = model.predict_proba(frame(query_cats, query_vectors), thread_count=3)
                model_file = directory / "tree.cbm"
                model.save_model(str(model_file))
                detail = dict(tree_count=model.tree_count_)
            else:
                model = deepcopy(base)
                detail = optimize(model, train_ids, torch.from_numpy(y))
                with torch.inference_mode():
                    prediction = torch.softmax(model.classify(model(query_ids)), dim=1).numpy()
                model_file = directory / "network.pt"
                torch.save(model.state_dict(), model_file)
            require(prediction.shape == (len(query), len(classes)) and np.isfinite(prediction).all()
                    and np.allclose(prediction.sum(1), 1., atol=1e-5), "Invalid probabilities")
            signatures = {}
            for role, probabilities in zip(ROLES, np.split(prediction, offsets), strict=True):
                path = directory / (role + ".npy")
                np.save(path, probabilities, allow_pickle=False)
                signatures[role] = sha(path)
            save(directory / "sealed.json", dict(protocol=sha(PLAN), launch=sha(shared / "launch.json"),
                 classes=classes.tolist(), predictions=signatures, supervised_fits=1, model_sha256=sha(model_file),
                 training=detail, api_requests=0))
            print(variant, head, detail, flush=True)
    save(shared / "sealed.json", dict(launch=sha(shared / "launch.json"), self_supervised_fits=1,
         supervised_fits=4, pretrained_weights=sha(shared / "pretrained.pt"), pretraining=pretrain, api_requests=0))


def grade(out):
    elect(out, METHODS, "masked", PLAN)
    folder = out / "masked_election"
    election = read(folder / "results.json")
    winner, weight = election["winner"]
    inputs = read(out / "inputs.json")
    controls = {}
    for role in ROLES:
        def probabilities(name):
            area = out / name
            manifest = read(area / "sealed.json")
            path = area / (role + ".npy")
            require(sha(path) == manifest["predictions"][role], "Unbound head-to-head prediction")
            return np.load(path, allow_pickle=False), np.asarray(manifest["classes"])
        a, classes = probabilities("trial_attribute_alignment")
        b, other = probabilities("trial_pretrained_icl")
        require(np.array_equal(classes, other), "Unmatched reference labels")
        reference = (a + b) / 2
        labels = bound(out, role + "_answers.joblib", inputs)
        checks = {}
        for head in ("tree", "neural"):
            scores = {}
            for name in ("scratch", "pretrained"):
                p, order = probabilities(f"masked_{name}_{head}")
                require(np.array_equal(classes, order), "Unmatched head labels")
                scores[name] = classes[p.argmax(1)] == labels
                scores[name + "_mix"] = classes[((1 - weight) * reference + weight * p).argmax(1)] == labels
            checks[head] = dict(scratch_correct=int(scores["scratch"].sum()),
                raw=paired(scores["pretrained"], scores["scratch"]),
                selected_share=paired(scores["pretrained_mix"], scores["scratch_mix"]))
        controls[role] = checks
    primary = winner.split("_")[-1]
    eligible = False
    if winner.startswith("pretrained_") and election["eligible"]:
        changes = [controls[r][primary]["selected_share"]["net"] for r in ROLES[1:]]
        eligible = min(changes) >= 0 and sum(changes) > 0
    save(folder / "controls.json", dict(election=sha(folder / "results.json"), heads=controls, weight=weight,
                                        pretraining_qualified=eligible))
    print("pretraining qualified", eligible, controls, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    (train if args.action == "train" else grade)(args.out)
