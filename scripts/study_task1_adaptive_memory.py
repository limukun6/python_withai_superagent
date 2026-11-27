"""Train an inductive categorical similarity function on residuals of each specific held-out teacher."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch import nn
from torch.nn import functional as F

from scripts.study_task1_candidates import ROLES, bound, paired
from scripts.study_task1_extensions import key
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.attribute_alignment import AttributeAlignmentTree
from superagent.joint_context import blocks
from superagent.models import _clean_categories
from superagent.splits import sample_groups

PLAN = ROOT / "research/task1_feedback_extension_1127/adaptive_residual_memory.json"


class ResidualMemory(nn.Module):
    def __init__(self, sizes, profile, context):
        super().__init__()
        self.profile = list(profile)
        self.context = list(context)
        self.register_buffer("offsets", torch.tensor(np.cumsum([0] + sizes[:-1])))
        self.embed = nn.Embedding(sum(sizes), 10)
        self.conditional = nn.Linear(10, len(sizes))
        self.gate = nn.Linear(11, 1)
        self.field = nn.Parameter(torch.zeros(len(sizes)))
        self.blocks = nn.Parameter(torch.ones(2))
        nn.init.zeros_(self.conditional.weight)
        nn.init.zeros_(self.conditional.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, query, base, bank, residuals, allowed=None):
        state = self.embed(query + self.offsets).mean(1)
        weights = F.softplus(self.field + torch.tanh(self.conditional(state)))
        matches = query[:, None, :] == bank[None, :, :]
        similarity = (matches * weights[:, None, :]).sum(2)
        block_weight = F.softplus(self.blocks)
        similarity = similarity + block_weight[0] * matches[:, :, self.profile].all(2)
        similarity = similarity + block_weight[1] * matches[:, :, self.context].all(2)
        if allowed is not None:
            similarity = similarity.masked_fill(~allowed, -torch.inf)
        cutoff = similarity.topk(min(64, len(bank)), dim=1).values[:, -1:]
        attention = similarity.masked_fill(similarity < cutoff, -torch.inf).softmax(1)
        effective = 1. / attention.square().sum(1)
        strength = self.gate(torch.cat((state, (base - .5).abs()[:, None]), dim=1)).sigmoid().flatten()
        return strength * effective / (effective + 12.) * (attention @ residuals)


def fit_metric(codes, base, target, groups, folds, sizes, profiles, conditions):
    torch.manual_seed(62413)
    rng = np.random.default_rng(49553)
    model = ResidualMemory(sizes, profiles, conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.012, weight_decay=.012)
    residual = target - base
    losses = []
    for _ in range(22):
        sum_loss = 0.
        for fold in rng.permutation(len(folds)):
            indices = folds[fold]
            ordered = rng.permutation(indices)
            for start in range(0, len(ordered), 64):
                query = ordered[start:start + 64]
                ids = rng.choice(indices, min(384, len(indices)), replace=False)
                allowed = groups[query, None] != groups[None, ids]
                require(allowed.any(axis=1).all(), "No disjoint memory available")
                adjustment = model(codes[query], base[query], codes[ids], residual[ids], torch.from_numpy(allowed))
                probability = torch.clamp(base[query] + adjustment, .002, .998)
                loss = F.binary_cross_entropy(probability, target[query])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sum_loss += float(loss.detach()) * len(query)
        losses.append(sum_loss / len(base))
    model.eval()
    return model, dict(epochs=22, first_log_loss=losses[0], last_log_loss=losses[-1],
                       uncorrected_log_loss=float(F.binary_cross_entropy(base, target)))


def cached_probability(out, source, role, classes):
    folder = out / source
    seal = read(folder / "sealed.json")
    file = folder / (role + ".npy")
    require(seal["classes"] == classes.tolist() and sha(file) == seal["predictions"][role],
            "Unbound base probabilities")
    return np.load(file, allow_pickle=False)[:, 1]


def train(out):
    torch.set_num_threads(2)
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    x = _clean_categories(fitting["x"])
    classes, y = np.unique(fitting["y"], return_inverse=True)
    require(len(classes) == 2, "Binary residuals required")
    group_ids = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=fitting["x"]))
    profiles, conditions = blocks(x)
    source = out / "distill_teachers"
    seal = read(source / "sealed.json")
    require(seal["inputs"] == sha(out / "inputs.json") and seal["overlap"] == 0
            and sha(source / "out_of_fold.npy") == seal["predictions"], "Unbound prior teachers")
    existing = np.load(source / "out_of_fold.npy", allow_pickle=False)
    aligned = np.full(len(x), np.nan)
    folds = []
    shared = out / "adaptive_memory_components"
    shared.mkdir(exist_ok=False)
    for i, (fitting_ids, held) in enumerate(StratifiedGroupKFold(4, shuffle=True, random_state=72731).split(x, y, group_ids)):
        record = seal["folds"][i]
        require(key(fitting_ids) == record["fitting_sha256"] and key(held) == record["held_sha256"]
                and not set(group_ids[held]).intersection(group_ids[fitting_ids]), "Mismatched teacher fold")
        model = AttributeAlignmentTree(dict(pair_limit=3)).fit(x[fitting_ids], y[fitting_ids])
        require(np.array_equal(model.classes_, [0, 1]), "Unexpected label order")
        aligned[held] = model.predict_proba(x[held])[:, 1]
        folds.append(held)
        model.classifier.save_model(str(shared / f"aligned_{i}.cbm"))
        save(shared / f"teacher_{i}.json", dict(protocol=sha(PLAN), held_rows=len(held),
             fitting=key(fitting_ids), held=key(held), tree=sha(shared / f"aligned_{i}.cbm"), fits=1))
    require(np.isfinite(aligned).all() and np.isfinite(existing).all(), "Missing out-of-group predictions")
    np.save(shared / "aligned_out_of_fold.npy", aligned, allow_pickle=False)
    save(shared / "teachers.json", dict(protocol=sha(PLAN), new_classifier_fits=4, reused_teacher_fits=8,
         borrowed=sha(source / "sealed.json"), probabilities=sha(shared / "aligned_out_of_fold.npy")))
    mapper = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1)
    codes = torch.from_numpy(mapper.fit_transform(x) + 1)
    sizes = [len(c) + 1 for c in mapper.categories_]
    groups = np.unique(group_ids, return_inverse=True)[1]
    target = torch.tensor(y, dtype=torch.float32)
    for name, values in (("anchor", existing[:, 0]), ("pair", (aligned + existing[:, 1]) / 2)):
        folder = out / ("adaptmem_" + name)
        folder.mkdir(exist_ok=False)
        base = torch.tensor(values, dtype=torch.float32)
        metric, training = fit_metric(codes, base, target, groups, folds, sizes, profiles, conditions)
        torch.save(metric.state_dict(), folder / "metric.pt")
        signatures = {}
        with torch.inference_mode():
            for role in ROLES:
                features = _clean_categories(bound(out, role + "_features.joblib", info))
                q = torch.from_numpy(mapper.transform(features) + 1)
                if name == "anchor":
                    start = cached_probability(out, "trial_anchor", role, classes)
                else:
                    start = (cached_probability(out, "trial_attribute_alignment", role, classes)
                             + cached_probability(out, "trial_pretrained_icl", role, classes)) / 2
                prior = torch.tensor(start, dtype=torch.float32)
                corrections = []
                for pos in range(0, len(q), 64):
                    batch = q[pos:pos + 64]
                    local = prior[pos:pos + 64]
                    corrections.append(torch.stack([metric(batch, local, codes[ix], (target - base)[ix])
                                                    for ix in folds]).mean(0))
                prob = np.clip(start + torch.cat(corrections).numpy(), .002, .998)
                require(np.isfinite(prob).all(), "Invalid corrected probabilities")
                file = folder / (role + ".npy")
                np.save(file, np.column_stack((1 - prob, prob)), allow_pickle=False)
                signatures[role] = sha(file)
        save(folder / "sealed.json", dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"),
             teachers=sha(shared / "teachers.json"), metric=sha(folder / "metric.pt"), classes=classes.tolist(),
             predictions=signatures, supervised_metric_fits=1, training=training, api_requests=0))
        print(name, training, flush=True)


def grade(out):
    elect(out, ("pair",), "adaptmem", PLAN)
    info = read(out / "inputs.json")
    cls = np.asarray(read(out / "trial_anchor/sealed.json")["classes"])
    scores = {}
    for role in ROLES:
        truth = bound(out, role + "_answers.joblib", info)
        ref = cached_probability(out, "trial_anchor", role, cls)
        new = cached_probability(out, "adaptmem_anchor", role, cls)
        scores[role] = paired(cls[(new > .5).astype(int)] == truth, cls[(ref > .5).astype(int)] == truth)
    save(out / "adaptmem_election/anchor.json", scores)
    print(scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "grade"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    (train if args.action == "train" else grade)(args.out)
