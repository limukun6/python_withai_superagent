"""Estimate partially pooled response classes from fit-side histories only."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit, logit, softmax
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.run_task1_exposure import paired
from scripts.run_task1_profile_screen import digest, load_split, save
from scripts.task1_exposure_models import context_axis
from superagent.joint_context import ContextTokenTree, tokens
from superagent.splits import sample_groups


class ResponseClasses:
    def fit(self, x, labels, profile, context):
        self.profile, self.context = profile, context
        self.axis = context_axis(x, labels, context)
        people, p = np.unique(tokens(x[:, profile]), return_inverse=True)
        events, first, e = np.unique(tokens(x[:, context]), return_index=True, return_inverse=True)
        kinds, c = np.unique(x[:, self.axis], return_inverse=True)
        self.people = {key: i for i, key in enumerate(people)}
        self.events = {key: i for i, key in enumerate(events)}
        self.kinds = {key: i for i, key in enumerate(kinds)}
        count = np.zeros((len(people), len(events)))
        yes = np.zeros_like(count)
        np.add.at(count, (p, e), 1)
        np.add.at(yes, (p, e), labels)
        self.global_rate = (labels.sum() + 1) / (len(labels) + 2)
        event_kind = c[first]
        indicators = np.eye(len(kinds))[event_kind]
        count_kind, yes_kind = count @ indicators, yes @ indicators
        background_kind = (yes_kind.sum(axis=0) + 2 * self.global_rate) / (count_kind.sum(axis=0) + 2)
        background_event = (yes.sum(axis=0) + 8 * background_kind[event_kind]) / (count.sum(axis=0) + 8)
        groups = min(5, len(people))
        weights = np.random.default_rng(315).dirichlet(np.full(groups, .2), size=len(people))
        for iteration in range(60):
            self.prior = (weights.sum(axis=0) + 1) / (len(people) + groups)
            self.coarse = (weights.T @ yes_kind + 12 * background_kind) / (weights.T @ count_kind + 12)
            deviations = logit(self.coarse) - logit(background_kind)
            target = expit(logit(background_event) + deviations[:, event_kind])
            self.fine = (weights.T @ yes + 24 * target) / (weights.T @ count + 24)
            likelihood = yes @ np.log(self.fine.T) + (count - yes) @ np.log1p(-self.fine.T)
            posterior = softmax(likelihood + np.log(self.prior), axis=1)
            change = np.max(np.abs(weights - posterior))
            weights = posterior
            if change < 1e-6:
                break
        self.post = weights
        self.iterations = iteration + 1
        return self

    def predict(self, x, tree_probs):
        results = []
        known = 0
        rows = zip(tokens(x[:, self.profile]), tokens(x[:, self.context]), x[:, self.axis], tree_probs, strict=True)
        for key, event, kind, tree_prob in rows:
            theta = np.full(len(self.prior), self.global_rate)
            if event in self.events:
                theta = self.fine[:, self.events[event]]
            elif kind in self.kinds:
                theta = self.coarse[:, self.kinds[kind]]
            background = self.prior @ theta
            personal = placebo = background
            if key in self.people:
                known += 1
                i = self.people[key]
                personal = self.post[i] @ theta
                placebo = self.post[(i + 1) % len(self.post)] @ theta
            base = logit(np.clip(tree_prob, 1e-6, 1 - 1e-6))
            corrected = expit(base + .5 * (logit(personal) - logit(background)))
            rotated = expit(base + .5 * (logit(placebo) - logit(background)))
            results.append((corrected, rotated, personal))
        return np.asarray(results).T, known


def run(task, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    x, y, members = load_split(task, 'train')
    groups = sample_groups(SimpleNamespace(X_train=x, kind='categorical_tabular'))
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=112801).split(x, y, groups))
    reference = json.loads((root / 'research/task1_exposure_01/results.json').read_text(encoding='utf-8'))
    files = [Path(__file__), root / 'scripts/task1_exposure_models.py', root / 'superagent/joint_context.py',
             root / 'scripts/run_task1_exposure.py', root / 'docs/batches/TASK1_RESPONSE_CLASSES_PLAN_20261128_zh.md']
    save(out / 'protocol.json', dict(sources={f.name: digest(f.read_bytes()) for f in files}, members=members,
         held_sha=[digest(b.astype('<i8').tobytes()) for _, b in folds], seed=112801, api_calls=0))
    rows = []
    trees = mixtures = 0
    promoted = False
    try:
        for fold, (train, held) in enumerate(folds):
            if fold >= 3 and not promoted:
                break
            assert not set(groups[train]) & set(groups[held])
            trees += 1
            tree = ContextTokenTree(dict(pair_limit=3)).fit(x[train], y[train])
            ref = tree.predict_proba(x[held])[:, 1]
            uncertain = np.abs(ref - .5) <= .15
            scores = dict(baseline=paired(ref, ref, y[held], uncertain))
            if fold < len(reference):
                assert scores['baseline']['prediction_sha'] == reference[fold]['methods']['baseline']['prediction_sha']
            mixtures += 1
            with threadpool_limits(limits=2):
                model = ResponseClasses().fit(x[train], y[train], tree.profile, tree.context)
                probs, known = model.predict(x[held], ref)
            assert np.isfinite(probs).all() and np.all((probs >= 0) & (probs <= 1))
            for name, pred in zip(('corrected', 'rotated', 'standalone'), probs, strict=True):
                scores[name] = paired(pred, ref, y[held], uncertain)
            rows.append(dict(fold=fold, held=len(held), known_rows=known, iterations=model.iterations, methods=scores))
            save(out / 'results.json', rows)
            print(fold, {name: row['correct'] for name, row in scores.items()}, flush=True)
            if fold == 2:
                delta = [r['methods']['corrected']['delta_pp'] for r in rows]
                net_vs_rotated = sum(r['methods']['corrected']['correct'] - r['methods']['rotated']['correct'] for r in rows)
                promoted = bool(np.mean(delta) >= .5 and sum(v > 0 for v in delta) >= 2
                                and min(delta) >= -.5 and net_vs_rotated > 0)
                save(out / 'screen.json', dict(promoted=promoted, mean_pp=float(np.mean(delta)), net_vs_rotated=net_vs_rotated))
        net = [r['methods']['corrected']['fixed'] - r['methods']['corrected']['broken'] for r in rows]
        confirmed = promoted and sum(net[3:]) > 0 and 100 * sum(net) / sum(r['held'] for r in rows) >= .5
        save(out / 'completion.json', dict(status='complete', classifier_fits=trees, supervised_mixture_fits=mixtures,
             confirmed=confirmed, seconds=round(time.monotonic() - start, 3), api_calls=0, course_yuan=0,
             designated_validation_opened=False, formal_test_opened=False))
    except Exception as exc:
        save(out / 'completion.json', dict(status='failed', tree_admissions=trees, mixture_admissions=mixtures,
             error=repr(exc), api_calls=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.output)
