"""Infer numerical order from units and intervals in otherwise categorical data."""
import argparse
import functools
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.attribute_alignment import comparable_fields, fitted_routes
from superagent.field_interactions import choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, blocks
from superagent.models import _clean_categories, _tabular_catboost
from superagent.splits import sample_groups

PLAN = ROOT / "research/task1_feedback_extension_1127/ordered_measurements.json"
METHODS = ("measured", "relative")
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
DURATIONS = {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60,
             "h": 3600, "hr": 3600, "hour": 3600, "d": 86400, "day": 86400, "week": 604800}


@functools.lru_cache(maxsize=4096)
def interpret(token):
    value = str(token).strip().lower().replace(",", "").replace("–", "-").replace("—", "-")
    clock = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", value)
    if clock:
        hours = int(clock[1])
        minutes = int(clock[2] or 0)
        if 1 <= hours <= 12 and minutes < 60:
            return "clock", (hours % 12 + 12 * (clock[3] == "pm")) * 60. + minutes
    clock = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", value)
    if clock:
        return "clock", 60. * int(clock[1]) + int(clock[2])
    duration = re.fullmatch(f"({NUMBER})"+r"\s*([a-z]+)", value)
    if duration:
        unit = duration[2]
        factor = DURATIONS.get(unit, DURATIONS.get(unit.removesuffix("s")))
        if factor:
            return "duration", float(duration[1]) * factor
    currency = set(re.findall(r"[$€£¥]", value))
    if len(currency) > 1:
        return None
    kind = "money_" + next(iter(currency)) if currency else "number"
    value = re.sub(r"[$€£¥]", "", value).strip()
    if value in ("never", "zero") and not currency:
        return kind, 0.
    if re.fullmatch(NUMBER, value):
        return kind, float(value)
    interval = re.fullmatch(f"({NUMBER})"+r"\s*(?:-|~|to|\.\.)\s*"+f"({NUMBER})", value)
    if interval and float(interval[1]) <= float(interval[2]):
        return kind, .5 * (float(interval[1]) + float(interval[2]))
    for pattern, direction in ((r"(?:<|lt|less(?:\s+than)?|under|below|fewer\s+than)", -1),
                               (r"(?:>|gt|greater(?:\s+than)?|above|over|more\s+than)", 1)):
        match = re.fullmatch(pattern+r"\s*("+NUMBER+")", value)
        if match:
            endpoint = float(match[1])
            return kind, endpoint + direction * .01 * max(1., abs(endpoint))
    plus = re.fullmatch(f"({NUMBER})"+r"\s*(?:\+|plus|or more|and above|or higher)", value)
    if plus:
        endpoint = float(plus[1])
        return kind, endpoint + .01 * max(1., abs(endpoint))
    return None


def recognize(x):
    kinds = {}
    for j, col in enumerate(x.T):
        levels, count = np.unique(col, return_counts=True)
        available = [(interpret(v), n) for v, n in zip(levels, count)
                     if str(v).strip().lower() not in ("", "__missing__", "nan", "n/a", "null")]
        histogram = {}
        for entry, frequency in available:
            if entry is not None:
                histogram[entry[0]] = histogram.get(entry[0], 0) + frequency
        if not histogram:
            continue
        dominant = max(histogram, key=histogram.get)
        scaled = [m[1] for m, _ in available if m is not None and m[0] == dominant]
        if histogram[dominant] >= .85 * sum(n for _, n in available) and len(set(scaled)) >= 2:
            kinds[j] = dominant
    return kinds


def quantities(x, kinds):
    matrix = np.full(x.shape, np.nan)
    numeric = []
    for j, kind in kinds.items():
        vals = [interpret(v) for v in x[:, j]]
        matrix[:, j] = [s[1] if s and s[0] == kind else np.nan for s in vals]
        numeric.append(matrix[:, j])
        if kind == "clock":
            angle = 2 * np.pi * matrix[:, j] / 1440.
            numeric.extend((np.sin(angle), np.cos(angle)))
    return matrix, np.column_stack(numeric)


def relative_values(x, values, families, routes):
    all_values = []
    for family, (gate, table, default) in zip(families, routes):
        ix = np.repeat(default, len(x)) if gate is None else np.array([table.get(v, default) for v in x[:, gate]])
        selected = values[np.arange(len(x)), ix]
        group = values[:, family]
        observed = np.isfinite(group).sum(axis=1)
        mean = np.nansum(group, axis=1) / np.maximum(observed, 1)
        rank = (group < selected[:, None]).sum(axis=1) / np.maximum(observed, 1)
        all_values.extend((selected, np.where(observed, mean, np.nan), selected-mean, np.where(np.isfinite(selected), rank, np.nan)))
    return np.column_stack(all_values)


def combine(categories, continuous):
    return pd.concat([pd.DataFrame(categories), pd.DataFrame(continuous)], axis=1, ignore_index=True)


class MeasurementTree(ContextTokenTree):
    def fit(self, rows, y):
        x = _clean_categories(rows)
        self.profile, self.context = blocks(x)
        self.pairs = choose_pairs(rank_pairs(x, y), self.params["pair_limit"])
        cats = ContextTokenTree.transform(self, x)
        self.kinds = recognize(x)
        require(self.kinds, "No coherent numeric fields")
        values, numeric = quantities(x, self.kinds)
        self.families = []
        self.routes = []
        if self.params["relative"]:
            self.families = [family for family in comparable_fields(x, self.profile)
                             if all(j in self.kinds for j in family) and len({self.kinds[j] for j in family}) == 1]
        if self.families:
            groups = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=x))
            extra = np.empty((len(x), 4 * len(self.families)))
            for train, held in GroupKFold(3).split(x, y, groups):
                routes = fitted_routes(x[train], y[train], self.families, self.context)
                extra[held] = relative_values(x[held], values[held], self.families, routes)
            self.routes = fitted_routes(x, y, self.families, self.context)
            numeric = np.column_stack((numeric, extra))
        self.classifier = _tabular_catboost(y, 6, 3.)
        self.classifier.set_params(thread_count=6)
        self.classifier.fit(combine(cats, numeric), y, cat_features=list(range(cats.shape[1])))
        self.classes_ = self.classifier.classes_
        return self

    def transform(self, rows):
        x = _clean_categories(rows)
        cats = ContextTokenTree.transform(self, x)
        values, numeric = quantities(x, self.kinds)
        if self.families:
            numeric = np.column_stack((numeric, relative_values(x, values, self.families, self.routes)))
        return combine(cats, numeric)


def run(out):
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    panels = [bound(out, r+"_features.joblib", info) for r in ROLES]
    for name in METHODS:
        folder = out / ("measure_" + name)
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        model = MeasurementTree(dict(relative=name == "relative", pair_limit=3)).fit(fitting["x"], fitting["y"])
        signatures = {}
        for role, panel in zip(ROLES, panels):
            p = np.asarray(model.predict_proba(panel), dtype=float)
            require(np.isfinite(p).all() and np.allclose(p.sum(1), 1.), "Invalid probability")
            path = folder / (role + ".npy")
            np.save(path, p, allow_pickle=False)
            signatures[role] = sha(path)
        record = dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"), classes=model.classes_.tolist(), fits=1,
                      field_kinds=list(model.kinds.values()), families=list(map(len, model.families)),
                      supervised_routing_estimates=4 if model.families else 0, predictions=signatures,
                      seconds=round(time.monotonic()-start, 2), api_requests=0)
        save(folder / "sealed.json", record)
        print(name, record["field_kinds"], record["families"], record["seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "elect"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "run":
        run(args.out)
    else:
        elect(args.out, METHODS, "measure", PLAN)
