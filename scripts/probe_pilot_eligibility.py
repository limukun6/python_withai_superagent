"""Measure avoidable refusal of feasible sparse-label pilot roles at the native fit boundary."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.probe_challenger_dispatch import FirstFitReached, before_fit
from scripts.unified_released_study import ROOT, read, require, save, sha, text_packet
from superagent import ab_run
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools import pilot_challenger as pilot
from tools.crossfit_challenger import training_roles
import unified_agent

PLAN = ROOT / "docs/batches/pilot_small_group_plan_20261124_zh.md"


def independent_layout(groups):
    def order(keys, prefix):
        return sorted(set(keys), key=lambda s: hashlib.sha256((prefix+s).encode()).digest())
    ranked = order(groups, "crossfit-challenger-v1/")
    pilot_set = {k for i, k in enumerate(ranked) if 1 <= i%9 <= 4}
    other = order(set(ranked) - pilot_set, "pilot-feedback-v1/")
    return {k: np.flatnonzero(np.isin(groups, list(values))) for k, values in
            (("pilot", pilot_set), ("learn", other[::2]), ("confirm", other[1::2]))}


def old_eligibility(groups, y):
    layout = independent_layout(groups)
    return layout if training_roles(groups, y) is not None and valid(layout, groups, y) else None


def valid(parts, groups, y):
    return all(len(set(groups[ix])) >= 4 and set(y[ix]) == {0, 1} for ix in parts.values())


def route(packet, folder, weights, strategy):
    args = ["unified_agent", "train", str(packet), "--run-dir", str(folder), "--strategy", strategy,
            "--weights", str(weights)]
    with patch.dict(os.environ), patch.object(sys, "argv", args), patch.object(ab_run, "Executor", side_effect=before_fit), \
            patch.object(pilot, "fit_one", side_effect=before_fit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            unified_agent.main()
        except FirstFitReached:
            pass
        else:
            raise ValueError("No fit boundary reached")
    if (folder / "pilot_protocol.json").exists():
        return {"route": "pilot", "candidates": sorted(read(folder / "pilot_protocol.json")["inventory"]["candidates"])}
    manifest = read(folder / "agent/run_manifest.json")
    return {"route": "common", "candidates": sorted(manifest["capabilities"])}


def probe(weights, packet_dir):
    rng = np.random.default_rng(241124)
    matrix, fixture = [], None
    for n in (45, 90, 180):
        rows = [(f"north district {i}", f"south district {i}") for i in range(n)]
        g = sample_groups(SimpleNamespace(kind="pair_text", X_train=rows), "shared_text")
        layout = independent_layout(g)
        for pct in (.05, .10, .20):
            missed = wrong = eligible = 0
            for _ in range(200):
                y = np.zeros(n, dtype=int)
                y[rng.choice(n, max(2, round(n*pct)), replace=False)] = 1
                allowed = valid(layout, g, y)
                received = pilot.pilot_roles(g, y)
                eligible += allowed
                missed += allowed and received is None
                wrong += not allowed and received is not None
                if received is not None:
                    require(all(np.array_equal(received[k], v) for k, v in layout.items()), "Incorrect group allocation")
                if fixture is None and n >= 90 and y.sum() >= 8 and allowed and training_roles(g, y) is None:
                    fixture = [(a, b, int(yi)) for (a,b), yi in zip(rows, y)]
            matrix.append(dict(groups=n, positive_fraction=pct, trials=200, feasible=eligible,
                               rejected_feasible=missed, admitted_infeasible=wrong))
    require(fixture is not None, "Sparse example unavailable")
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        file = base / "anonymous.zip"
        text_packet(file, fixture, True)
        observed = {s: route(file, base / s, weights, s) for s in pilot.STRATEGIES}
        source = sha(file)
    registered = {}
    for i in range(5):
        folder = packet_dir / f"fold_{i}"
        task = load_training_task(folder / "train.zip")
        g = sample_groups(task, "shared_text")
        got = pilot.pilot_roles(g, task.y_train)
        expected = read(folder / "pilot-accuracy/run/pilot_protocol.json")["roles"]
        require(all(fingerprint(v.tolist()) == expected[k]["ids"] for k,v in got.items()), "Course roles changed")
        registered[str(i)] = {k: fingerprint(v.tolist()) for k,v in got.items()}
    return dict(plan=sha(PLAN), matrix=matrix, fixture=source, fixture_routes=observed, development_roles=registered,
                classifier_fits=0, model_api_calls=0)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("packets", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--emulate-old", action="store_true")
    a=p.parse_args()
    if a.emulate_old:
        with patch.object(pilot, "pilot_roles", old_eligibility):
            result = probe(a.weights.resolve(), a.packets)
    else:
        result = probe(a.weights.resolve(), a.packets)
    save(a.output, result)
    print({k: sum(row[k] for row in result["matrix"]) for k in ("trials", "feasible", "rejected_feasible", "admitted_infeasible")})
    print({s: row["route"] for s, row in result["fixture_routes"].items()})
