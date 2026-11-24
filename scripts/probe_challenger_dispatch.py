"""Exercise the actual unified parser and both execution branches, stopping at their first fit boundary."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from scripts.audit_anonymous_goal import csv_bytes
from scripts import unified_encoder_portfolio as portfolio
from scripts.unified_released_study import ROOT, make_zip, read, require, save, sha
from superagent import ab_run
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from tools import challenger_training as training
from tools.encoder_assets import BGE
import unified_agent


CASES = (("numbers", "numeric_tabular", "roc_auc"), ("fields", "mixed_tabular", "macro_f1"),
         ("text", "single_text", "accuracy"), ("pixels", "image", "macro_f1"),
         ("equivalent", "pair_text", "accuracy"), ("preference", "pair_text", "accuracy"))
PLAN = ROOT / "docs/batches/crossfit_reachability_plan_20261124_zh.md"


def packet(path, name, metric):
    n = 180
    x = np.random.default_rng(428).normal(size=(n, 4))
    manifest = dict(schema_version=1, objective="classification", metric=metric)
    if name in ("numbers", "pixels"):
        manifest["modality"] = "image" if name == "pixels" else "tabular"
        data = io.BytesIO()
        if name == "numbers":
            manifest["positive_label"] = 0
            np.savez(data, X=x, y=np.arange(n) % 2)
        else:
            np.savez(data, X=x.reshape(n, 2, 2), y=np.arange(n) % 3)
        member, body = "inputs.npz", data.getvalue()
    else:
        if name == "fields":
            modality, types = "tabular", ("numeric", "categorical")
            rows = [(x[i, 0], str(i % 5), str(i % 3)) for i in range(n)]
        elif name == "text":
            modality, types = "text", ("text",)
            rows = [(f"{'oak' if i % 2 else 'pine'} sample {i}", str(i % 2)) for i in range(n)]
        else:
            modality, types = "pair_text", ("text", "text")
            symmetric = name == "equivalent"
            manifest.update(pair_semantics="symmetric" if symmetric else "preference",
                            pair_label_one="equivalent" if symmetric else "left")
            rows = [(f"north district {i}", f"south district {i}", i % 2) for i in range(n)]
        names = [f"variable_{i}" for i in range(len(types))]
        manifest.update(modality=modality, target="answer", columns=[dict(name=k, type=t) for k, t in zip(names, types)])
        member, body = "inputs.csv", csv_bytes([*names, "answer"], rows)
    manifest["splits"] = {"train": dict(format=Path(member).suffix[1:], path=member)}
    make_zip(path, [("manifest.json", json.dumps(manifest).encode()), (member, body)])


class FirstFitReached(BaseException):
    """Exit before any fit without triggering an unrelated failure-recovery path."""


def before_fit(*args, **kwargs):
    raise FirstFitReached()


def inspect(packet_path, folder, weights):
    task = load_training_task(packet_path)
    goal = MetricSpec.for_task(task).record()
    args = ["unified_agent", "train", str(packet_path), "--run-dir", str(folder), "--strategy", "crossfit-challenger",
            "--weights", str(weights)]
    with patch.dict(os.environ), patch.object(sys, "argv", args), patch.object(ab_run, "Executor", side_effect=before_fit), \
            patch.object(training, "fit_one", side_effect=before_fit), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            unified_agent.main()
        except FirstFitReached:
            pass
        else:
            raise ValueError("Expected to reach a real execution boundary")
    if (folder / "challenger_protocol.json").exists():
        meta = read(folder / "challenger_protocol.json")
        route = "challenger"
        candidates = sorted(meta["inventory"]["candidates"])
        reasons = meta["inventory"]["excluded"]
        require(meta["rows"] == len(task.y_train) and meta["package"] == sha(packet_path), "Mismatched training input")
    else:
        native = read(folder / "agent/run_manifest.json")
        contract = native["task_contract"]
        require(contract["metric"] == goal, "Fallback scheduler changed the task metric or label ordering")
        route, candidates, reasons = "common", sorted(native["capabilities"]), {}
        if task.spec.get("input_schema", {}).get("pair_semantics") == "symmetric":
            require(all(c["swap_label_rule"] == "preserves" for c in native["capabilities"].values()),
                    "Symmetric input reached a directional candidate")
    require(candidates and not list(folder.rglob("fits.json")), "Missing methods or unexpected fitting")
    return dict(kind=task.kind, metric=goal, route=route, candidates=candidates, excluded=reasons)


def probe(weights):
    cases, variations = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for i, (key, kind, metric) in enumerate(CASES):
            a, b = base / f"packet_{i}.zip", base / f"renamed_{31-i}.zip"
            packet(a, key, metric)
            shutil.copyfile(a, b)
            row = inspect(a, base / f"first_{i}", weights)
            repeated = inspect(b, base / f"repeat_{i}", weights)
            require(row == repeated and row["kind"] == kind and row["metric"]["name"] == metric and
                    row["route"] == ("challenger" if key == "preference" else "common"), "Unexpected dispatch or name dependence")
            if key == "numbers":
                require(row["metric"]["labels"] == [1, 0], "Positive label zero was lost")
            cases[key] = dict(package=sha(a), same_on_rename=True, **row)
        reference = cases["preference"]["candidates"]
        for tag, omit in (("without_bge", BGE), ("without_reference", "e5-small-v2")):
            simulated = base / tag
            simulated.mkdir()
            for p in weights.iterdir():
                if p.name != omit and p.is_dir():
                    (simulated / p.name).symlink_to(p.resolve(), target_is_directory=True)
            row = inspect(a, base / f"run_{tag}", simulated)
            if tag == "without_bge":
                require(row["route"] == "challenger" and row["candidates"] == [n for n in reference if n != BGE]
                        and row["excluded"].get(BGE) == "not_installed", "Healthy challenger was lost")
            else:
                require(row["route"] == "common", "Missing reference failed to transfer execution")
            variations[tag] = row
        with patch.object(portfolio, "MODELS", dict(reversed(list(portfolio.MODELS.items())))):
            row = inspect(a, base / "run_reverse", weights)
        require(row["candidates"] == reference and row["route"] == "challenger", "Catalog enumeration changes eligibility")
        variations["reversed_inventory"] = row
    return dict(plan=sha(PLAN), cases=cases, variations=variations, calls_before_first_fit=15,
                classifier_fits=0, model_api_calls=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = probe(args.weights.resolve())
    save(args.output, result)
    print([(k, row["route"], len(row["candidates"])) for k, row in result["cases"].items()])
