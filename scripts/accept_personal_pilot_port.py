"""Exercise the personal branch's ordinary training and prediction packages on anonymous inputs."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

from scripts.run_conditional_handoff import load_export
from scripts.unified_released_study import ROOT, read, require, save, sha, text_packet
from scripts.verify_brier_replication import ORIGINAL, PUBLIC as COLD
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from scripts.verify_pilot_backup import PUBLIC as BACKUP

PIN = "6ca66a5e57222a841a0b68392ab97854c66f70ae"
PLAN = ROOT / "docs/batches/personal_pilot_port_plan_20261124_zh.md"
PUBLIC = ROOT / "research/pilot_personal_port_1124"
BGE = BACKUP / "fold_0/pilot-backup"


def prepare(packet, dest):
    require(read(BACKUP / "protocol.json")["parent"] == sha(ORIGINAL / "frozen.json") and
            read(ORIGINAL / "frozen.json")["protocol"] == sha(ORIGINAL / "protocol.json"), "Unexpected backup parent")
    expected = read(ORIGINAL / "protocol.json")["folds"]["0"]
    require(sha(packet) == expected["features_packet"], "Anonymous feature panel differs")
    with zipfile.ZipFile(packet) as z:
        with z.open("rows.csv") as data:
            reader = csv.reader(io.TextIOWrapper(data, encoding="utf-8"))
            require(next(reader) == ["option_a", "option_b"], "Only a feature-only anonymous pair packet is allowed")
            original = [tuple(row) for row in reader]
    require(len(original) == 1316 and all(len(pair) == 2 for pair in original), "Unregistered panel")
    all_rows = original + [(b, a) for a, b in original] + [(a, a) for a, _ in original[:8]]
    text_packet(dest, all_rows, False)
    return np.asarray([a != b for a, b in original])


def check_rows(dest, mask):
    first = load_export(BGE / "direct/predictions.csv")["prediction"]
    results = {}
    for arm in ("original", "bounded"):
        path = dest / arm / "predictions.csv"
        rows = load_export(path)
        y = rows["prediction"]
        n = len(first)
        require(len(y) == 2 * n + 8 and np.array_equal(rows["row_id"], np.arange(len(y))), "Wrong panel row order")
        results[arm] = dict(rows=len(y), export=sha(path), original_differences=int(np.sum(y[:n] != first)),
                            swap_failures=int(np.sum(y[n:2*n][mask] != 1 - y[:n][mask])),
                            nonzero_self_pairs=int(np.sum(y[-8:] != 0)))
    a, b = (load_export(dest / arm / "predictions.csv")["prediction"] for arm in ("original", "bounded"))
    diffs = int(np.sum(a != b))
    passes = diffs == 0 and all(r["original_differences"] == r["swap_failures"] == r["nonzero_self_pairs"] == 0
                                for r in results.values())
    return dict(arms=results, differences=diffs, nonidentical_original_rows=int(mask.sum()), passes=passes)


def run(out, personal, source, smoke, weights):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=personal, text=True).strip()
    require(head == PIN and not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=personal),
            "Personal branch runtime differs from registered sources")
    out.mkdir(parents=True, exist_ok=False)
    finished = read(smoke / "pilot_completion.json")
    previous = COLD / "smoke/pilot-brier"
    require(sha(smoke / "decision.json") == sha(previous / "run/decision.json")
            and sha(smoke / "nomination.json") == sha(previous / "run/nomination.json")
            and sha(smoke / "direct/predictions.csv") == sha(previous / "direct/predictions.csv")
            and finished["new_classifier_fits"] == finished["model_api_calls"] == 0, "The anonymous training port differs")
    for f in ("nomination.json", "decision.json", "pilot_completion.json", "direct/predictions.csv"):
        copy(smoke / f, out / "smoke" / f)
    reference = source / "pilot-backup/run"
    selection = read(BGE / "run/agent/final_selection.json")
    require(sha(reference / "agent/selected_model.joblib") == selection["model_sha256"]
            and selection["ensemble_components"][-1] == "bge-small-en-v1.5", "Missing frozen BGE selection")
    features = out / "features.zip"
    mask = prepare(source / "features.zip", features)
    environment = dict(os.environ, AB_B1_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       VECLIB_MAXIMUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONPATH=os.pathsep.join(map(str, (personal / "scripts/ab_acceptance_guard", personal))))
    environment.pop("SUPERAGENT_PORTFOLIO_CACHE", None)
    def call(args, log):
        with log.open("x") as stream:
            subprocess.run([sys.executable, "-m", *map(str, args)], cwd=personal, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    optimized = out / "optimized"
    call(["scripts.bounded_preference_delivery", reference, optimized], out / "package.log")
    for arm, path in (("original", reference), ("bounded", optimized)):
        folder = out / arm
        folder.mkdir()
        call(["unified_agent", "predict", features, "--run-dir", path, "--weights", weights,
              "--output", folder / "predictions.csv"], folder / "native.log")
    end = check_rows(out, mask)
    save(out / "result.json", dict(personal_source_commit=PIN, plan=sha(PLAN), features=sha(features),
              research_model=selection["model_sha256"], optimized_model=read(optimized / "agent/final_selection.json")["model_sha256"],
              smoke=dict(new_fits=0, reused=finished["reused_classifier_fits"], rows=24,
                         export=sha(previous / "direct/predictions.csv")), performance=end, model_api_calls=0, fits=0))
    print("BGE package check:", end)


def publish(out):
    for f in ("result.json", "smoke/nomination.json", "smoke/decision.json", "smoke/pilot_completion.json",
              "smoke/direct/predictions.csv", "original/predictions.csv", "bounded/predictions.csv",
              "optimized/agent/final_selection.json"):
        copy(out / f, PUBLIC / f)
    seal_directory(PUBLIC)


def verify(original_packet):
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text(), "Changed personal evidence")
    result = read(PUBLIC / "result.json")
    with tempfile.TemporaryDirectory() as tmp:
        panel = Path(tmp) / "features.zip"
        mask = prepare(original_packet, panel)
        require(sha(panel) == result["features"] and result["performance"] == check_rows(PUBLIC, mask), "BGE replay differs")
    prior = COLD / "smoke/pilot-brier"
    smoke = PUBLIC / "smoke"
    selection = read(PUBLIC / "optimized/agent/final_selection.json")
    require(result["plan"] == sha(PLAN) and result["personal_source_commit"] == PIN and
            result["optimized_model"] == selection["model_sha256"] and result["research_model"] == selection["original_model_sha256"]
            and all(sha(smoke / f) == sha(prior / "run" / f) for f in ("nomination.json", "decision.json")) and
            result["smoke"]["export"] == sha(smoke / "direct/predictions.csv") == sha(prior / "direct/predictions.csv"),
            "Personal package provenance differs")
    print("Verified the personal anonymous training path and both BGE exports including swapped and identical pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    run_cmd = stages.add_parser("run")
    for key in ("out", "personal", "source", "smoke", "weights"):
        run_cmd.add_argument(key, type=lambda x: Path(x).resolve())
    stages.add_parser("publish").add_argument("out", type=Path)
    stages.add_parser("verify").add_argument("original_packet", type=Path)
    args = parser.parse_args()
    if args.stage == "run":
        run(args.out, args.personal, args.source, args.smoke, args.weights)
    elif args.stage == "publish":
        publish(args.out)
    else:
        verify(args.original_packet)
