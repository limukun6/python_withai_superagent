"""Run independent native pilot searches, reusing only individual fits of verified provenance."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from scripts.study_crossfit_challenger import call, native_predict
from scripts.unified_released_study import ROOT, read, require, save, sha, text_packet
from tools.pilot_challenger import STRATEGIES
from tools.pilot_fit_cache import FILES, SOURCES, build, recipe


PUBLIC_PRIOR = ROOT / "research/crossfit_challenger_1124"
PREVIOUS = "410ab95fd9f341f64752fda2f3a2d1175e6eaa42"
PLAN = ROOT / "docs/batches/pilot_nomination_plan_20261124_zh.md"


def link_file(source, dest):
    if dest.exists():
        require(sha(source) == sha(dest), "Different existing input")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)


def verify_previous_recipe():
    for name in SOURCES:
        original = subprocess.check_output(["git", "show", f"{PREVIOUS}:{name}"], cwd=ROOT)
        require(hashlib.sha256(original).hexdigest() == sha(ROOT / name), "Historical fit recipe changed: " + name)
    installed = recipe()["runtime"]
    for req in ("requirements-ab-encoder-tested.txt", "requirements-tested.txt"):
        for line in (ROOT / req).read_text().splitlines():
            if "==" in line:
                name, pin = line.split("==")
                if name in installed:
                    require(installed[name] == pin, "Historical environment differs: " + name)


def previous_components(prior, fold):
    if fold == 0 or prior is None:
        return []
    directory = prior / f"fold_{fold}"
    public = PUBLIC_PRIOR / directory.name
    selections = []
    for stage in ("run/inner_a", "run/refit", "comparison_refit"):
        for ref in sorted((public / stage).glob("*/trained.json")):
            local = directory / ref.parent.relative_to(public)
            require(all(sha(local / name) == sha(ref.parent / name) for name in FILES if name != "model.joblib")
                    and sha(local / "model.joblib") == read(ref)["model"], "Local history differs from published receipt")
            selections.append(local)
    return selections


def current_components(arm):
    directory = arm / "run"
    receipt = read(directory / "pilot_completion.json")
    sources = []
    for action in receipt["actions"]:
        path = directory / action["stage"] / action["candidate"]
        require(action["ledger"] == sha(path / "fits.json") and action["model"] == sha(path / "model.joblib"),
                "Cannot reuse an unbound classifier")
        sources.append(path)
    return sources


def prepare(out, prior):
    out.mkdir(parents=True, exist_ok=False)
    verify_previous_recipe()
    registered = read(PUBLIC_PRIOR / "protocol.json")
    require(sha(prior / "protocol.json") == sha(PUBLIC_PRIOR / "protocol.json"), "Prior plan differs")
    for i in range(5):
        key = f"fold_{i}"
        spec = registered["folds"][str(i)]
        for filename, digest in (("train.zip", spec["train_packet"]), ("features.zip", spec["features_packet"])):
            require(sha(prior / key / filename) == digest, "Anonymous packet differs")
            link_file(prior / key / filename, out / key / filename)
    save(out / "protocol.json", dict(plan=sha(PLAN), previous=sha(PUBLIC_PRIOR / "protocol.json"),
         folds=registered["folds"], cold_limit_per_strategy=9, new_fit_limit=30, model_api_limit=0,
         source_recipe=recipe()))


def run_arm(folder, strategy, sources, weights, features):
    arm = folder / strategy
    arm.mkdir(exist_ok=True)
    run = arm / "run"
    for f in ("train.zip", "features.zip"):
        link_file(folder / f, arm / f)
    package = sha(arm / "train.zip")
    if not (run / "pilot_completion.json").exists():
        require(not run.exists() and not (arm / "training.log").exists(), "Partial training path requires receipt inspection")
        args = ["train", arm / "train.zip", "--run-dir", run, "--strategy", strategy,
                "--weights", weights, "--feature-bank", features]
        if sources:
            cache = arm / "reuse"
            if not cache.exists():
                build(cache, package, sources)
            args.extend(("--fit-cache", cache))
        call(args, arm / "training.log")
    perf = read(run / "pilot_completion.json")
    spec = read(run / "pilot_protocol.json")
    require(spec["strategy"] == strategy and spec["package"] == package and
            perf["decision_sha256"] == sha(run / "decision.json") and
            perf["model_sha256"] == sha(run / "agent/selected_model.joblib") and
            perf["new_classifier_fits"] + perf["reused_classifier_fits"] == perf["cold_classifier_fits"] <= 9
            and perf["model_api_calls"] == 0, "Incomplete training or fit budget")
    return perf


def train_pair(folder, history, weights, cache):
    active = []
    for variant in STRATEGIES:
        perf = run_arm(folder, variant, active + history, weights, cache)
        print(folder.name, variant, perf["new_classifier_fits"], "new fits", flush=True)
        active.extend(current_components(folder / variant))
    first, second = (folder / s / "run" for s in STRATEGIES)
    require(read(first / "pilot_protocol.json")["roles"] == read(second / "pilot_protocol.json")["roles"],
            "Searches received different groups")
    with np.load(first / "feedback.npz") as a, np.load(second / "feedback.npz") as b:
        require(np.array_equal(a["learn_ids"], b["learn_ids"]) and np.allclose(a["learn"], b["learn"], atol=2e-5, rtol=0)
                and np.array_equal(a["learn"] > .5, b["learn"] > .5), "Pilot evidence differs between strategies")


def smoke(out, weights, cache):
    out.mkdir(parents=True, exist_ok=False)
    data = [(f"{'oak' if i%3 else 'pine'} district meeting {i}",
             f"residents discuss the {'first' if i%5 else 'last'} meeting {i}",
             int((i % 3 != 0) != (i % 7 == 0))) for i in range(216)]
    text_packet(out / "train.zip", data[:192], True)
    text_packet(out / "features.zip", [r[:2] for r in data[192:]], False)
    train_pair(out, [], weights, cache)
    rows = {}
    for strategy in STRATEGIES:
        arm = out / strategy
        with_cache, raw = native_predict(arm, weights, cache), native_predict(arm, weights)
        require(len(raw) == 24 and np.array_equal(with_cache, raw), "Cache and cold raw predictions differ")
        end = read(arm / "run/pilot_completion.json")
        rows[strategy] = dict(new=end["new_classifier_fits"], reused=end["reused_classifier_fits"],
                              model=end["model_sha256"], decision=end["decision_sha256"],
                              export=sha(arm / "direct/predictions.csv"), rows=len(raw))
    require(sum(r["new"] for r in rows.values()) <= 10, "Smoke budget exceeded")
    save(out / "smoke.json", dict(arms=rows, train=sha(out / "train.zip"), features=sha(out / "features.zip"),
                                    training_rows=192, model_api_calls=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("smoke", "prepare", "train"))
    p.add_argument("--out", required=True, type=lambda s: Path(s).resolve())
    for arg in ("prior", "weights", "cache", "smoke"):
        p.add_argument("--"+arg, type=lambda s: Path(s).resolve())
    a = p.parse_args()
    if a.stage == "smoke":
        smoke(a.out, a.weights, a.cache)
    elif a.stage == "prepare":
        prepare(a.out, a.prior)
    else:
        require(read(a.smoke / "smoke.json")["model_api_calls"] == 0 and
                read(a.out / "protocol.json")["plan"] == sha(PLAN), "Invalid preflight or plan")
        verify_previous_recipe()
        for i in range(5):
            train_pair(a.out / f"fold_{i}", previous_components(a.prior, i), a.weights, a.cache)
