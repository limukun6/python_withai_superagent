"""Offline engineering audit of classifiers added after the Agent was packaged."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from superagent.ab_run import sha, sources


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts/provider_fixture.py"
PLAN = ROOT / "docs/batches/B_runtime_provider_20261124_zh.md"
ALIAS = "course-method-demo"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def install_example(directory, version):
    directory.mkdir()
    shutil.copyfile(FIXTURE, directory / "independent_methods.py")
    info = directory / f"course_method_demo-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {ALIAS}\nVersion: {version}\n")
    (info / "entry_points.txt").write_text(
        "[superagent.classifiers.v1]\n"
        "median = independent_methods:MedianPrototypeClassifier\n"
        "threshold = independent_methods:BinaryThresholdClassifier\n"
        "missing = nonexistent_method_fixture:MissingClassifier\n")


def inspect_contracts(agent, out):
    from superagent.open_methods import propose
    from superagent.runtime_inventory import installed_extensions

    extensions, rejected = installed_extensions()
    own = {key: cls.__name__ for key, cls in extensions.items() if key.startswith(ALIAS + "@")}
    require(len(own) == 2, "Wrong number of compatible fixture classifiers")
    own_errors = {key: reason for key, reason in rejected.items() if key.startswith(ALIAS + "@")}
    require(len(own_errors) == 1, "Broken optional entry was not isolated")
    manifest = read(agent / "run_manifest.json")
    profile = read(agent / "training_profile.json")
    result = {}
    for case, labels, metric in (("binary", [0, 1], "accuracy"), ("multiclass", [0, 1, 2], "accuracy"),
                                 ("probabilities", [0, 1], "roc_auc")):
        task = copy.deepcopy(manifest["task_contract"])
        task["metric"].update(labels=labels, name=metric)
        proposals, evidence = propose(task, profile, [])
        visible = sorted(own[spec["class"]] for spec in proposals.values() if spec["class"] in own)
        expected = sorted(own.values()) if case == "binary" else ["MedianPrototypeClassifier"]
        require(visible == expected, "Incorrect task-compatibility decision for " + case)
        result[case] = dict(visible=visible, excluded={own[k]: v for k,v in evidence["excluded"].items() if k in own})
    write(out, dict(installed=own, invalid=own_errors, contracts=result))


def bundle(output, evidence, spec, name):
    import joblib
    model = joblib.load(evidence)["model"]
    require(model.provider_requirement_.key.startswith(ALIAS + "@1.0/median="), "Provider pin missing from the fitted model")
    selected = output / "selected_model.joblib"
    joblib.dump(dict(kind="numeric_tabular", candidate=name, model=model), selected)
    shutil.copyfile(spec, output / "task_spec.json")
    write(output / "final_selection.json", dict(selected_experiment="explicit_interface_fixture", family=name,
          model_file=selected.name, model_format="candidate_bundle_v1", test_accessed=False,
          model_sha256=sha(selected), task_spec_sha256=sha(spec)))


def run(out, inputs):
    out.mkdir(parents=True, exist_ok=False)
    train, features = inputs / "train.zip", inputs / "features.zip"
    write(out / "protocol.json", dict(source=sources(), plan=sha(PLAN), runner=sha(Path(__file__)),
          fixture=sha(FIXTURE), train=sha(train), features=sha(features), native_fit_limit=5, transport_fit_limit=1,
          seed=60281, real_api_limit=0))
    for folder, version in (("installed", "1.0"), ("upgraded", "2.0")):
        install_example(out / folder, version)

    def command(args, log, *, version="installed", physical="physical.jsonl", reject=False):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (ROOT / "scripts/ab_acceptance_guard", ROOT, out / version))),
                   AB_B1_OFFLINE="1", AB_B1_FIT_LOG=str(out / physical), AB_B1_PER_WORKER_FIT_LIMIT="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
        with (out / log).open("x") as stream:
            child = subprocess.run([sys.executable, "-m", *map(str,args)], env=env, cwd=ROOT, timeout=660,
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        if reject:
            require(child.returncode != 0 and "Required installed classifier version or entry point" in (out/log).read_text(),
                    "A different provider version loaded, or failed for an unrelated reason")
        else:
            require(child.returncode == 0, f"{log} failed ({child.returncode})")

    agent = out / "agent"
    command(["superagent", "run", train, "--scheduler", "ab-v1", "--planner", "local", "--portfolio", "ab-v1",
             "--method-space", "runtime", "--task-profile", "train", "--fit-limit", "5", "--seed", "60281",
             "--time-limit", "600", "--trial-timeout", "180", "--run-dir", agent], "native.log")
    outcome = read(agent / "run_outcome.json")
    require(outcome["status"] == "completed" and outcome["full_refit_completed"] and outcome["source_unchanged"]
            and outcome["new_api_requests"] == 0, "Incomplete autonomous training")
    charges = read(agent / "classifier_fits.json")
    actual = [json.loads(s) for s in (out / "physical.jsonl").read_text().splitlines()]
    require(1 <= len(actual) == charges["attempted_classifier_fits"] == charges["completed_classifier_fits"] <= 5,
            "Native and independent fit counts differ")
    command(["scripts.check_b_installed_methods", "inspect", agent, out/"contracts.json"], "contract.log")
    seen = read(out / "contracts.json")["installed"]
    start = read(agent / "runtime_registry_0.json")
    require(set(seen) <= {s["class"] for s in start["entries"].values()}, "Native Agent did not discover both extensions")
    registry = read(agent / "candidate_registry.json")
    installed_ids = {n for n,s in registry["entries"].items() if s["class"] in seen}
    actions = read(agent / "actions.json")
    choices = []
    for index, item in enumerate(actions, 1):
        if item["action"] == "explore":
            ranked = read(agent / f"method_choice_{index}.json")
            choices.append(dict(chosen=ranked["chosen"], candidates=len(ranked["candidates"]),
                                installed_eligible=[v["candidate"] for v in ranked["candidates"] if v["candidate"] in installed_ids]))
    require(any(c["installed_eligible"] for c in choices), "Extension absent from actual scheduler choices")

    def exports(directory, prefix):
        hashes = []
        for i in range(2):
            dest = out / f"{prefix}_{i}" / "rows.csv"
            command(["superagent", "predict", features, "--run-dir", directory, "--output", dest], f"{prefix}_{i}.log")
            meta = read(dest.parent / "prediction_manifest.json")
            with dest.open() as f:
                ids = [int(r["row_id"]) for r in csv.DictReader(f)]
            require(ids == list(range(33)) and meta["prediction_count"] == 33 and not meta["test_labels_read"], "Wrong output rows")
            hashes.append(sha(dest))
        require(hashes[0] == hashes[1], "Cross-process prediction differs")
        return hashes

    autonomous_exports = exports(agent, "native_export")
    transport = out / "transport"
    transport.mkdir()
    audit_candidate = next(n for n,s in start["entries"].items() if seen.get(s["class"]) == "MedianPrototypeClassifier")
    write(transport / "registry.json", start)
    fitted, ledger = transport / "refit.joblib", transport / "fits.json"
    admit = transport / "admission.json"
    write(admit, dict(maximum_classifier_fits=1, ledger_path=str(ledger)))
    command(["superagent.experiment_worker", agent/"refit_data.joblib", audit_candidate, fitted, transport/"registry.json",
             "--fit-admission", admit], "interface_fit.log", physical="interface_physical.jsonl")
    physical = [json.loads(s) for s in (out/"interface_physical.jsonl").read_text().splitlines()]
    require(len(physical) == read(ledger)["completed_classifier_fits"] == 1 and physical[0]["candidate"] == audit_candidate,
            "Interface fixture cost changed")
    command(["scripts.check_b_installed_methods", "bundle", transport, fitted, agent/"task_spec.json", audit_candidate], "bundle.log")
    interface_exports = exports(transport, "interface_export")
    invalid = out / "invalid" / "rows.csv"
    command(["superagent", "predict", features, "--run-dir", transport, "--output", invalid],
            "version_mismatch.log", version="upgraded", reject=True)
    require(not invalid.exists(), "Wrote predictions using a mismatched version")
    final = read(agent / "final_selection.json")["family"]
    result = dict(protocol=sha(out/"protocol.json"), contracts=sha(out/"contracts.json"), initial_installed=len(seen),
                  choices=choices, native_fit_count=len(actual), native_fits_on_installed=sum(v["candidate"] in installed_ids for v in actual),
                  native_winner=final, winner_installed=final in installed_ids, native_exports=autonomous_exports, exported_rows=33,
                  explicit_interface_candidate=audit_candidate, explicit_interface_fits=len(physical), interface_exports=interface_exports,
                  version_mismatch_rejected=True, new_model_api_calls=0)
    write(out/"result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for mode, args in (("run", ("out", "inputs")), ("inspect", ("agent", "out")), ("bundle", ("output", "fitted", "spec", "name"))):
        sub = commands.add_parser(mode)
        for arg in args:
            sub.add_argument(arg, type=str if arg=="name" else lambda v: Path(v).resolve())
    a = parser.parse_args()
    if a.operation == "run":
        run(a.out, a.inputs)
    elif a.operation == "inspect":
        inspect_contracts(a.agent, a.out)
    else:
        bundle(a.output, a.fitted, a.spec, a.name)
