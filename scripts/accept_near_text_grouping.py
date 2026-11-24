"""Verify the optional production grouping on the training corpus and a real anonymous benchmark."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.audit_pilot_title_overlap import PUBLIC as AUDIT
from scripts.study_crossfit_challenger import study_splits
from scripts.unified_released_study import ROOT, SCHEMA, make_zip, read, require, save, sha
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from superagent.data import TrainingTask, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition, sample_groups

PLAN = ROOT / "docs/batches/near_text_grouping_plan_20261124_zh.md"
PUBLIC = ROOT / "research/near_text_grouping_1124"
SEEDS = range(41101, 41106)


def course(archive):
    original, _, splits = study_splits(archive)
    main = np.sort(np.concatenate([held for _, held in splits]))
    task = TrainingTask(original.kind, [original.X_train[i] for i in main], original.y_train[main], original.spec)
    labels = sample_groups(task, "near_text")
    require(checksums(AUDIT) == (AUDIT / "SHA256SUMS").read_text(), "Independent group audit changed")
    with (AUDIT / "flags.csv").open(encoding="utf-8") as stream:
        expected = list(csv.DictReader(stream))
    require(np.array_equal(main, [int(row["row_id"]) for row in expected])
            and labels.tolist() == [r["expanded_group"] for r in expected]
            and len(set(labels)) == 4737, "Optimized groups differ from the independent exhaustive index")
    receipts = []
    for seed in SEEDS:
        fit, held, info = grouped_partition(task, seed, "near_text")
        shared = set(labels[fit]) & set(labels[held])
        require(not shared and info["total_groups"] == len(set(labels)), "Expanded group crosses the split")
        receipts.append(dict(seed=seed, train_rows=len(fit), eval_rows=len(held), shared_groups=len(shared),
                             train_ids=fingerprint(main[fit].tolist()), eval_ids=fingerprint(main[held].tolist()),
                             train_groups=info["train_groups"], eval_groups=info["heldout_groups"]))
    return dict(plan=sha(PLAN), independent_audit=sha(AUDIT / "frozen.json"), rows=len(labels), groups=len(set(labels)),
                ids=fingerprint(labels.tolist()), splits=receipts, classifier_fits=0, model_api_calls=0)


def fixture(path):
    def examples(lower, upper):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["option_a", "option_b", "outcome"])
        for i in range(lower, upper):
            root = " ".join(f"topic{i}x{j}" for j in range(11))
            for variant in range(2):
                signal = (i * 7 + variant) % 5 < 3
                a = root + (" updates" if variant else "")
                b = f"{'early' if signal else 'late'} briefing {i} update {variant}"
                writer.writerow([a, b, int(signal != (i % 11 == 0))])
        return buffer.getvalue().encode()
    splits = {s: dict(format="csv", path=s + ".csv") for s in ("train", "validation")}
    manifest = dict(schema_version=1, objective="classification", metric="accuracy", splits=splits, **SCHEMA)
    make_zip(path, [("manifest.json", json.dumps(manifest).encode()),
                    ("train.csv", examples(0, 120)), ("validation.csv", examples(120, 150))])


def inspect_fixture(packet, splits):
    task = load_training_task(packet)
    codes = sample_groups(task, "near_text")
    require(len(set(codes)) == 120 and len(set(sample_groups(task, "shared_text"))) == 240,
            "Fixture does not distinguish the two grouping methods")
    summaries = []
    for i, record in enumerate(splits):
        fit, held, info = grouped_partition(task, 41101 + 10000 * i, "near_text")
        require(record["seed"] == 41101 + 10000 * i and record["train_indices"] == fit.tolist()
                and record["search_indices"] == held.tolist() and not set(codes[fit]) & set(codes[held])
                and info["total_groups"] == 120, "Benchmark split differs from the declared method")
        summaries.append(dict(seed=record["seed"], held_rows=len(held), held_ids=fingerprint(held.tolist()),
                              fit_ids=fingerprint(fit.tolist()), shared_groups=0))
    return summaries


def smoke(out):
    out.mkdir(parents=True, exist_ok=False)
    packet = out / "anonymous.zip"
    fixture(packet)
    physical = (out / "physical.jsonl").resolve()
    env = dict(os.environ, AB_B1_OFFLINE="1", AB_B1_FIT_LOG=str(physical), AB_B1_PER_WORKER_FIT_LIMIT="1",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT / "scripts/ab_acceptance_guard", ROOT))))
    command = [sys.executable, "-m", "superagent.benchmark", str(packet.resolve()), "--strategies", "random",
               "--seeds", "41101", "--max-trials", "1", "--confirmation-repeats", "3", "--confirmation-top-k", "1",
               "--search-grouping", "near_text", "--confirmation-grouping", "near_text", "--batch-budget-rmb", "0",
               "--output-dir", str((out / "benchmark").resolve())]
    with (out / "benchmark.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summaries = list((out / "benchmark").glob("*/*/summary.json"))
    require(len(summaries) == 1, "Unexpected number of benchmark runs")
    run = summaries[0].parent
    counts = read(run / "summary.json")
    inputs = [read(run / name) for name in ("split.json", "confirmation_split_1.json", "confirmation_split_2.json")]
    measured = [json.loads(line) for line in physical.read_text().splitlines()]
    require(counts["valid_policy_run"] and counts["classifier_fit_accounting"]["complete_count_known"]
            and counts["classifier_fit_accounting"]["completed_classifier_fits"] == len(measured) == 3
            and counts["tokens"] == 0, "Incomplete physical classifier or model-call accounting")
    for i, row in enumerate(inputs):
        save(out / f"split_{i}.json", {k: row[k] for k in ("seed", "train_indices", "search_indices")})
    evidence = dict(package=sha(packet), plan=sha(PLAN), protocol=sha(run.parent / "protocol.json"),
                    folds=inspect_fixture(packet, inputs), training_rows=240, synthetic_validation_rows=60,
                    actual_classifier_fits=len(measured), classifier_events=measured, model_api_calls=0, tokens=counts["tokens"])
    save(out / "smoke.json", evidence)
    print("Benchmark completed:", measured)


def publish(out, archive):
    for name in ("smoke.json", "split_0.json", "split_1.json", "split_2.json", "physical.jsonl"):
        copy(out / name, PUBLIC / name)
    paths = list((out / "benchmark").glob("*/*/summary.json"))
    require(len(paths) == 1, "Ambiguous benchmark receipt")
    run = read(paths[0])
    protocol = paths[0].parent.parent / "protocol.json"
    require(sha(protocol) == read(out / "smoke.json")["protocol"], "Different benchmark launch")
    launch = read(protocol)
    config = {k: launch[k] for k in ("strategies", "seeds", "max_trials", "confirmation_repeats", "confirmation_top_k",
                                      "search_grouping", "confirmation_grouping", "batch_budget_rmb", "source_sha256")}
    receipt = {k: run[k] for k in ("strategy", "seed", "selected_candidate", "valid_policy_run", "llm_calls",
                                  "tokens", "course_cost_rmb", "test_accessed", "classifier_fit_accounting")}
    measurements = run["confirmation"]["candidates"][0]["measurements"]
    receipt["measurements"] = [{k: m[k] for k in ("seed", "status", "new_fit")} for m in measurements]
    save(PUBLIC / "benchmark.json", dict(launch=config, run=receipt, local_summary_hash=sha(paths[0])))
    save(PUBLIC / "course.json", course(archive))
    seal_directory(PUBLIC)


def verify(archive):
    import tempfile
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text() and
            read(PUBLIC / "course.json") == course(archive), "Changed grouping acceptance")
    smoke_record = read(PUBLIC / "smoke.json")
    with tempfile.TemporaryDirectory() as tmp:
        packet = Path(tmp) / "fixture.zip"
        fixture(packet)
        records = [read(PUBLIC / f"split_{i}.json") for i in range(3)]
        require(sha(packet) == smoke_record["package"] and inspect_fixture(packet, records) == smoke_record["folds"],
                "Fixture rows or grouping changed")
    events = [json.loads(line) for line in (PUBLIC / "physical.jsonl").read_text().splitlines()]
    benchmark = read(PUBLIC / "benchmark.json")
    run, launch = benchmark["run"], benchmark["launch"]
    require(launch["search_grouping"] == launch["confirmation_grouping"] == "near_text" and
            run["valid_policy_run"] and not run["test_accessed"] and run["llm_calls"] == run["course_cost_rmb"] == 0 and
            [(m["seed"], m["status"], m["new_fit"]) for m in run["measurements"]] ==
            [(41101, "completed", False), (51101, "completed", True), (61101, "completed", True)]
            and run["classifier_fit_accounting"]["completed_classifier_fits"] == len(events)
            and all(e["candidate"] == run["selected_candidate"] for e in events), "Benchmark control flow differs")
    require(events == smoke_record["classifier_events"] and len(events) == smoke_record["actual_classifier_fits"] == 3
            and smoke_record["model_api_calls"] == smoke_record["tokens"] == 0, "Incorrect engineering cost")
    print("Verified 6,608 row assignments, five stricter development splits and three real fixture fits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    sm = stages.add_parser("smoke")
    pub = stages.add_parser("publish")
    ver = stages.add_parser("verify")
    for cmd in (sm, pub):
        cmd.add_argument("out", type=Path)
    for cmd in (pub, ver):
        cmd.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke(args.out)
    elif args.stage == "publish":
        publish(args.out, args.archive)
    else:
        verify(args.archive)
