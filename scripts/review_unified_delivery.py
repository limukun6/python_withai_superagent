"""Publish or independently recheck the two raw-package runs and the subsequent Task2 comparison."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np

from scripts.unified_confidence_study import paired_interval
from scripts.unified_released_study import EVIDENCE, ROOT, archive, predictions, read, require, sha, title_rows
from scripts.unified_third_member import assess
from scripts.verify_unified_raw import MEMBERS, TAGS, signatures
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PUBLIC = ROOT / "research/unified_raw_1124"


def publish(raw, confidence, third=None):
    selections = {name: raw / name for name in ("protocol.json", "results.json", "designated/predictions_frozen.json", "portable.json")}
    for tag in TAGS:
        for suffix in ("completion.json", "physical.jsonl"):
            selections[f"{tag}/{suffix}"] = raw / tag / suffix
        rel = f"designated/{tag}_0/predictions.csv"
        selections[rel] = raw / rel
    for name in ("freeze.json", "groups.json", "predictions.csv", "result.json"):
        selections["confidence/" + name] = confidence / name
    if third:
        for name in ("freeze.json", "selection.json", "predictions.csv", "result.json"):
            selections["third/" + name] = third / name
    for name, origin in selections.items():
        dest = PUBLIC / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        require(not dest.exists() or sha(dest) == sha(origin), "An existing research record differs")
        shutil.copyfile(origin, dest)
    files = sorted(p for p in PUBLIC.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (PUBLIC / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.relative_to(PUBLIC).as_posix() + "\n" for p in files))


def verify(packages):
    actual = {p.relative_to(PUBLIC).as_posix(): sha(p) for p in PUBLIC.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    expected = {line[66:]: line[:64] for line in (PUBLIC / "SHA256SUMS").read_text().splitlines()}
    require(expected == actual, "Public evidence changed")
    protocol = read(PUBLIC / "protocol.json")
    require(protocol["sources"] == signatures(), "Raw-entry sources differ")
    seals = read(PUBLIC / "designated/predictions_frozen.json")
    results = read(PUBLIC / "results.json")
    portable = read(PUBLIC / "portable.json")
    require(portable["entrypoint"] == protocol["sources"]["entrypoint"], "Portable entry changed")
    for tag in TAGS:
        record = read(PUBLIC / tag / "completion.json")
        v = results[tag]
        file = PUBLIC / f"designated/{tag}_0/predictions.csv"
        hard = predictions(file, v["rows"])
        historical = predictions(EVIDENCE / f"validation/reference_{tag}_0/predictions.csv", v["rows"])
        fits = [json.loads(line) for line in (PUBLIC / tag / "physical.jsonl").read_text().splitlines()]
        require(record["training"]["entrypoint_sha256"] == protocol["sources"]["entrypoint"] and
                record["model"] == v["model"] == portable["models"][tag]["model"] and
                sha(file) == v["prediction"] == portable["models"][tag]["prediction"] and seals[tag] == [sha(file)] * 2 and
                int((hard == historical).sum()) == v["identical_to_historical_rows"] and record["actions"][-1]["action"] == "stop"
                and set(record["source_members_opened"]) == set(MEMBERS[tag]) and record["api_calls"] == 0
                and 1 <= len(fits) <= protocol["fit_limit_per_run"] and fits == record["fits"], "Admission or model export differs")
        if packages:
            package = archive(packages, tag)
            require(sha(package) == protocol["archives"][tag], "Unexpected source archive")
            if tag == "table":
                with zipfile.ZipFile(package) as z:
                    truth = np.load(io.BytesIO(z.read("in_vehicle_coupon_recommendation/y_val.npy")), allow_pickle=False)
            else:
                truth = np.array([r[2] for r in title_rows(package, "val", labels=True)])
            require(len(truth) == v["rows"] and int((truth == hard).sum()) == v["correct"] and
                    v["accuracy"] == v["correct"] / len(truth), "Course validation count differs")
    title = read(PUBLIC / "titles/completion.json")
    ids = read(EVIDENCE / "title_reserve_ids.json")
    require(title["training"]["audit_row_ids_sha256"] == fingerprint(ids) and
            title["training"]["audit_rows"] + title["training"]["model_training_rows"] == title["training"]["source_training_rows"],
            "Raw entry did not regenerate the training panel")
    follow = read(PUBLIC / "confidence/result.json")
    frozen = read(PUBLIC / "confidence/freeze.json")
    require(sha(PUBLIC / "confidence/predictions.csv") == frozen["predictions"] and frozen["model"] == title["model"] and
            frozen["runner"] == sha(ROOT / "scripts/unified_confidence_study.py") and
            frozen["plan"] == sha(ROOT / "docs/batches/unified_confidence_protocol_20261124_zh.md") and
            not follow["designated_validation_used"] and not follow["official_test_used"] and
            follow["fits"] == follow["model_api_calls"] == 0, "Confidence experiment provenance differs")
    grouped = read(PUBLIC / "confidence/groups.json")
    sizes, deltas = np.array(grouped["rows"]), np.array(grouped["differences"])
    interval = paired_interval(sizes, deltas)
    require(int(sizes.sum()) == follow["rows"] and len(sizes) == follow["groups"] and
            int(deltas.sum()) == follow["gained"] - follow["lost"] == follow["candidate_correct"] - follow["control_correct"] and
            np.allclose(interval, follow["interval_fraction"]) and
            bool(deltas.sum() >= 5 and interval[0] > 0) == follow["passed"], "Grouped comparison differs")
    if packages:
        with (PUBLIC / "confidence/predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        training = title_rows(archive(packages, "titles"), "train", labels=True)
        truth = np.array([training[k][2] for k in ids])
        control = np.asarray([int(row["control"]) for row in rows]) == truth
        candidate = np.asarray([int(row["percentile"]) for row in rows]) == truth
        disagree = np.asarray([bool(int(row["disagree"])) for row in rows])
        require([int(row["row_id"]) for row in rows] == list(range(len(ids))) and
                int(control.sum()) == follow["control_correct"] == title["independent_training_audit"]["correct"] and
                int(candidate.sum()) == follow["candidate_correct"] and int((candidate & ~control).sum()) == follow["gained"] and
                int((control & ~candidate).sum()) == follow["lost"] and int(disagree.sum()) == follow["member_disagreements"] and
                int((~control & ~disagree).sum()) == follow["errors_both_members_share"] and
                int((~control & disagree).sum()) == follow["control_errors_on_disagreements"], "Training reserve counts differ")
    third = PUBLIC / "third"
    choice, frozen, comparison = (read(third / name) for name in ("selection.json", "freeze.json", "result.json"))
    candidates = choice["candidates"]
    best = max(candidates, key=lambda k: (candidates[k]["net"], candidates[k]["rescued_by_candidate_alone"], -candidates[k]["trial"]))
    independent = choice["independent"]
    gate = bool(candidates[best]["net"] > 0 and independent and all(r["net"] > 0 for r in independent.values())
                and comparison["matched"]["passes"] and comparison["delivered"]["passes"])
    require(comparison["refit_gate_passed"] == gate and best == comparison["chosen"] == choice["chosen"] and
            comparison["additional_fits"] == comparison["model_api_calls"] == 0 and not comparison["course_validation_used"] and
            not comparison["course_test_used"] and frozen["predictions"] == sha(third / "predictions.csv") and
            frozen["selection"] == sha(third / "selection.json") and choice["delivered"] == title["model"] and
            choice["study"] == sha(ROOT / "scripts/unified_third_member.py") and
            choice["plan"] == sha(ROOT / "docs/batches/unified_third_member_protocol_20261124_zh.md"), "Third-member provenance differs")
    if packages:
        with (third / "predictions.csv").open() as stream:
            observations = list(csv.DictReader(stream))
        require([int(r["row_id"]) for r in observations] == list(range(len(ids))), "Third-member row order changed")
        groups = sample_groups(SimpleNamespace(kind="pair_text", X_train=[training[i][:2] for i in ids]), "shared_text")
        _, group_numbers = np.unique(groups, return_inverse=True)
        hard = {k: np.asarray([int(r[k]) for r in observations]) for k in observations[0]}
        for label in ("matched", "delivered"):
            require(assess(truth, (hard[label + "_pair"], hard[label + "_triple"]), group_numbers) == comparison[label],
                    "Third-member comparison failed to reproduce")
        common_errors = (hard["first_delivered"] != truth) & (hard["second_delivered"] != truth)
        require(int(common_errors.sum()) == comparison["delivered_common_errors"] == follow["errors_both_members_share"] and
                int((common_errors & (hard["third"] == truth)).sum()) == comparison["corrected_by_third_alone"] and
                int((common_errors & (hard["delivered_triple"] == truth)).sum()) == comparison["corrected_by_actual_triple"],
                "Third-member common-error counts differ")
    print("Verified raw-package grades, portable predictions and both paired Task2 experiments.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", type=Path)
    p.add_argument("--publish", nargs=2, type=Path, metavar=("RAW_RUN", "CONFIDENCE_RUN"))
    p.add_argument("--third-run", type=Path)
    args = p.parse_args()
    if args.publish:
        publish(*args.publish, third=args.third_run)
    verify(args.packages)
