"""Replay the locked rule and each raw export, and align both partitions on original row IDs."""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np

from scripts.evaluate_pilot_nomination import project
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_brier_replication import RULE, partitions, prepare, score, signature
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from scripts.verify_pilot_backup import stage_score
from scripts.verify_pilot_nomination import PUBLIC as ORIGINAL, receipts, verify_one_receipt
from scripts.verify_unified_raw import MEMBERS
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import rows_at
from tools.pilot_challenger import approve, nominate, panel_scores, pilot_roles

PUBLIC = ROOT / "research/pilot_brier_replication_1124"


def copy_trial(src, destination):
    for file in ("sealed.json", *read(src / "sealed.json")["records"]):
        copy(src / file, destination / file)
    save(destination / "fit_receipts.json", receipts(src, (RULE,)))


def trials_on_original_rows(directory, splits, key, size):
    output = np.full((size, 2), -1, dtype=int)
    for i, (_, indices) in enumerate(splits):
        rows = load_export(directory / f"fold_{i}/rows.csv")
        require(np.array_equal(rows["row_id"], np.arange(len(indices))) and np.all(output[indices] == -1),
                "Overlapping prediction rows")
        output[indices] = np.column_stack([rows["baseline"], rows[key]])
    return output


def sensitivity(archive, current):
    task, groups, splits, old = partitions(archive)
    y = np.asarray(task.y_train)
    first = trials_on_original_rows(ORIGINAL, old, RULE + "_approved", len(y))
    second = trials_on_original_rows(current, splits, "agent", len(y))
    main = np.flatnonzero(first[:, 0] != -1)
    require(np.array_equal(main, np.flatnonzero(second[:, 0] != -1)) and len(main) == 6608,
            "Partitions do not cover identical source examples")
    truth = y[main]
    delta = []
    for matrix in (first, second):
        base, final = matrix[main].T
        delta.append((final == truth).astype(int) - (base == truth).astype(int))
    cells = [[int(np.sum((delta[0] == a) & (delta[1] == b))) for b in (-1, 0, 1)] for a in (-1, 0, 1)]
    return dict(rows=len(main), groups=len(set(groups[main])), row_ids=fingerprint(main.tolist()),
                old_result=sha(ORIGINAL / "result.json"), current_result=sha(current / "result.json"),
                first_change_by_second_change=cells, order=["lost", "unchanged", "gained"],
                agent_across_splits=comparison(first[main, 1], second[main, 1], truth, groups[main]),
                baseline_across_splits=comparison(first[main, 0], second[main, 0], truth, groups[main]))


def publish(out, smoke, archive):
    for name in ("protocol.json", "frozen.json", "result.json"):
        copy(out / name, PUBLIC / name)
    copy(smoke / "smoke.json", PUBLIC / "smoke/smoke.json")
    for name in ("run/nomination.json", "run/decision.json", "run/pilot_completion.json", "direct/predictions.csv"):
        copy(smoke / RULE / name, PUBLIC / "smoke" / RULE / name)
    for i in range(5):
        copy_trial(out / f"fold_{i}", PUBLIC / f"fold_{i}")
    save(PUBLIC / "partition_sensitivity.json", sensitivity(archive, PUBLIC))
    seal_directory(PUBLIC)


def verify_trial(folder, packet, outer):
    task = load_training_task(packet)
    package = sha(packet)
    groups = sample_groups(task, "shared_text")
    roles = pilot_roles(groups, task.y_train)
    describe = {k: dict(rows=len(ix), groups=len(set(groups[ix])), ids=fingerprint(ix.tolist()))
                for k, ix in roles.items()}
    require(sum(item["groups"] for item in describe.values()) == len(set(groups)), "Overlapping feedback groups")
    run = folder / RULE / "run"
    meta, nom, decision, end, selection = (read(run / name) for name in
        ("pilot_protocol.json", "nomination.json", "decision.json", "pilot_completion.json", "agent/final_selection.json"))
    names = meta["inventory"]["candidates"]
    require(meta["strategy"] == nom["strategy"] == end["strategy"] == RULE and meta["package"] == package
            and meta["roles"] == describe and nom["members"] == names and nom["learning_rows"] == describe["learn"]["ids"],
            "Wrong input, groups or nomination rule")
    learn = roles["learn"]
    with np.load(run / "feedback.npz", allow_pickle=False) as fb:
        require(np.array_equal(fb["learn_ids"], learn), "Learning rows differ")
        evidence = panel_scores(names, fb["learn"], rows_at(task, learn), task.y_train[learn], groups[learn])
        nominee = nominate(evidence, RULE)
        require(nom["chosen"] == nominee and nom["learning"] == evidence, "Nomination cannot be reproduced")
        check = stage_score(fb, "confirm", roles["confirm"], nominee, task, groups)
    choice = approve(nominee, evidence, check, len(roles["confirm"]))
    choice.update(nomination=sha(run / "nomination.json"), feedback=sha(run / "feedback.npz"))
    require(decision == choice and end["decision_sha256"] == sha(run / "decision.json"), "Approval differs")
    selected = decision["approved"]
    members = names[:2] + ([selected] if selected else [])
    seal = read(folder / "sealed.json")
    base = read(folder / "baseline/run/agent/final_selection.json")
    require(members == seal["members"] == selection["ensemble_components"] and base["ensemble_components"] == names[:2]
            and selection["model_sha256"] == end["model_sha256"] and not selection["test_accessed"]
            and not base["test_accessed"], "Delivered model differs")
    actions = [("pilot", n) for n in names] + [("refit", n) for n in members]
    records = read(folder / "fit_receipts.json")
    require(actions == [(a["stage"], a["candidate"]) for a in end["actions"]]
            and {f"{RULE}/{s}/{n}" for s, n in actions} == set(records), "Unaccounted model fit")
    require(end["new_classifier_fits"] == end["cold_classifier_fits"] == seal["fits"] == len(actions) <= 9
            and end["reused_classifier_fits"] == end["model_api_calls"] == 0, "Incorrect cold budget")
    for action in end["actions"]:
        stage, name = action["stage"], action["candidate"]
        ids = describe["pilot"]["ids"] if stage == "pilot" else fingerprint(list(range(len(groups))))
        identity = dict(ids=ids, package=package)
        record = verify_one_receipt(records[f"{RULE}/{stage}/{name}"], identity, name)
        require(action["identity"] == identity and action["model"] == record["model"] and
                action["ledger"] == record["ledger"] and not action["reused"], "Fit or identity differs")
        if stage == "refit":
            require(seal["refit_models"][name] == record["model"], "Refitted head differs")
    with np.load(folder / "scores.npz", allow_pickle=False) as scores:
        p = scores["scores"]
        require(p.shape == (len(outer), len(members)) and np.isfinite(p).all()
                and ((p >= 0) & (p <= 1)).all(), "Missing or invalid outer probabilities")
        projected = project(members, p, outer)
    row_ids = np.arange(len(outer))
    rows = load_export(folder / "rows.csv")
    require(np.array_equal(rows["row_id"], row_ids), "Outer row order changed")
    for key, member, arm in (("baseline", "baseline", "baseline"), ("agent", selected or "baseline", RULE)):
        native = load_export(folder / arm / "direct/predictions.csv")
        require(np.array_equal(native["row_id"], row_ids) and np.array_equal(native["prediction"], rows[key])
                and np.array_equal(rows[key], projected[member]), "Native export differs")


def verify(archive):
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text()
            and checksums(ORIGINAL) == (ORIGINAL / "SHA256SUMS").read_text(), "Evidence checksum changed")
    seen = set()
    open_member = zipfile.ZipFile.open
    def guarded(z, member, *args, **kwargs):
        if isinstance(z.filename, (str, Path)) and Path(z.filename).resolve() == archive.resolve():
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            require(name in MEMBERS["titles"], "Unexpected course archive member")
            seen.add(name)
        return open_member(z, member, *args, **kwargs)
    with patch.object(zipfile.ZipFile, "open", guarded), tempfile.TemporaryDirectory() as tmp:
        packets = Path(tmp) / "packets"
        prepare(packets, archive)
        require(read(packets / "protocol.json") == read(PUBLIC / "protocol.json"), "Original inputs or source changed")
        task, _, splits, _ = partitions(archive)
        for i, (_, held) in enumerate(splits):
            key = f"fold_{i}"
            verify_trial(PUBLIC / key, packets / key / "train.zip", rows_at(task, held))
        score(PUBLIC, archive)
        require(sensitivity(archive, PUBLIC) == read(PUBLIC / "partition_sensitivity.json"), "Paired sensitivity differs")
    require(seen == set(MEMBERS["titles"]), "Unexpected source files")
    smoke = PUBLIC / "smoke"
    summary = read(smoke / "smoke.json")
    target = smoke / RULE
    parent = ORIGINAL / "smoke" / RULE
    require(summary["parent"] == sha(ORIGINAL / "smoke/smoke.json") and summary["source"] == signature()
            and summary["new_classifier_fits"] == summary["model_api_calls"] == 0
            and sha(target / "run/nomination.json") == sha(parent / "run/nomination.json")
            and summary["decision"] == sha(target / "run/decision.json")
            and summary["export"] == sha(target / "direct/predictions.csv") == sha(parent / "direct/predictions.csv"),
            "Anonymous smoke differs")
    print("Reproduced cold decisions, ten raw exports, grouped score, and partition sensitivity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    pub = stages.add_parser("publish")
    pub.add_argument("out", type=Path)
    pub.add_argument("smoke", type=Path)
    pub.add_argument("archive", type=Path)
    check = stages.add_parser("verify")
    check.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.stage == "publish":
        publish(args.out, args.smoke, args.archive)
    else:
        verify(args.archive)
