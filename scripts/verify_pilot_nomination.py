"""Verify native decisions, fit lineage, published predictions and grouped scores without new fits."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np

from scripts.evaluate_pilot_nomination import columns_for, project, score
from scripts.run_conditional_handoff import load_export
from scripts.study_crossfit_challenger import prepare as reconstruct_packets, study_splits
from scripts.study_pilot_nomination import PUBLIC_PRIOR
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from scripts.verify_unified_raw import MEMBERS
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import rows_at, ties
from tools.pilot_challenger import STRATEGIES, approve, compare_vote, nominate, panel_scores, pilot_roles


PUBLIC = ROOT / "research/pilot_nomination_1124"


def receipts(folder):
    parts = {}
    for arm in STRATEGIES:
        run = folder / arm / "run"
        for a in read(run / "pilot_completion.json")["actions"]:
            src = run / a["stage"] / a["candidate"]
            parts[f"{arm}/{a['stage']}/{a['candidate']}"] = src
    for p in sorted((folder / "shadow").glob("*/trained.json")):
        parts["shadow/" + p.parent.name] = p.parent
    records = {}
    for key, src in parts.items():
        train = read(src / "trained.json")
        require(sha(src / "model.joblib") == train["model"] and sha(src / "fits.json") == train["ledger"],
                "Fit receipt does not bind the model")
        records[key] = {k: read(src / name) for k, name in (("fit", "trained.json"), ("ledger", "fits.json"),
                                                              ("admission", "admission.json"))}
    return records


def publish(out, smoke):
    for name in ("protocol.json", "frozen.json", "result.json"):
        copy(out / name, PUBLIC / name)
    copy(smoke / "smoke.json", PUBLIC / "smoke/smoke.json")
    for s in STRATEGIES:
        for name in ("run/pilot_protocol.json", "run/pilot_completion.json", "run/decision.json", "run/nomination.json",
                     "direct/predictions.csv", "cached/predictions.csv"):
            copy(smoke / s / name, PUBLIC / "smoke" / s / name)
    save(PUBLIC / "smoke/fit_receipts.json", receipts(smoke))
    for i in range(5):
        folder = out / f"fold_{i}"
        target = PUBLIC / folder.name
        for name in ("sealed.json", *read(folder / "sealed.json")["records"]):
            copy(folder / name, target / name)
        save(target / "fit_receipts.json", receipts(folder))
    seal_directory(PUBLIC)


def verify_one_receipt(payload, identity, candidate):
    trained, ledger, admission = (payload[k] for k in ("fit", "ledger", "admission"))
    ledger_sha = hashlib.sha256(json.dumps(ledger, indent=2).encode()).hexdigest()
    require(trained["identity"] == admission["training_identity"] == identity and trained["ledger"] == ledger_sha and
            trained["name"] == admission["candidate"] == ledger["candidate"] == candidate and
            ledger["calls"] == trained["fits"] == [{"classifier": "LogisticRegression", "status": "completed"}]
            and ledger["completed_classifier_fits"] == ledger["attempted_classifier_fits"] ==
            admission["maximum_classifier_fits"] == 1 and ledger["status"] == "completed" and
            trained["model_api_calls"] == 0, "Different classifier identity or budget")
    return trained


def verify_fold(folder, train_packet, outer, i):
    task = load_training_task(train_packet)
    package = sha(train_packet)
    groups = sample_groups(task, "shared_text")
    roles = pilot_roles(groups, task.y_train)
    ids = {k: fingerprint(v.tolist()) for k, v in roles.items()}
    full = fingerprint(list(range(len(groups))))
    require(len(set(groups)) == sum(len(set(groups[ix])) for ix in roles.values()), "Overlapping pilot/learn/confirm groups")
    recs = read(folder / "fit_receipts.json")
    seal = read(folder / "sealed.json")
    prior_root = PUBLIC_PRIOR / folder.name
    available = set()
    if i:
        for scope in ("run/inner_a", "run/refit", "comparison_refit"):
            for p in (prior_root / scope).glob("*/trained.json"):
                r = read(p)
                available.add((r["identity"]["ids"], r["name"], r["model"], r["ledger"]))
    choices, total_new = {}, 0
    for s in STRATEGIES:
        run = folder / s / "run"
        meta, nom, choice, end = (read(run / f) for f in ("pilot_protocol.json", "nomination.json",
                                                           "decision.json", "pilot_completion.json"))
        names = meta["inventory"]["candidates"]
        require(meta["package"] == package and meta["strategy"] == nom["strategy"] == end["strategy"] == s
                and meta["roles"] == {k: dict(rows=len(v), groups=len(set(groups[v])), ids=ids[k]) for k, v in roles.items()}
                and nom["members"] == seal["pilot_members"] == names and meta["cold_fit_ceiling"] == len(names)+3,
                "Goal, budget or input groups changed")
        learn, confirm = roles["learn"], roles["confirm"]
        with np.load(run / "feedback.npz", allow_pickle=False) as fb:
            require(np.array_equal(fb["learn_ids"], learn), "Mismatched nomination rows")
            observed = panel_scores(names, fb["learn"], rows_at(task, learn), task.y_train[learn], groups[learn])
            candidate = nominate(observed, s)
            require(observed == nom["learning"] and nom["chosen"] == candidate and nom["learning_rows"] == ids["learn"],
                    "Candidate nomination differs")
            checked = None
            if candidate:
                require(np.array_equal(fb["confirm_ids"], confirm) and fb["confirm"].shape == (len(confirm), 3),
                        "Confirmation refers to different rows")
                p = fb["confirm"]
                checked = compare_vote(p[:, :2].mean(1), p[:, 2], ties(rows_at(task, confirm)), task.y_train[confirm], groups[confirm])
            else:
                require(fb["confirm"].size == fb["confirm_ids"].size == 0, "Unexpected confirmation")
        expected = approve(candidate, observed, checked, len(confirm))
        expected.update(nomination=sha(run / "nomination.json"), feedback=sha(run / "feedback.npz"))
        select = read(run / "agent/final_selection.json")
        chosen_names = names[:2] + ([choice["approved"]] if choice["approved"] else [])
        require(choice == expected and end["decision_sha256"] == sha(run / "decision.json") and
                end["model_sha256"] == select["model_sha256"] and chosen_names == select["ensemble_components"] and
                select["selected_experiment"] == (choice["approved"] or "baseline") and not select["test_accessed"],
                "Declining or approving the nominee cannot be reproduced")
        expected_keys = [("pilot", n) for n in names] + [("refit", n) for n in chosen_names]
        require([(a["stage"], a["candidate"]) for a in end["actions"]] == expected_keys, "Unaccounted fit")
        previous = set(available)
        for a in end["actions"]:
            stage, n = a["stage"], a["candidate"]
            identity = {"ids": ids["pilot"] if stage == "pilot" else full, "package": package}
            payload = recs[f"{s}/{stage}/{n}"]
            r = verify_one_receipt(payload, identity, n)
            ident = (identity["ids"], n, r["model"], r["ledger"])
            require(a["identity"] == identity and a["model"] == r["model"] and a["ledger"] == r["ledger"]
                    and (not a["reused"] or ident in previous), "Reuse not traceable to an earlier identical fit")
            available.add(ident)
        fresh = sum(not a["reused"] for a in end["actions"])
        require(end["new_classifier_fits"] == fresh and end["reused_classifier_fits"] == len(expected_keys)-fresh and
                end["cold_classifier_fits"] == len(expected_keys) <= 9 and end["model_api_calls"] == 0, "Mismatch in fit counts")
        total_new += fresh
        choices[s] = choice
    expected = {f"{s}/{a['stage']}/{a['candidate']}" for s in STRATEGIES
                for a in read(folder/s/"run/pilot_completion.json")["actions"]}
    for n in seal["shadow_names"]:
        key = "shadow/"+n
        r = verify_one_receipt(recs[key], {"ids": full, "package": package}, n)
        available.add((full, n, r["model"], r["ledger"]))
        expected.add(key)
    require(expected == set(recs) and total_new + len(seal["shadow_names"]) == seal["total_new"] and
            seal["shadow_new"] == len(seal["shadow_names"]), "Shadow fit budget differs")
    for n, r in seal["refit_records"].items():
        require((full, n, r["model"], r["ledger"]) in available and r["identity"] == {"ids": full, "package": package},
                "Unverified refit used for controls")
    with np.load(folder / "outer.npz", allow_pickle=False) as p:
        pilots = project(seal["pilot_members"], p["pilot"], outer)
        refits = project(seal["refit_members"], p["refit"], outer)
    prior = prior_root / "direct/predictions.csv"
    require(sha(prior) == seal["prior_predictions"], "Baseline reference changed")
    all_columns = columns_for(pilots, refits, choices, load_export(prior)["prediction"])
    rows = load_export(folder / "rows.csv")
    require(np.array_equal(rows.pop("row_id"), np.arange(len(outer))) and set(rows) == set(all_columns)
            and all(np.array_equal(rows[k], v) for k, v in all_columns.items()), "Row-wise decisions differ")
    for s in STRATEGIES:
        direct = load_export(folder / s / "direct/predictions.csv")
        require(np.array_equal(direct["row_id"], np.arange(len(outer))) and
                np.array_equal(direct["prediction"], rows[s+"_approved"]), "Raw native predictions differ")


def verify(archive):
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text(), "Evidence checksum differs")
    seen = set()
    original = zipfile.ZipFile.open
    def guard(z, member, *args, **kwargs):
        if isinstance(z.filename, (str, Path)) and Path(z.filename).resolve() == archive.resolve():
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            require(name in MEMBERS["titles"], "Nontraining course member")
            seen.add(name)
        return original(z, member, *args, **kwargs)
    with patch.object(zipfile.ZipFile, "open", guard), tempfile.TemporaryDirectory() as tmp:
        packets = Path(tmp) / "packets"
        reconstruct_packets(archive, packets)
        require(read(packets/"protocol.json")["folds"] == read(PUBLIC/"protocol.json")["folds"], "Original data differs")
        task, _, splits = study_splits(archive)
        for i, (_, held) in enumerate(splits):
            key = f"fold_{i}"
            verify_fold(PUBLIC / key, packets / key / "train.zip", rows_at(task, held), i)
        score(PUBLIC, archive)
    require(seen == set(MEMBERS["titles"]), "Unexpected course input")
    smoke = PUBLIC / "smoke"
    summary = read(smoke / "smoke.json")
    for s, row in summary["arms"].items():
        require(sha(smoke / s / "direct/predictions.csv") == sha(smoke / s / "cached/predictions.csv") == row["export"]
                and row["decision"] == sha(smoke/s/"run/decision.json") and row["model"] ==
                read(smoke/s/"run/pilot_completion.json")["model_sha256"], "Anonymous delivery changed")
    print("Verified nomination, independent confirmation, fit counts, ten raw exports and paired scores.")
    print("Course members read:", sorted(seen))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="stage", required=True)
    pub = commands.add_parser("publish")
    pub.add_argument("out", type=Path)
    pub.add_argument("smoke", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("archive", type=Path)
    a = p.parse_args()
    if a.stage == "publish":
        publish(a.out, a.smoke)
    else:
        verify(a.archive)
