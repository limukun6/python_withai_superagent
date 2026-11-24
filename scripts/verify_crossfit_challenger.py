"""Publish safe evidence and independently replay the nomination, vote, fit counts and paired score."""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np

from scripts.diagnose_crossfit_challenger import diagnose
from scripts.evaluate_crossfit_challenger import decision_from_feedback, project, score
from scripts.probe_challenger_dispatch import probe
from scripts import replay_challenger_nomination as replay
from scripts.run_conditional_handoff import load_export
from scripts.study_crossfit_challenger import prepare, study_splits
from scripts.unified_released_study import ROOT, read, require, sha
from scripts.verify_lifecycle_audit import copy
from scripts.verify_unified_raw import MEMBERS
from superagent.ab_sentence import reuse_frozen_vectors
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import screen_vectors
from tools.crossfit_challenger import nominate, residual_directions, training_roles
from tools.portfolio_training import configure

PUBLIC = ROOT / "research/crossfit_challenger_1124"
ROTATED_PUBLIC = ROOT / "research/challenger_nomination_rotation_1124"


def checksums(root):
    return "".join(f"{sha(p)}  {p.relative_to(root).as_posix()}\n" for p in sorted(root.rglob("*"))
                   if p.is_file() and p.name != "SHA256SUMS")


def seal_directory(directory):
    total = checksums(directory)
    sums = directory / "SHA256SUMS"
    if sums.exists():
        require(sums.read_text() == total, "Existing evidence differs")
    else:
        sums.write_text(total)


def publish(out, smoke):
    for name in ("protocol.json", "sealed.json", "result.json", "diagnostics.json", "dispatch.json",
                 "replay/frozen.json", "replay/result.json"):
        copy(out / name, PUBLIC / name)
    for name in ("smoke.json", "run/decision.json", "run/challenger_completion.json", "run/agent/final_selection.json",
                 "cached/predictions.csv", "direct/predictions.csv"):
        copy(smoke / name, PUBLIC / "smoke" / name)
    for i in range(5):
        folder = out / f"fold_{i}"
        dest = PUBLIC / folder.name
        for name in ("screen.npz", "rows.csv", "choice.json"):
            copy(out / "replay" / folder.name / name, PUBLIC / "replay" / folder.name / name)
        sealed = read(folder / "sealed.json")
        for name in ("sealed.json", "run/challenger_protocol.json", "run/agent/final_selection.json", *sealed["records"]):
            copy(folder / name, dest / name)
        for stage in ("run/inner_a", "run/inner_b", "run/refit", "comparison_refit"):
            for ledger in sorted((folder / stage).glob("*/fits.json")):
                for name in ("fits.json", "admission.json", "trained.json"):
                    path = ledger.parent / name
                    copy(path, dest / path.relative_to(folder))
    seal_directory(PUBLIC)
    for name in ("frozen.json", "result.json"):
        copy(out / "rotation" / name, ROTATED_PUBLIC / name)
    for i in range(5):
        for name in ("screen.npz", "choice.json", "rows.csv"):
            rel = Path(f"fold_{i}") / name
            copy(out / "rotation" / rel, ROTATED_PUBLIC / rel)
    seal_directory(ROTATED_PUBLIC)


def verify_fit(path, name, package, ids):
    admission, ledger, trained = (read(path / n) for n in ("admission.json", "fits.json", "trained.json"))
    identity = {"package": package, "ids": fingerprint(ids.tolist())}
    require(admission["candidate"] == ledger["candidate"] == trained["name"] == name and
            admission["training_identity"] == trained["identity"] == identity and
            admission["maximum_classifier_fits"] == ledger["attempted_classifier_fits"] ==
            ledger["completed_classifier_fits"] == len(ledger["calls"]) == 1 and
            ledger["calls"] == trained["fits"] == [{"classifier": "LogisticRegression", "status": "completed"}]
            and ledger["status"] == "completed" and trained["ledger"] == sha(path / "fits.json")
            and trained["model_api_calls"] == 0, "Unmatched fit, training group or native ledger")
    return trained


def verify_fold(folder, packet, task, held, verify_features=False):
    run = folder / "run"
    meta, end, decision, nom = (read(run / n) for n in ("challenger_protocol.json", "challenger_completion.json",
                                                        "decision.json", "nomination.json"))
    inventory = meta["inventory"]["candidates"]
    names = decision["members"]
    model_selection = read(run / "agent/final_selection.json")
    require(meta["package"] == sha(packet) and meta["rows"] == len(task.y_train)
            and meta["maximum_classifier_fits"] == 9 and meta["model_api_limit"] == end["model_api_calls"] == 0
            and end["decision_sha256"] == sha(run / "decision.json")
            and model_selection["model_sha256"] == end["model_sha256"] and end["strategy"] == decision["chosen"]
            and end["strategy"] == model_selection["selected_experiment"] and not model_selection["test_accessed"],
            "Agent receipt or budget changed")
    groups = sample_groups(task, "shared_text")
    roles = training_roles(groups, task.y_train)
    full = np.arange(len(groups))
    require(meta["roles"] == {k: dict(rows=len(v), ids=fingerprint(v.tolist()), groups=len(set(groups[v])))
                              for k, v in roles.items()}, "Role fingerprints differ")
    require(sum(len(set(groups[v])) for v in roles.values()) == len(set(groups)), "Nomination and OOF groups collide")
    with np.load(run / "nomination.npz", allow_pickle=False) as packed:
        require(set(packed.files) == {"row_ids", "anchors", *inventory[2:]}
                and np.array_equal(packed["row_ids"], roles["nominate"])
                and packed["anchors"].shape == (len(roles["nominate"]), 2), "Misaligned screening inputs")
        for n in inventory[2:]:
            gradients = packed[n]
            require(gradients.ndim == 2 and gradients.shape[0] == 2 and
                    np.isclose(gradients[0] @ gradients[1], nom["scores"][n], atol=1e-14, rtol=1e-10),
                    "Nomination score differs")
            if verify_features:
                ids = roles["nominate"]
                actual = residual_directions(screen_vectors(n, [task.X_train[j] for j in ids]),
                                             task.y_train[ids] - packed["anchors"].mean(1), groups[ids])
                require(np.allclose(actual, gradients, rtol=1e-6, atol=1e-9),
                        "Archived screen does not match raw features, labels and anchor residuals")
    nominee = nominate(nom["scores"])
    require(set(nom["scores"]) == set(inventory[2:]) and nominee == nom["candidate"]
            and names == inventory[:2] + ([nominee] if nominee else [])
            and nom["inputs"] == sha(run / "nomination.npz")
            and decision["nomination"] == sha(run / "nomination.json"), "Wrong runtime nomination")
    controls = read(folder / "control.json")
    require(controls["decision_sha256"] == end["decision_sha256"], "Different autonomous decision for control")
    legacy = None
    if nominee is not None:
        require(decision["oof"] == sha(run / "oof.npz"), "Changed OOF feedback")
        with np.load(run / "oof.npz", allow_pickle=False) as oof:
            choice, legacy = decision_from_feedback(names, task, roles, oof)
        require(choice["chosen"] == decision["chosen"] and choice["net"] == decision["net"]
                and np.allclose(choice["weights"], decision["weights"], atol=1e-12)
                and legacy["chosen"] == controls["legacy"]["chosen"], "OOF decision differs on replay")
    else:
        require(decision["chosen"] == "baseline" and controls["legacy"] is None, "Unexpected no-challenger vote")
    model_names = names if decision["chosen"] == "agreement_weight" else names[:2]
    require(model_names == model_selection["ensemble_components"], "Exported members differ")
    stages = {"inner_a": (names, roles["a"]), "inner_b": (names if nominee else [], roles["b"]),
              "refit": (model_names, full)}
    expected = {}
    for stage, (candidates, ids) in stages.items():
        for name in candidates:
            expected[stage, name] = verify_fit(run / stage / name, name, meta["package"], ids)
    actions = {(r["stage"], r["candidate"]): r for r in end["actions"] if r["action"] == "fit"}
    require(expected.keys() == actions.keys() and len(expected) == end["classifier_fits"] and
            all(row["ledger"] == actions[key]["ledger"] and row["model"] == actions[key]["model"]
                for key, row in expected.items()), "Candidate lineage differs")
    sealed = read(folder / "sealed.json")
    all_models = []
    for name in names:
        refit = expected.get(("refit", name))
        if refit is None:
            refit = verify_fit(folder / "comparison_refit" / name, name, meta["package"], full)
        all_models.append({k: refit[k] for k in ("name", "model", "ledger")})
    require(all_models == sealed["models"] and sealed["autonomous_fits"] == len(expected) and
            sealed["comparison_only_fits"] == len(names) - len(model_names) and
            len(list(folder.rglob("fits.json"))) == len(expected) + sealed["comparison_only_fits"] <= 9,
            "Autonomous and shadow fit counts differ")
    with np.load(folder / "outer_scores.npz", allow_pickle=False) as p:
        recomputed = project(p["probabilities"], held, decision, legacy)
    saved = load_export(folder / "rows.csv")
    direct = load_export(folder / "direct/predictions.csv")
    require(np.array_equal(saved.pop("row_id"), np.arange(len(held))) and recomputed.keys() == saved.keys()
            and all(np.array_equal(saved[k], v) for k, v in recomputed.items()) and
            np.array_equal(direct["row_id"], np.arange(len(held))) and np.array_equal(direct["prediction"], saved["agent"]),
            "Saved, reconstructed or cold native predictions differ")


def verify(archive, weights=None, cache=None):
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text(), "Public evidence checksum differs")
    require(checksums(ROTATED_PUBLIC) == (ROTATED_PUBLIC / "SHA256SUMS").read_text(), "Rotated evidence differs")
    smoke = PUBLIC / "smoke"
    summary = read(smoke / "smoke.json")
    require(summary["matches"] and sha(smoke / "cached/predictions.csv") ==
            sha(smoke / "direct/predictions.csv") == summary["prediction"] and
            summary["model"] == read(smoke / "run/agent/final_selection.json")["model_sha256"] and
            summary["classifier_fits"] == read(smoke / "run/challenger_completion.json")["classifier_fits"] <= 9
            and summary["rows"] == len(load_export(smoke / "direct/predictions.csv")["row_id"]), "Anonymous smoke differs")
    original = zipfile.ZipFile.open
    seen = set()
    def only_training(z, member, *args, **kwargs):
        if isinstance(z.filename, (str, Path)) and Path(z.filename).resolve() == archive.resolve():
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            require(name in MEMBERS["titles"], "Unexpected course member")
            seen.add(name)
        return original(z, member, *args, **kwargs)
    with patch.object(zipfile.ZipFile, "open", only_training), tempfile.TemporaryDirectory() as tmp:
        packets = Path(tmp) / "packets"
        prepare(archive, packets)
        require(sha(PUBLIC / "protocol.json") == sha(packets / "protocol.json"), "Input packages or split differ")
        whole, _, splits = study_splits(archive)
        if weights is not None:
            configure(weights, cache)
        for i, (_, held) in enumerate(splits):
            packet = packets / f"fold_{i}" / "train.zip"
            # Native folds train in separate processes. Match their input batches as well as their weights.
            with reuse_frozen_vectors():
                verify_fold(PUBLIC / f"fold_{i}", packet, load_training_task(packet),
                            [whole.X_train[j] for j in held], verify_features=weights is not None)
        score(PUBLIC, archive)
        diagnose(PUBLIC, archive)
        replay.freeze(PUBLIC / "replay", archive, weights, cache)
        replay.score(PUBLIC / "replay", archive)
        replay.freeze(ROTATED_PUBLIC, archive, weights, cache, rotate=True)
        replay.score(ROTATED_PUBLIC, archive, rotate=True)
    require(seen == set(MEMBERS["titles"]), "Course training member audit failed")
    dispatcher = read(PUBLIC / "dispatch.json")
    require(dispatcher["classifier_fits"] == dispatcher["model_api_calls"] == 0 and
            dispatcher["calls_before_first_fit"] == 15 and len(dispatcher["cases"]) == 6 and
            all(v["same_on_rename"] for v in dispatcher["cases"].values()), "Incomplete dispatch evidence")
    if weights is not None:
        require(probe(weights.resolve()) == dispatcher, "Live task/asset dispatch differs")
    print("Verified five nominations, fit admissions, independent raw predictions, paired score and diagnostic strata.")
    print("Read course members:", sorted(seen))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest="stage", required=True)
    publish_cmd = stages.add_parser("publish")
    publish_cmd.add_argument("out", type=Path)
    publish_cmd.add_argument("smoke", type=Path)
    verify_cmd = stages.add_parser("verify")
    verify_cmd.add_argument("archive", type=Path)
    verify_cmd.add_argument("--weights", type=Path)
    verify_cmd.add_argument("--cache", type=Path)
    args = p.parse_args()
    if args.stage == "publish":
        publish(args.out, args.smoke)
    else:
        verify(args.archive, args.weights, args.cache)
