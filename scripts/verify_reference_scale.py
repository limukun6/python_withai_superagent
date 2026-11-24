"""Validate the new training rows, each physical fit, the second confirmation and native outputs."""
import argparse
from pathlib import Path
import tempfile

import numpy as np

from scripts.run_conditional_handoff import load_export
from scripts.study_brier_replication import RULE
from scripts.study_consensus_veto import packet_for, trials
from scripts import study_reference_scale as study
from scripts.unified_released_study import ROOT, read, require, sha
from scripts.verify_brier_replication import ORIGINAL, PUBLIC as REGROUPED
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from scripts.verify_pilot_nomination import verify_one_receipt
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.pilot_challenger import pilot_roles

PUBLIC = ROOT / "research/reference_scale_1124"


def publish(out):
    for name in ("protocol.json", "frozen.json", "result.json"):
        copy(out / name, PUBLIC / name)
    for key, rec in read(out / "frozen.json")["records"].items():
        for name in rec["files"]:
            copy(out / key / name, PUBLIC / key / name)
    seal_directory(PUBLIC)


def verify(archive):
    for folder in (PUBLIC, ORIGINAL, REGROUPED):
        require(checksums(folder) == (folder / "SHA256SUMS").read_text(), "Evidence checksum changed")
    protocol, frozen = read(PUBLIC / "protocol.json"), read(PUBLIC / "frozen.json")
    require(protocol["source"] == sha(Path(study.__file__)) and protocol["plan"] == sha(study.PLAN)
            and protocol["first"] == sha(ORIGINAL / "frozen.json") and protocol["second"] == sha(REGROUPED / "frozen.json"),
            "Incorrect upstream evidence or source")
    fits = 0
    for name, src, data, _, train, held in trials(archive):
        key = name + "/" + src.name
        dest = PUBLIC / key
        rec = frozen["records"][key]
        run, old, nom = study.previous(src)
        with tempfile.TemporaryDirectory() as tmp:
            task = packet_for(src, data, train, Path(tmp))
            identity = dict(package=sha(Path(tmp) / "train.zip"))
        end = read(run / "pilot_completion.json")
        receipts = read(dest / "receipts.json")
        require(len(receipts) == (2 if old["approved"] else 0) == rec["fits"] and
                end["cold_classifier_fits"] + rec["fits"] <= 11, "Unaccounted training or path over budget")
        fits += rec["fits"]
        matrix = None
        if old["approved"]:
            groups = sample_groups(task, "shared_text")
            role = pilot_roles(groups, task.y_train)
            ix = np.sort(np.r_[role["pilot"], role["learn"]])
            identity.update(ids=fingerprint(ix.tolist()))
            require(not set(groups[ix]) & set(groups[role["confirm"]]) and set(receipts) == set(nom["members"][:2]),
                    "Unregistered confirm training or wrong reference heads")
            for n in nom["members"][:2]:
                verify_one_receipt(receipts[n], identity, n)
            with np.load(dest / "larger.npz", allow_pickle=False) as fb:
                matrix = fb["scores"]
        decision = study.reconsider(src, task, matrix)
        require(decision == read(dest / "decision.json"), "Stronger-reference decision differs")
        original_rows = load_export(src / "rows.csv")
        previous = load_export(src / RULE / "direct/predictions.csv")
        rows = load_export(dest / "rows.csv")
        changed = decision["third"] != decision["approved"]
        expected = original_rows["baseline"] if changed else previous["prediction"]
        delivery = (dest if changed else src / RULE) / "direct/predictions.csv"
        native = load_export(delivery)
        require(changed == rec["changed"] and rec["native"] == sha(delivery)
                and np.array_equal(rows["row_id"], np.arange(len(held)))
                and np.array_equal(rows["row_id"], native["row_id"])
                and np.array_equal(rows["baseline"], original_rows["baseline"])
                and np.array_equal(rows["previous"], previous["prediction"])
                and np.array_equal(rows["scaled"], expected) and np.array_equal(native["prediction"], expected),
                "New raw-text delivery differs")
        if changed:
            selection = read(dest / "run/agent/final_selection.json")
            require(selection["ensemble_components"] == nom["members"][:2] and not selection["test_accessed"],
                    "Incorrect recalled delivery")
    require(fits == protocol["extra_fits"] == 16, "Unaccounted reference training")
    study.score(PUBLIC, archive)
    print("Verified 16 physical fits, original and enlarged training boundaries, ten decisions and all raw outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("publish").add_argument("out", type=Path)
    sub.add_parser("verify").add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.stage == "publish":
        publish(args.out)
    else:
        verify(args.archive)
