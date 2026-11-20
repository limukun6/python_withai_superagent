"""Replay equal-probability voting from saved finalists, without fitting any model."""
import argparse
import math
import os
from pathlib import Path

import joblib
import numpy as np

from superagent.data import load_training_task
from superagent.models import SoftVoteModel, predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, train_only, write


def load_model(path):
    return joblib.load(path)["model"]


def freeze_file(path, value):
    if path.exists():
        expect(read(path) == value, f"Frozen artifact changed at {path}")
    else:
        write(path, value)


def prepare_fold(source, full_folder):
    rec = receipt(source)
    confirmation = read(source / "confirmation.json")
    winner = confirmation["selected_candidate"]
    names = [winner, *[n for n in confirmation["shortlist"] if n != winner]]
    expect(len(names) == 3 and rec["winner"] == winner, "Need three confirmed finalists")
    numbers = {r["candidate"]: r["trial"] for r in read(source / "trials.json")}
    paths = {name: source / f"trial_{numbers[name]}.joblib" for name in names}
    full = {}
    if full_folder is not None:
        hashes = read(full_folder / "fits.json")
        expect(set(hashes) == set(names[1:]), "Different diagnostic finalists")
        for i, name in enumerate(names[1:], 1):
            path = full_folder / f"full_{i}.joblib"
            expect(digest(path) == hashes[name], "Full refit changed")
            full[name] = str(path)
    return dict(names=names, source=str(source), winner_hash=rec["model_sha256"],
                primary_files={name: str(p) for name, p in paths.items()},
                primary_hashes={name: digest(p) for name, p in paths.items()},
                inner_hash=digest(source / "inner_data.joblib"), extra_full=full,
                extra_full_hashes={name: digest(Path(p)) for name, p in full.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("archive", "reference", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--diagnostic-full", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    os.environ["SUPERAGENT_OFFLINE"] = "1"
    os.environ.pop("SUPERAGENT_PRETRAINED_RUNTIME", None)
    protocol = read(args.reference / "protocol.json")
    expect(protocol["archive_sha256"] == digest(args.archive), "Archive changed")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for i, block in enumerate(protocol["locations"]):
        full = args.diagnostic_full / f"fold_{i}" if args.diagnostic_full else None
        records.append(prepare_fold(Path(block[args.arm]), full))
    code_root = Path(__file__).resolve().parents[1]
    freeze_file(root / "protocol.json", dict(
        archive_sha256=protocol["archive_sha256"], reference=digest(args.reference / "protocol.json"),
        arm=args.arm, seed=protocol["seed"], records=records, source_sha256=digest(Path(__file__)),
        probability_routine_sha256=digest(code_root / "superagent/models.py"),
        rule="Equal probability weights across the three confirmed finalists. Compare always averaging and a gate based only on the primary search split. Adopt if net corrected rows are at least 0.2 percentage points and at least the square root of disagreements.",
        cost="Both policies use the single already-refitted winner and two retained search models. Fully refitted averaging is a separate diagnostic when supplied. New fits and API calls are zero."))
    if args.freeze:
        return
    opened = []
    with train_only(args.archive, opened):
        task = load_training_task(args.archive)
    votes = []
    for i, record in enumerate(records):
        folder = root / f"fold_{i}"
        folder.mkdir(exist_ok=True)
        source = Path(record["source"])
        inner = joblib.load(source / "inner_data.joblib")
        classes = np.unique(inner["y_train"])
        models = [load_model(record["primary_files"][name]) for name in record["names"]]
        anchor = predict_with_model(task.kind, models[0], inner["X_search"])
        blend = SoftVoteModel(task.kind, models, classes).predict(inner["X_search"])
        truth = inner["y_search"]
        corrected = int(np.sum((anchor != truth) & (blend == truth)))
        lost = int(np.sum((anchor == truth) & (blend != truth)))
        adopt = corrected-lost >= max(.002*len(truth), math.sqrt(corrected+lost))
        freeze_file(folder / "decision.json", dict(adopt=adopt, corrected=corrected, lost=lost, rows=len(truth)))
        winner = load_model(source / "selected_model.joblib")
        vote = SoftVoteModel(task.kind, [winner, *models[1:]], classes)
        full_vote = None
        if record["extra_full"]:
            others = [load_model(record["extra_full"][name]) for name in record["names"][1:]]
            full_vote = SoftVoteModel(task.kind, [winner, *others], classes)
        votes.append((winner, vote, full_vote, adopt))
    folds = []
    for i, (winner, vote, full_vote, adopt) in enumerate(votes):
        ids = protocol["indices"][i]["outer"]
        x = take(task.X_train, ids)
        truth = task.y_train[ids]
        anchor = predict_with_model(task.kind, winner, x)
        frozen = read(args.reference / f"fold_{protocol['fold_numbers'][i]-1}" / "outer_predictions.json")
        expect(frozen["rows"] == ids, "Outer row order changed")
        np.testing.assert_array_equal(frozen[args.arm], anchor)
        always = vote.predict(x)
        predictions = dict(incumbent=anchor, always=always, gated=always if adopt else anchor)
        if full_vote is not None:
            predictions["diagnostic_full"] = full_vote.predict(x)
        write(root / f"fold_{i}" / "outer_predictions.json",
              dict(rows=ids, truth=truth.tolist(), **{k:v.tolist() for k,v in predictions.items()}))
        folds.append(dict(rows=len(ids), adopted=adopt,
                          correct={k:int(np.sum(v == truth)) for k,v in predictions.items()}))
    result = dict(rows=sum(f["rows"] for f in folds), folds=folds, training_members=opened,
                  correct={k:sum(f["correct"][k] for f in folds) for k in folds[0]["correct"]},
                  new_classifier_fits=0, new_api_calls=0)
    write(root / "results.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
