"""One gated transfer to human-ranked English answers using the registered cold strategy."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_brier_replication import RULE, freeze_one, signature
from scripts.study_pilot_nomination import run_arm
from scripts.unified_released_study import ROOT, read, require, save, sha, text_packet
from scripts.verify_brier_replication import copy_trial, verify_trial
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import normalized_text
from tools.challenger_training import rows_at
from tools.encoder_assets import MODELS
from tools.portfolio_encoders import build_bank
from tools.portfolio_training import configure

REVISION = "a2c461b87baf5f2c581c20ca75d13d4e7b21c5cb"
ARCHIVE_SHA = "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c"
PLAN = ROOT / "docs/batches/pilot_external_preference_plan_20261124_zh.md"
PUBLIC = ROOT / "research/pilot_external_preference_1124"


def hashed(prefix, value):
    return hashlib.sha256((prefix + value).encode()).digest()


def make_pairs(messages):
    parents = {}
    replies = defaultdict(lambda: defaultdict(list))
    for msg in messages:
        text = msg.get("text")
        if (msg.get("lang") != "en" or msg.get("deleted") is not False or msg.get("synthetic") is not False
                or not isinstance(text, str)):
            continue
        length = len(text.split())
        if msg["parent_id"] is None and msg["role"] == "prompter" and 3 <= length <= 40:
            require(msg["message_tree_id"] not in parents, "More than one root in a tree")
            parents[msg["message_tree_id"]] = (msg["message_id"], text)
        elif msg["role"] == "assistant" and msg["rank"] in (0, 1) and 10 <= length <= 80:
            replies[(msg["message_tree_id"], msg["parent_id"])][msg["rank"]].append(text)
    pairs, used = [], set()
    for tree in sorted(parents, key=lambda t: (hashed("pilot-external-trees-v1/", t), t)):
        parent, question = parents[tree]
        options = replies.get((tree, parent), {})
        if len(options.get(0, [])) != 1 or len(options.get(1, [])) != 1:
            continue
        first, second = options[0][0], options[1][0]
        parts = {normalized_text(piece) for piece in (question, first, second)}
        if len(parts) != 3 or parts & used:
            continue
        used.update(parts)
        left_wins = hashed("pilot-external-order-v1/", tree)[0] % 2 == 0
        left, right = (first, second) if left_wins else (second, first)
        pairs.append((tree, question + "\n" + left, question + "\n" + right, int(left_wins)))
        if len(pairs) == 1024:
            break
    return pairs


def extract(source):
    require(sha(source) == ARCHIVE_SHA, "Public archive differs from the pinned publisher object")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        pairs = make_pairs(json.loads(line) for line in stream)
    return [p for i, p in enumerate(pairs) if i % 5], pairs[::5]


def promotion(parent):
    frozen = read(parent / "frozen.json")
    result = read(parent / "result.json")
    require(frozen["protocol"] == sha(parent / "protocol.json") and read(parent / "protocol.json")["source"] == signature()
            and result["frozen"] == sha(parent / "frozen.json") and result["passes"]
            and result["positive_folds"] >= 4 and result["performance"]["net"] >= 5
            and result["performance"]["paired_95_fraction"][0] > 0, "Cold nomination did not qualify for transfer")
    return sha(parent / "result.json")


def prepare(folder, source, parent):
    upstream = promotion(parent)
    train, held = extract(source)
    folder.mkdir(parents=True, exist_ok=False)
    eligible = (len(train) >= 200 and len(held) >= 50
                and {p[-1] for p in train} == {p[-1] for p in held} == {0, 1})
    meta = dict(promotion=upstream, revision=REVISION, publisher_sha256=ARCHIVE_SHA, plan=sha(PLAN),
                agent_source=signature(), preparer=sha(Path(__file__)), feasible=eligible,
                train_trees=len(train), outer_trees=len(held), train_tree_order=fingerprint([p[0] for p in train]),
                outer_tree_order=fingerprint([p[0] for p in held]), max_cold_fits=9, model_api_limit=0)
    if eligible:
        text_packet(folder / "train.zip", [p[1:] for p in train], True)
        text_packet(folder / "features.zip", [p[1:3] for p in held], False)
        meta.update(training=sha(folder / "train.zip"), features=sha(folder / "features.zip"))
    save(folder / "protocol.json", meta)
    print("External eligibility:", eligible, "training trees", len(train), "outer trees", len(held))


def execute(folder, weights):
    meta = read(folder / "protocol.json")
    require(meta["feasible"] and meta["agent_source"] == signature() and meta["preparer"] == sha(Path(__file__))
            and sha(folder / "train.zip") == meta["training"] and sha(folder / "features.zip") == meta["features"],
            "Anonymous inputs are unavailable or changed")
    cache = folder / "training_vectors"
    configure(weights, cache)
    task = load_training_task(folder / "train.zip")
    rows = rows_at(task, np.arange(len(task.y_train)))
    for name in MODELS:
        build_bank(name, rows, cache)
    end = run_arm(folder, RULE, [], weights, cache)
    require(end["reused_classifier_fits"] == 0 and end["cold_classifier_fits"] <= 9, "Unexpected external fit")
    if not (folder / "sealed.json").exists():
        freeze_one(folder, weights, cache)


def score(folder, source, parent):
    meta = read(folder / "protocol.json")
    seal = read(folder / "sealed.json")
    require(meta["promotion"] == promotion(parent) and meta["feasible"] and
            all(sha(folder / name) == digest for name, digest in seal["records"].items()), "Missing frozen external outputs")
    _, held = extract(source)
    require(fingerprint([p[0] for p in held]) == meta["outer_tree_order"], "Tree order differs")
    rows = load_export(folder / "rows.csv")
    require(np.array_equal(rows["row_id"], np.arange(len(held))), "Outer rows differ")
    stats = comparison(rows["baseline"], rows["agent"], np.array([p[-1] for p in held]), np.array([p[0] for p in held]))
    decision = read(folder / RULE / "run/decision.json")
    result = dict(performance=stats, sealed=sha(folder / "sealed.json"), protocol=sha(folder / "protocol.json"),
                  choice=decision["approved"], nominee=decision["nominee"], fits=seal["fits"], model_api_calls=0)
    save(folder / "result.json", result)
    print("External transfer:", result)


def publish(folder):
    meta = read(folder / "protocol.json")
    copy(folder / "protocol.json", PUBLIC / "protocol.json")
    if meta["feasible"]:
        copy_trial(folder, PUBLIC)
        copy(folder / "result.json", PUBLIC / "result.json")
    seal_directory(PUBLIC)


def verify(source, parent):
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text(), "External evidence changed")
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "trial"
        prepare(rebuilt, source, parent)
        meta = read(PUBLIC / "protocol.json")
        require(meta == read(rebuilt / "protocol.json"), "Anonymous external split differs")
        if meta["feasible"]:
            _, held = extract(source)
            verify_trial(PUBLIC, rebuilt / "train.zip", [r[1:3] for r in held])
            score(PUBLIC, source, parent)
    print("Verified the pinned external data and all applicable trial evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(required=True, dest="stage")
    prep = stages.add_parser("prepare")
    execute_cmd = stages.add_parser("execute")
    scoring = stages.add_parser("score")
    publishing = stages.add_parser("publish")
    check = stages.add_parser("verify")
    for cmd in (prep, execute_cmd, scoring, publishing):
        cmd.add_argument("folder", type=lambda s: Path(s).resolve())
    for cmd in (prep, scoring, check):
        cmd.add_argument("source", type=Path)
        cmd.add_argument("parent", type=Path)
    execute_cmd.add_argument("weights", type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.folder, args.source, args.parent)
    elif args.stage == "execute":
        execute(args.folder, args.weights)
    elif args.stage == "score":
        score(args.folder, args.source, args.parent)
    elif args.stage == "publish":
        publish(args.folder)
    else:
        verify(args.source, args.parent)
