"""Learn a local veto from existing training receipts, seal predictions, then score both partitions."""
import argparse
from pathlib import Path
import tempfile

import joblib
import numpy as np

from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_brier_replication import RULE, partitions
from scripts.study_crossfit_challenger import native_predict
from scripts.study_encoder_fusion import save_csv
from scripts.study_pilot_nomination import link_file
from scripts.unified_released_study import ROOT, read, require, save, sha, text_packet
from scripts.verify_brier_replication import ORIGINAL, PUBLIC as REGROUPED
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from superagent.data import load_training_task
from superagent.splits import sample_groups
from tools.challenger_training import rows_at, ties
from tools.consensus_veto import ConsensusPreferenceVote, adjusted, confirm, propose
from tools.pilot_challenger import pilot_roles
from tools.portfolio_training import write_selection
from tools.preference_handoff import binary

PLAN = ROOT / "docs/batches/pilot_consensus_veto_plan_20261124_zh.md"
PUBLIC = ROOT / "research/pilot_consensus_veto_1124"
MODEL_CODE = ROOT / "tools/consensus_veto.py"


def trials(archive):
    task, groups, fresh, old = partitions(archive)
    for name, public, splits in (("first", ORIGINAL, old), ("second", REGROUPED, fresh)):
        for i, (train, held) in enumerate(splits):
            yield name, public / f"fold_{i}", task, groups, train, held


def decide(src, task):
    run = src / RULE / "run"
    global_action = read(run / "decision.json")
    third = global_action["approved"]
    result = dict(global_decision=sha(run / "decision.json"), third=third, nomination=None, approved=None,
                  learning=None, confirmation=None, discrete_choices=int(bool(third)))
    if third is None:
        return result
    groups = sample_groups(task, "shared_text")
    roles = pilot_roles(groups, task.y_train)
    learn, check = roles["learn"], roles["confirm"]
    names = read(run / "nomination.json")["members"]
    with np.load(run / "feedback.npz", allow_pickle=False) as fb:
        require(np.array_equal(fb["learn_ids"], learn) and np.array_equal(fb["confirm_ids"], check),
                "Unmatched inner feedback")
        subset = [0, 1, names.index(third)]
        region, evidence = propose(fb["learn"][:, subset], ties(rows_at(task, learn)), task.y_train[learn], groups[learn])
        chosen, validation = confirm(region, fb["confirm"], ties(rows_at(task, check)), task.y_train[check], groups[check])
    result.update(nomination=region, approved=chosen, learning=evidence, confirmation=validation)
    return result


def replay(src, split_name, outer, d):
    seal = read(src / "sealed.json")
    if split_name == "first":
        names, filename, key = seal["refit_members"], "outer.npz", "refit"
    else:
        names, filename, key = seal["members"], "scores.npz", "scores"
    require(sha(src / filename) == seal["records"][filename], "Published refit scores changed")
    tie = ties(outer)
    with np.load(src / filename, allow_pickle=False) as file:
        p = file[key]
        base = binary(p[:, :2].mean(1), tie)
        agree = binary(p[:, 0], tie) == binary(p[:, 1], tie)
        native = load_export(src / RULE / "direct/predictions.csv")
        if d["third"]:
            three = p[:, [0, 1, names.index(d["third"])]]
            global_pred = binary(adjusted(three, tie, None), tie)
            output = binary(adjusted(three, tie, d["approved"]), tie)
        else:
            global_pred = output = base
    require(np.array_equal(native["row_id"], np.arange(len(outer)))
            and np.array_equal(global_pred, native["prediction"]), "Original native prediction differs")
    return dict(reference_agree=agree.astype(int), baseline=base, global_agent=global_pred, veto_agent=output)


def export_change(dest, src, local, task, decision, weights):
    original = local / RULE / "run/agent"
    selection = read(src / RULE / "run/agent/final_selection.json")
    require(sha(original / "selected_model.joblib") == selection["model_sha256"], "Underlying selected model changed")
    old = joblib.load(original / "selected_model.joblib")["model"]
    wrapper = ConsensusPreferenceVote(old.models, decision["approved"])
    write_selection(dest / "run/agent", wrapper, selection["ensemble_components"], task.spec,
                    decision["approved"], "confirmed-consensus-reversion")
    link_file(local / "features.zip", dest / "features.zip")
    return native_predict(dest, weights)


def packet_for(src, task, indices, folder):
    packet = folder / "train.zip"
    text_packet(packet, [(*task.X_train[i], int(task.y_train[i])) for i in indices], True)
    require(sha(packet) == read(src / RULE / "run/pilot_protocol.json")["package"], "Different training data")
    return load_training_task(packet)


def freeze(out, archive, parents, weights):
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, src, task, _, indices, held in trials(archive):
        dest = out / name / src.name
        dest.mkdir(parents=True)
        training = packet_for(src, task, indices, dest)
        d = decide(src, training)
        save(dest / "decision.json", d)
        predictions = replay(src, name, rows_at(task, held), d)
        save_csv(dest / "rows.csv", range(len(held)), predictions)
        if d["approved"]:
            local = parents[name] / src.name
            result = export_change(dest, src, local, training, d, weights)
            require(np.array_equal(result, predictions["veto_agent"]), "Changed deployment differs from replay")
            native = sha(dest / "direct/predictions.csv")
        else:
            native = sha(src / RULE / "direct/predictions.csv")
        records[name + "/" + src.name] = dict(decision=sha(dest / "decision.json"), rows=sha(dest / "rows.csv"),
                                               parent=sha(src / "sealed.json"), native=native,
                                               new_native_export=bool(d["approved"]))
    save(out / "frozen.json", dict(plan=sha(PLAN), model_source=sha(MODEL_CODE), evaluation=sha(Path(__file__)),
                                    records=records, classifier_fits=0, continuous_fits=0, model_api_calls=0))


def score(out, archive):
    frozen = read(out / "frozen.json")
    require(frozen["plan"] == sha(PLAN) and frozen["model_source"] == sha(MODEL_CODE)
            and frozen["evaluation"] == sha(Path(__file__)), "Veto source differs")
    collected, details = {}, {}
    for name, src, task, groups, _, held in trials(archive):
        dest = out / name / src.name
        record = frozen["records"][name + "/" + src.name]
        require(sha(dest / "rows.csv") == record["rows"] and sha(dest / "decision.json") == record["decision"] and
                sha(src / "sealed.json") == record["parent"], "Unsealed veto output")
        rows = load_export(dest / "rows.csv")
        require(np.array_equal(rows.pop("row_id"), np.arange(len(held))), "Wrong row order")
        decision = read(dest / "decision.json")
        details.setdefault(name, []).append(dict(third=decision["third"], nominated=decision["nomination"],
                                                approved=decision["approved"], choices=decision["discrete_choices"]))
        table = collected.setdefault(name, {})
        for k, v in dict(**rows, truth=task.y_train[held], groups=groups[held]).items():
            table.setdefault(k, []).append(v)
    results = {}
    for name, cols in collected.items():
        d = {k: np.concatenate(v) for k, v in cols.items()}
        y, g = d["truth"], d["groups"]
        require(len(y) == 6608 and len(set(g)) == 4874, "Wrong evaluation panel")
        by_region = {}
        for label, flag in (("agree", 1), ("disagree", 0)):
            subset = d["reference_agree"] == flag
            by_region[label] = comparison(d["baseline"][subset], d["global_agent"][subset], y[subset], g[subset])
        results[name] = dict(global_to_veto=comparison(d["global_agent"], d["veto_agent"], y, g),
                             reference_to_global_by_region=by_region, decisions=details[name])
    deltas = [v["global_to_veto"] for v in results.values()]
    passes = all(v["net"] > 0 for v in deltas) and any(v["paired_95_fraction"][0] > 0 for v in deltas)
    save(out / "result.json", dict(frozen=sha(out / "frozen.json"), comparisons=results, passes=passes))
    print("Veto off versus on:", {k: v["global_to_veto"]["net"] for k, v in results.items()}, "passes", passes)


def publish(out):
    for name in ("frozen.json", "result.json"):
        copy(out / name, PUBLIC / name)
    for key, record in read(out / "frozen.json")["records"].items():
        src, dest = out / key, PUBLIC / key
        names = ["decision.json", "rows.csv"]
        if record["new_native_export"]:
            names.extend(("direct/predictions.csv", "run/agent/final_selection.json"))
        for name in names:
            copy(src / name, dest / name)
    seal_directory(PUBLIC)


def verify(archive):
    frozen = read(PUBLIC / "frozen.json")
    for folder in (PUBLIC, ORIGINAL, REGROUPED):
        require(checksums(folder) == (folder / "SHA256SUMS").read_text(), "Source evidence differs")
    for name, src, task, _, indices, held in trials(archive):
        dest = PUBLIC / name / src.name
        with tempfile.TemporaryDirectory() as tmp:
            training = packet_for(src, task, indices, Path(tmp))
            decision = decide(src, training)
        require(decision == read(dest / "decision.json"), "Training choice cannot be reproduced")
        reconstructed = replay(src, name, rows_at(task, held), decision)
        rows = load_export(dest / "rows.csv")
        require(all(np.array_equal(v, rows[k]) for k, v in reconstructed.items()), "Frozen predictions differ")
        record = frozen["records"][name + "/" + src.name]
        require(bool(decision["approved"]) == record["new_native_export"], "Wrong native delivery type")
        native = (dest if decision["approved"] else src / RULE) / "direct/predictions.csv"
        require(sha(native) == record["native"] and np.array_equal(load_export(native)["prediction"], rows["veto_agent"]),
                "Native output differs")
    score(PUBLIC, archive)
    print("Verified ten independent vote decisions and their exact delivered predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    f = stages.add_parser("freeze")
    s = stages.add_parser("score")
    p = stages.add_parser("publish")
    v = stages.add_parser("verify")
    for cmd in (f, s, p):
        cmd.add_argument("out", type=Path)
    for cmd in (f, s, v):
        cmd.add_argument("archive", type=Path)
    for key in ("first", "second", "weights"):
        f.add_argument("--" + key, required=True, type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.out, args.archive, dict(first=args.first, second=args.second), args.weights)
    elif args.stage == "score":
        score(args.out, args.archive)
    elif args.stage == "publish":
        publish(args.out)
    else:
        verify(args.archive)
