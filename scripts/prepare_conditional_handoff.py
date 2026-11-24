"""Materialize disjoint training roles and fit native base learners for the Task2 routing study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.unified_released_study import EVIDENCE, ROOT, env, read, require, save, sha
from superagent.ab_complement import canonical_probabilities
from superagent.ab_methods import JOINT_PREFERENCE, PREFERENCE_CATALOG_ENV
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PUBLIC_BASE = ROOT / "research/unified_raw_1124"
PLAN = ROOT / "docs/batches/unified_conditional_handoff_plan_20261124_zh.md"
GATE = ROOT / "tools/preference_handoff.py"
ROLES = ("learn", "confirm", "outer")


def members(run):
    historical = read(PUBLIC_BASE / "third/selection.json")
    selected = read(run / "agent/final_selection.json")
    require(historical["delivered"] == selected["model_sha256"] == sha(run / "agent/selected_model.joblib") and
            historical["members"] == selected["ensemble_components"], "Frozen pair identity changed")
    arms = [*historical["members"], historical["chosen"]]
    require(len(arms) == len(set(arms)) == 3, "Expected a pair and a distinct feedback-selected expert")
    return arms, historical


def dump(path, data):
    with path.open("xb") as f:
        joblib.dump(data, f, compress=1)


def partition(groups, reserve):
    mask = np.ones(len(groups), dtype=bool)
    mask[reserve] = False
    require(set(groups[mask]).isdisjoint(groups[~mask]), "Historical reserve shares a headline component")
    unique = sorted(set(groups[mask]), key=lambda g: hashlib.sha256(("handoff-folds-v1/" + g).encode()).digest())
    buckets = {g: idx % 10 for idx, g in enumerate(unique)}
    codes = np.asarray([buckets.get(g, -1) for g in groups])
    for fold in range(5):
        offset = (codes - 2*fold) % 10
        roles = {"base": np.flatnonzero(~mask | (offset >= 5)), "learn": np.flatnonzero(mask & (offset >= 2) & (offset < 4)),
                 "confirm": np.flatnonzero(mask & (offset == 4)), "outer": np.flatnonzero(mask & (offset < 2))}
        assigned = [set(groups[ix]) for ix in roles.values()]
        require(sum(len(v) for v in roles.values()) == len(groups) and
                sum(map(len, assigned)) == len(set.union(*assigned)), "Overlapping groups across training roles")
        yield roles


def prepare(args):
    out, run, packet = args.out, args.run, args.packet
    source = read(EVIDENCE / "protocol.json")
    arms, historical = members(run)
    require(sha(packet) == source["training"]["titles_full"]["packet"], "Requires the original training-only packet")
    reserve_path = EVIDENCE / "title_reserve_ids.json"
    require(sha(reserve_path) == source["reserve_ids"], "Historical membership differs")
    task = load_training_task(packet)
    groups = sample_groups(task, "shared_text")
    reserve = np.asarray(read(reserve_path))
    raw_meta = read(run / "training_protocol.json")
    require(raw_meta["audit_row_ids_sha256"] == fingerprint(reserve.tolist()) and len(task.y_train) == 8301, "Raw run differs")
    out.mkdir(parents=True, exist_ok=False)
    unlock = out / "scoring_only"
    unlock.mkdir()
    splits = []
    assessed = []
    for fold, indices in enumerate(partition(groups, reserve)):
        folder = out / f"fold_{fold}"
        folder.mkdir()
        base = indices["base"]
        x = [task.X_train[i] for i in base]
        dump(folder / "base.joblib", dict(kind="pair_text", X_train=x, y_train=task.y_train[base], mode="refit",
             X_search=x[:1], pair_semantics="preference", metric_spec={"name":"accuracy","labels":[0,1],"direction":"maximize"}))
        panels = {role: {"pairs":[task.X_train[i] for i in indices[role]], "groups":groups[indices[role]].tolist(),
                         "ids":indices[role].tolist()} for role in ROLES}
        dump(folder / "features.joblib", panels)
        # The routing process receives labels for learning and confirmation only.
        np.savez(folder / "feedback.npz", **{k:task.y_train[indices[k]] for k in ("learn","confirm")})
        save(unlock / f"fold_{fold}.json", dict(ids=indices["outer"].tolist(),
             groups=groups[indices["outer"]].tolist(), labels=task.y_train[indices["outer"]].tolist()))
        roles = {k: {"rows":len(ix), "groups":len(set(groups[ix])), "ids":fingerprint(ix.tolist())} for k,ix in indices.items()}
        splits.append(dict(roles=roles, payload=sha(folder / "base.joblib"), features=sha(folder / "features.joblib"),
                           feedback=sha(folder / "feedback.npz"), targets=sha(unlock / f"fold_{fold}.json")))
        assessed.extend(indices["outer"].tolist())
    require(sorted(assessed + reserve.tolist()) == list(range(len(task.y_train))), "Outer folds must cover each new row once")
    save(out / "protocol.json", dict(plan=sha(PLAN), gate=sha(GATE), prepare=sha(Path(__file__)), members=arms,
         source_model=historical["delivered"], source_training_packet=sha(packet), source_selection=sha(PUBLIC_BASE/"third/selection.json"),
         common_entry=sha(ROOT/"unified_agent.py"), preexisting_audit_only_in_base=len(reserve), splits=splits,
         outer_rows=len(assessed), base_fits_per_fold=3, maximum_gate_fits_per_fold=3, model_api_limit=0))
    print("Frozen five disjoint outer folds:", [s["roles"]["outer"]["rows"] for s in splits], flush=True)


def offline(weights, fitted):
    ep = env(weights, fitted)
    ep[PREFERENCE_CATALOG_ENV] = JOINT_PREFERENCE
    return ep


def fit_fold(args, fold):
    protocol = read(args.out / "protocol.json")
    folder = args.out / f"fold_{fold}"
    expected, payload = protocol["splits"][fold], folder/"base.joblib"
    require(sha(payload)==expected["payload"] and sha(Path(__file__))==protocol["prepare"], "Base payload or runner changed")
    line_log = folder / "physical.jsonl"
    environment = offline(args.weights, line_log)
    models = []
    for i, name in enumerate(protocol["members"]):
        file = folder / f"member_{i}.joblib"
        ledger = folder / f"member_{i}.fits.json"
        admission = folder / f"member_{i}.admission.json"
        if not ledger.exists() and not file.exists():
            save(admission, dict(maximum_classifier_fits=1, ledger_path=str(ledger)))
            with (folder / f"member_{i}.log").open("x") as log:
                proc = subprocess.run([sys.executable,"-m","superagent.experiment_worker",str(payload),name,str(file),
                                      "--fit-admission",str(admission)], cwd=ROOT, env=environment,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=330)
            require(proc.returncode==0, f"Native worker failed on fold {fold}, member {i}; see its durable ledger")
        receipt = read(ledger)
        require(file.is_file() and receipt["candidate"]==name and receipt["status"]=="completed"
                and receipt["attempted_classifier_fits"]==receipt["completed_classifier_fits"]==1, "Unfinished native learner")
        models.append(dict(name=name, model=sha(file), fits=1, observed_classifier=receipt["calls"][0]["classifier"]))
    physical = [json.loads(line) for line in line_log.read_text().splitlines()]
    require(physical == [{"candidate":r["name"],"classifier":r["observed_classifier"]} for r in models], "Physical fit ledger differs")
    save(folder / "models_frozen.json", dict(models=models, physical=physical))
    if not (folder/"scores_frozen.json").exists():
        with (folder/"inference.log").open("x") as log:
            done = subprocess.run([sys.executable,"-m","scripts.prepare_conditional_handoff","infer","--folds",str(fold),
                 "--out",str(args.out)], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=330)
        require(done.returncode==0, f"Fold {fold} inference failed")
    print(f"Fold {fold}: all three native fits complete; feature-only probabilities are sealed.",flush=True)


def infer_fold(out, fold):
    protocol=read(out/"protocol.json")
    folder=out/f"fold_{fold}"
    require(sha(folder/"features.joblib")==protocol["splits"][fold]["features"],"Wrong feature panel")
    panels=joblib.load(folder/"features.joblib")
    pairs=[pair for role in ROLES for pair in panels[role]["pairs"]]
    records=read(folder/"models_frozen.json")["models"]
    columns=[]
    for i, record in enumerate(records):
        path=folder/f"member_{i}.joblib"
        require(sha(path)==record["model"],"Model changed before inference")
        columns.append(canonical_probabilities(joblib.load(path)["model"], pairs)[1][:,1])
    matrix=np.column_stack(columns)
    offsets=np.cumsum([0]+[len(panels[k]["pairs"]) for k in ROLES])
    with (folder/"probabilities.npz").open("xb") as f:
        np.savez(f, **{k:matrix[offsets[i]:offsets[i+1]] for i,k in enumerate(ROLES)})
    save(folder/"scores_frozen.json",dict(scores=sha(folder/"probabilities.npz"), models=sha(folder/"models_frozen.json"),
                                         features=sha(folder/"features.joblib"), order=list(ROLES)))


def historical_scores(args):
    run, out=args.run, args.out
    arms, record=members(run)
    source=read(out/"protocol.json")
    require(arms==source["members"],"Historical expert set changed")
    path=run/"audit_features.zip"
    pairs=load_prediction_features(path,"pair_text",read(run/"agent/task_spec.json"))
    finished=joblib.load(run/"agent/selected_model.joblib")["model"]
    other=record["trial_artifacts"][arms[-1]]
    model=run/"agent"/f"trial_{other['trial']}.joblib"
    require(sha(model)==other["artifact"],"Previously selected expert changed")
    objects=[*finished.models, joblib.load(model)["model"]]
    data=np.column_stack([canonical_probabilities(obj,pairs)[1][:,1] for obj in objects])
    with (out/"historical_probabilities.npy").open("xb") as stream:
        np.save(stream,data,allow_pickle=False)
    save(out/"historical_inputs.json",dict(probabilities=sha(out/"historical_probabilities.npy"),
         features=sha(path), trial=other["artifact"], delivered=source["source_model"], rows=len(pairs)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare","fit","infer","historical"))
    for k in ("out","packet","run","weights"):
        parser.add_argument("--"+k, required=k=="out", type=lambda s:Path(s).resolve())
    parser.add_argument("--folds",type=int,nargs="+")
    a=parser.parse_args()
    if a.stage=="prepare":
        prepare(a)
    elif a.stage=="historical":
        historical_scores(a)
    else:
        for n in a.folds:
            fit_fold(a,n) if a.stage=="fit" else infer_fold(a.out,n)
