"""Freeze, then score, a two-stage selection replay using only released training rows."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

from scripts.fit_portfolio_fold import PRIOR
from scripts.prepare_conditional_handoff import partition
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_encoder_fusion import save_csv
from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha
from scripts.verify_encoder_portfolio import PUBLIC
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.feedback_policy_pool import pooled_probability
from tools.preference_handoff import binary
from tools.preference_portfolio import panel, probabilities
from tools.split_confirmation import choose_separately, feedback_halves

PLAN = ROOT/"docs/batches/lifecycle_audit_plan_20261124_zh.md"
ROUNDS = {"first":("learn","outer"), "rotated":("outer","learn")}
POLICIES = ("baseline","single","pair","constant","contextual")


def training_rows(archive):
    with zipfile.ZipFile(archive) as z:
        payload = z.read("Released_Task2-Headlines_CTR/train.csv")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"),newline="")))
    pairs = [(r["headline_A"],r["headline_B"]) for r in rows]
    y = np.array([int(r["label_pairwise"]) for r in rows])
    groups = sample_groups(SimpleNamespace(kind="pair_text",X_train=pairs),"shared_text")
    return pairs, y, groups, hashlib.sha256(payload).hexdigest()


def freeze(archive, out):
    out.mkdir(parents=True, exist_ok=True)
    pairs, y, groups, source = training_rows(archive)
    previous = read(PRIOR/"protocol.json")
    reserves = np.asarray(read(EVIDENCE/"title_reserve_ids.json"))
    manifest = {line[66:]:line[:64] for line in (PUBLIC/"SHA256SUMS").read_text().splitlines()}
    seals = {}
    for fold, ids in enumerate(partition(groups,reserves)):
        prefix = f"fold_{fold}/"
        scores_file = PUBLIC/prefix/"all_probabilities.npz"
        require(sha(scores_file) == manifest[prefix+"all_probabilities.npz"], "Frozen model probabilities changed")
        data = {}
        with np.load(scores_file,allow_pickle=False) as scores:
            for role in ("learn","confirm","outer"):
                require(fingerprint(ids[role].tolist()) == previous["splits"][fold]["roles"][role]["ids"],
                        "Unexpected feedback row order")
                data[role] = panel(scores[role], [pairs[j] for j in ids[role]])
        for round_name,(learn,outer) in ROUNDS.items():
            path = prefix+("rotation/" if round_name=="rotated" else "")+"decision.json"
            require(sha(PUBLIC/path) == manifest[path], "Frozen learning rule changed")
            saved = read(PUBLIC/path)
            rules = saved["choice"]["rules"]
            def columns(role):
                return {k:probabilities(rules[k],data[role]) for k in POLICIES}
            confirm = ids["confirm"]
            decision = choose_separately(columns(learn),y[ids[learn]],data[learn]["ties"],columns("confirm"),
                                         y[confirm],data["confirm"]["ties"],groups[confirm])
            first,second = feedback_halves(groups[confirm])
            decision["confirm_rows_sha256"] = [fingerprint(confirm[mask].tolist()) for mask in (first,second)]
            output = out/f"{round_name}_{fold}"
            output.mkdir(exist_ok=True)
            save(output/"decision.json", decision)
            held = data[outer]
            predictions = columns(outer)
            old_max = predictions[saved["choice"]["chosen"]]
            old = pooled_probability(saved["pool"]["deployed_weights"],rules,held) if round_name=="rotated" else old_max
            choices = {"baseline":predictions["baseline"], "previous_max":old_max, "previous_deployed":old,
                       "nominee":predictions[decision["nominee"]], "agent":predictions[decision["selected"]]}
            labels = {n:binary(p,held["ties"]) for n,p in choices.items()}
            export = output/"predictions.csv"
            if not export.exists():
                save_csv(export, ids[outer], labels)
            stored = load_export(export)
            require(np.array_equal(stored["row_id"],ids[outer]) and all(np.array_equal(stored[n],v) for n,v in labels.items()),
                    "Export differs from locked choice")
            seals[output.name] = {"decision":sha(output/"decision.json"),"predictions":sha(export),
                                  "stored_rules":manifest[path],"stored_probabilities":sha(scores_file)}
    save(out/"sealed.json",dict(plan=sha(PLAN),policy=sha(ROOT/"tools/split_confirmation.py"),
         runner=sha(Path(__file__)), source_training_member=source, experiments=seals))


def score(archive, out):
    seal = read(out/"sealed.json")
    _, y, groups, source = training_rows(archive)
    require(seal["plan"] == sha(PLAN) and seal["policy"] == sha(ROOT/"tools/split_confirmation.py")
            and seal["runner"] == sha(Path(__file__)) and source == seal["source_training_member"],"Unsealed policy or source")
    reserved = np.asarray(read(EVIDENCE/"title_reserve_ids.json"))
    roles = list(partition(groups,reserved))
    results = {}
    for name,(_,outer) in ROUNDS.items():
        columns, folds, seen_groups = {}, [], set()
        for fold in range(5):
            folder = out/f"{name}_{fold}"
            record = seal["experiments"][folder.name]
            require(sha(folder/"predictions.csv")==record["predictions"] and sha(folder/"decision.json")==record["decision"],
                    "Frozen prediction changed")
            data = load_export(folder/"predictions.csv")
            ids = data["row_id"]
            require(np.array_equal(ids,roles[fold][outer]) and not seen_groups.intersection(groups[ids]),"Repeated evaluation groups")
            seen_groups.update(groups[ids])
            decision = read(folder/"decision.json")
            baseline = data["baseline"]
            folds.append(dict(fold=fold,rows=len(ids),nominee=decision["nominee"],selected=decision["selected"],
                net={k:int((a==y[ids]).sum()-(baseline==y[ids]).sum()) for k,a in data.items() if k!="row_id"}))
            for k,a in data.items():
                columns.setdefault(k,[]).append(a)
        data = {k:np.concatenate(v) for k,v in columns.items()}
        ids = data.pop("row_id")
        require(len(set(ids)) == len(ids) == 6608, "Wrong number of frozen evaluation rows")
        arms = {k:dict(correct=int((v==y[ids]).sum()), **comparison(data["baseline"],v,y[ids],groups[ids])) for k,v in data.items()}
        wins = sum(f["net"]["agent"]>0 for f in folds)
        result = arms["agent"]
        results[name] = dict(rows=len(ids),groups=len(seen_groups),folds=folds,arms=arms,
                             positive_folds=wins,passes=wins>=4 and result["net"]>=5 and result["paired_95_fraction"][0]>0)
    save(out/"result.json",dict(frozen=sha(out/"sealed.json"),rounds=results,new_classifier_fits=0,
         new_continuous_fits=0,nomination_and_gate_evaluations=10,model_api_calls=0))
    for name,r in results.items():
        print(name,{k:(v["correct"],v["net"]) for k,v in r["arms"].items()},r["positive_folds"],r["passes"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","score"))
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args = p.parse_args()
    (freeze if args.stage=="freeze" else score)(args.archive,args.out)
