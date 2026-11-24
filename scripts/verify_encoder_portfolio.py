"""Publish portable numerical evidence and verify every decision and score against training rows only."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from scripts.fit_portfolio_fold import PLAN, PRIOR
from scripts.prepare_conditional_handoff import partition
from scripts.rotate_encoder_feedback import PLAN as ROTATION_PLAN
from scripts.run_conditional_handoff import comparison, load_export
from scripts.study_encoder_fusion import NAMES, measures
from scripts.unified_encoder_portfolio import training_roles
from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha, title_rows
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.encoder_assets import MODELS
from tools.feedback_policy_pool import evidence_pool, pooled_probability
from tools.preference_handoff import binary
from tools.preference_portfolio import designs, evaluate_rule, outputs, panel, probabilities, shortlist, weights


PUBLIC = ROOT/"research/unified_encoder_fusion_1124"
SOURCES = ("tools/encoder_assets.py","tools/portfolio_encoders.py","tools/preference_portfolio.py","tools/portfolio_training.py",
           "tools/feedback_policy_pool.py","scripts/fit_portfolio_fold.py","scripts/study_encoder_fusion.py",
           "scripts/unified_encoder_portfolio.py","scripts/rotate_encoder_feedback.py","scripts/check_portfolio_cold.py",
           "scripts/check_portfolio_features.py")


def copy_one(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists() or sha(destination) == sha(source), "Published asset has changed")
    shutil.copyfile(source, destination)


def publish(study, cold):
    for name in ("result.json","frozen.json","rotation_result.json","rotation_frozen.json","feature_reload.json"):
        copy_one(study/name, PUBLIC/name)
    for i in range(5):
        f = f"fold_{i}"
        files = ["decision.json","sealed.json","routing_admission.json","rows.csv","all_probabilities.npz",
                 "rotation/admission.json","rotation/decision.json","rotation/sealed.json","rotation/rows_0.csv"]
        for name in MODELS:
            files.extend(name+"/"+n for n in ("fits.json","scores.json","trained.json"))
        for name in files:
            copy_one(study/f/name, PUBLIC/f/name)
    for name in ("protocol.json","decision.json","completion.json","routing_admission.json","learn_scores.npy","confirm_scores.npy",
                 "audit_seal.json","audit_result.json","cached/predictions.csv","reference_export/predictions.csv"):
        copy_one(cold/name, PUBLIC/"cold"/name)
    for m in NAMES:
        copy_one(cold/"members"/m/"fits.json",PUBLIC/"cold/members"/m/"fits.json")
    save(PUBLIC/"protocol.json", {"registered":sha(PLAN), "rotation":sha(ROTATION_PLAN),
         "cold_plan":sha(ROOT/"docs/batches/unified_encoder_entry_plan_20261124_zh.md"),
         "previous_folds":sha(PRIOR/"protocol.json"), "encoders":json.loads(json.dumps(MODELS)),
         "source_digests":{p:sha(ROOT/p) for p in SOURCES}})
    targets = sorted(p for p in PUBLIC.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (PUBLIC/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(PUBLIC).as_posix()}\n" for p in targets))


def check_choice(choice, train, labels, confirm, y_confirm):
    rules = choice["rules"]
    require(choice["members"] == NAMES and rules["single"] == shortlist(train,labels,1)
            and rules["pair"] == shortlist(train,labels,2), "Learning shortlist differs")
    for key in ("constant","contextual"):
        r = rules[key]
        w = weights(r, train)
        x = designs(train,r)
        p = train["scores"]
        q = np.clip(probabilities(r, train),1e-9,1-1e-9)
        penalty = np.full((1,x.shape[2]),.02)
        penalty[0,0] = .004
        slope = ((q-labels)/(q*(1-q)))[:,None] * w * (p-q[:,None])
        grad = np.einsum("im,imk->mk",slope,x)/len(q) + penalty*np.asarray(r["coefficients"])
        require(np.abs(grad).max() < 1.5e-5, "Saved fusion weights do not optimize the registered training loss")
    minimum = max(2,math.ceil(.002*len(y_confirm)))
    best, current = (0,0), "baseline"
    for name in ("baseline","single","pair","constant","contextual"):
        rule = rules[name]
        a, b = evaluate_rule(rule,train,labels), evaluate_rule(rule,confirm,y_confirm)
        eligible = a["net"]>0 and b["net"]>=minimum
        require(choice["evidence"][name] == {"learning":a,"confirmation":b,"eligible":eligible}, "Training evidence differs")
        rank = (b["net"],-b["switches"])
        if eligible and rank>best:
            best, current = rank, name
    require(choice["chosen"] == current and minimum == choice["minimum_confirmation_gain"], "Recorded decision differs")


def native_digest(values):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["row_id","prediction"])
    writer.writerows(enumerate(map(int, values)))
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


def check_ledger(path, name, digest):
    item = read(path)
    require(sha(path)==digest and item["candidate"]==name and item["attempted_classifier_fits"]==
            item["completed_classifier_fits"]==item["maximum_classifier_fits"]==1 and item["status"]=="completed"
            and item["calls"]==[{"classifier":"LogisticRegression","status":"completed"}],"Unbound classifier fit")


def same_record(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(same_record(a[k],b[k]) for k in a)
    if isinstance(a, float):
        return isinstance(b, (float,int)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12)
    return a==b


def verify(archive):
    lines = (PUBLIC/"SHA256SUMS").read_text().splitlines()
    expect = {s[66:]:s[:64] for s in lines}
    require(expect == {p.relative_to(PUBLIC).as_posix():sha(p) for p in PUBLIC.rglob("*")
                       if p.is_file() and p.name!="SHA256SUMS"}, "Artifact checksum mismatch")
    protocol = read(PUBLIC/"protocol.json")
    require(protocol["registered"]==sha(PLAN) and protocol["rotation"]==sha(ROTATION_PLAN)
            and protocol["previous_folds"]==sha(PRIOR/"protocol.json")
            and protocol["source_digests"]=={p:sha(ROOT/p) for p in SOURCES}
            and protocol["cold_plan"]==sha(ROOT/"docs/batches/unified_encoder_entry_plan_20261124_zh.md")
            and protocol["encoders"]==json.loads(json.dumps(MODELS)), "Experiment or model identity changed")
    require(sha(archive)==read(EVIDENCE/"protocol.json")["archives"]["titles"],"Unexpected released archive")
    raw = title_rows(archive,"train",labels=True)
    y = np.asarray([r[2] for r in raw])
    pairs = [r[:2] for r in raw]
    groups = sample_groups(SimpleNamespace(kind="pair_text",X_train=pairs),"shared_text")
    reserved = np.asarray(read(EVIDENCE/"title_reserve_ids.json"))
    main, rotation = read(PUBLIC/"result.json"), read(PUBLIC/"rotation_result.json")
    totals = [{"ids":[],"predictions":{}} for _ in range(2)]
    prev = read(PRIOR/"protocol.json")
    seals = [read(PUBLIC/name) for name in ("frozen.json","rotation_frozen.json")]
    for i, roles in enumerate(partition(groups,reserved)):
        f = PUBLIC/f"fold_{i}"
        basis = prev["splits"][i]
        seal = read(f/"sealed.json")
        require(seal==seals[0][i] and seal["runner"]==sha(ROOT/"scripts/study_encoder_fusion.py")
                and seal["selection"]==sha(f/"decision.json") and seal["probabilities"]==sha(f/"all_probabilities.npz")
                and seal["predictions"]==sha(f/"rows.csv"),"First predictions or decision changed")
        with np.load(f/"all_probabilities.npz",allow_pickle=False) as data, np.load(PRIOR/f"fold_{i}/probabilities.npz",allow_pickle=False) as reference:
            d = {}
            for role in ("learn","confirm","outer"):
                ids = roles[role]
                require(fingerprint(ids.tolist())==basis["roles"][role]["ids"]
                        and np.array_equal(data[role][:,:3],reference[role]), "Different base models, group membership or row order")
                d[role] = panel(data[role],[pairs[j] for j in ids])
        for name in MODELS:
            rec = read(f/name/"scores.json")
            trained = read(f/name/"trained.json")
            check_ledger(f/name/"fits.json",name,rec["ledger"])
            require(rec["identity"]["base"]==basis["payload"] and rec["identity"]==trained["identity"]
                    and rec["model"]==trained["model"] and rec["identity"]["plan"]==sha(PLAN), "New head used another training role")
        for j,(learn,outer,stored) in enumerate((("learn","outer",main),("outer","learn",rotation))):
            loc = f if j==0 else f/"rotation"
            selected = read(loc/"decision.json")
            choice = selected["choice"]
            check_choice(choice,d[learn],y[roles[learn]],d["confirm"],y[roles["confirm"]])
            measured = outputs(choice,d[outer])
            csv_path = loc/("rows.csv" if j==0 else "rows_0.csv")
            if j:
                rs = read(loc/"sealed.json")
                require(rs==seals[1][i] and rs["selection"]==sha(loc/"decision.json")
                        and rs["rows"]==[sha(csv_path)]*2,"Rotated prediction freeze differs")
                checked = evidence_pool(choice,d[learn],y[roles[learn]],groups[roles[learn]],d["confirm"],
                                        y[roles["confirm"]],groups[roles["confirm"]])
                require(same_record(checked,selected["pool"]),"Continuous feedback pooling cannot be reproduced")
                measured["original_agent"] = measured["agent"]
                for arm in ("proposed","deployed"):
                    measured[arm] = binary(pooled_probability(checked[arm+"_weights"],choice["rules"],d[outer]),d[outer]["ties"])
                measured["agent"] = measured["deployed"]
            else:
                require(seal["native"]==native_digest(measured["agent"]), "Raw-input native export disagrees")
            actual = load_export(csv_path)
            ix = roles[outer]
            require(actual["row_id"].tolist()==ix.tolist() and all(np.array_equal(actual[n],v) for n,v in measured.items()),
                    "Saved row predictions differ from the training-feedback decision")
            result = measures(actual,y[ix],groups[ix])
            require(all(stored["folds"][i][k]==v for k,v in result.items()), "Per-fold scoring or uncertainty differs")
            totals[j]["ids"].extend(ix)
            for n,v in actual.items():
                totals[j]["predictions"].setdefault(n,[]).append(v)
    for stored, rows in zip((main,rotation),totals,strict=True):
        ix = np.asarray(rows["ids"])
        cols = {k:np.concatenate(v) for k,v in rows["predictions"].items()}
        require(len(set(ix))==len(ix)==6608 and set(ix).isdisjoint(reserved), "Repeated or missing evaluation group")
        pooled = measures(cols,y[ix],groups[ix])
        wins = sum(f["methods"]["agent"]["net"]>0 for f in stored["folds"])
        a = pooled["methods"]["agent"]
        require(pooled==stored["pooled"] and wins==stored["positive_folds"]
                and stored["passes"]==bool(wins>=4 and a["net"]>=5 and a["paired_95_fraction"][0]>0), "Pooled result differs")

    cold = PUBLIC/"cold"
    cp, end, decision = (read(cold/n) for n in ("protocol.json","completion.json","decision.json"))
    allow = np.flatnonzero(~np.isin(np.arange(len(y)),reserved))
    cold_roles = training_roles(groups,y,allow)
    for k,ix in {**cold_roles,"audit":reserved}.items():
        require(fingerprint(ix.tolist())==cp["roles"][k]["ids"],"Anonymous native split differs")
    d = {k:panel(np.load(cold/(k+"_scores.npy"),allow_pickle=False),[pairs[j] for j in cold_roles[k]]) for k in ("learn","confirm")}
    check_choice(decision,d["learn"],y[cold_roles["learn"]],d["confirm"],y[cold_roles["confirm"]])
    for c in end["members"]:
        check_ledger(cold/"members"/c["name"]/"fits.json",c["name"],c["ledger"])
    base, current = (load_export(cold/k/"predictions.csv") for k in ("reference_export","cached"))
    audit, frozen = read(cold/"audit_result.json"), read(cold/"audit_seal.json")
    require(np.array_equal(current["row_id"],np.arange(len(reserved))) and np.array_equal(base["row_id"],current["row_id"])
            and native_digest(current["prediction"])==frozen["deployed_predictions"][0]==frozen["deployed_predictions"][1]
            and native_digest(base["prediction"])==frozen["reference"] and audit["seal"]==sha(cold/"audit_seal.json")
            and frozen["decision"]==sha(cold/"decision.json") and end["selected"]==decision["chosen"]==audit["chosen"]
            and all(audit[k]==v for k,v in comparison(base["prediction"],current["prediction"],y[reserved],groups[reserved]).items()),
            "Cold deployment or its paired score differs")
    require((main["new_classifier_fits"],main["new_continuous_fits"],main["new_discrete_selections"])==(15,10,10)
            and (end["classifier_fits"],end["continuous_fits"],end["discrete_selections"])==(6,2,2)
            and (rotation["added_base_fits"],rotation["continuous_fits"],rotation["discrete_selections"],rotation["evidence_pools"])==(0,10,10,5)
            and all(item["model_api_calls"]==0 for item in (main,rotation,end)), "Reported resource counts differ")
    if audit["chosen"]=="constant":
        actual_weights=weights(decision["rules"]["constant"],{"scores":np.zeros((1,len(NAMES)))})[0]
        require(np.allclose(audit["fixed_weights"],actual_weights,atol=1e-12), "Reported cold ensemble weights differ")
    print("Verified both 6,608-row grouped evaluations, the 1,693-row matched anonymous audit, all 21 new physical heads, "
          "and the saved fusion choices; read train.csv only, with zero new classifier or weight fits.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--publish",type=Path)
    p.add_argument("--cold",type=Path)
    a=p.parse_args()
    if a.publish:
        if a.cold is None:
            p.error("--publish requires --cold")
        publish(a.publish,a.cold)
    verify(a.archive)
