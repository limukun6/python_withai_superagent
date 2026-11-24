"""Reconstruct the lifecycle audit from training inputs, synthetic labels and frozen predictions."""
import argparse
import csv
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
from sklearn.metrics import f1_score

from scripts.audit_anonymous_goal import prepare
from scripts.audit_confirmation_coverage import reconstruct
from scripts.probe_portfolio_availability import CASES, EXTRA, probe
from scripts.study_split_confirmation import freeze, score
from scripts.unified_released_study import ROOT, archive, read, require, save, sha
from scripts.verify_unified_raw import MEMBERS

PUBLIC = ROOT/"research/lifecycle_audit_1124"


def copy(source, dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        require(sha(dest)==sha(source),"Refusing to replace changed evidence")
    else:
        shutil.copyfile(source,dest)


def publish(run):
    for name in ("confirmation_trace.json","confirmation_coverage.json","assets_before.json","assets_after.json"):
        copy(run/name,PUBLIC/name)
    copy(run/"selection/sealed.json",PUBLIC/"selection_first_seal.json")
    for p in (run/"selection_checked").rglob("*"):
        if p.is_file() and p.name in ("result.json","predictions.csv","decision.json","sealed.json"):
            copy(p,PUBLIC/"selection"/p.relative_to(run/"selection_checked"))
    for arm in ("before","after"):
        src, dst = run/arm, PUBLIC/"goal"/arm
        for name in ("result.json","frozen.json","prediction_0/predictions.csv","prediction_1/predictions.csv"):
            copy(src/name,dst/name)
        copy(src/"run/agent/classifier_fits.json",dst/"classifier_fits.json")
        history = read(src/"run/agent/history.json")
        save(dst/"trials.json", [{k:r[k] for k in ("trial","split_id","candidate","primary_metric","status")} for r in history])
    sums = "".join(f"{sha(p)}  {p.relative_to(PUBLIC).as_posix()}\n" for p in sorted(PUBLIC.rglob("*")) if p.is_file() and p.name!="SHA256SUMS")
    target = PUBLIC/"SHA256SUMS"
    if target.exists():
        require(target.read_text()==sums,"Prior publication differs")
    else:
        target.write_text(sums)


def verify(packages, weights=None):
    expected = {line[66:]:line[:64] for line in (PUBLIC/"SHA256SUMS").read_text().splitlines()}
    require(expected=={p.relative_to(PUBLIC).as_posix():sha(p) for p in PUBLIC.rglob("*") if p.is_file() and p.name!="SHA256SUMS"},
            "Public evidence differs")
    reread = {}
    original = zipfile.ZipFile.open
    for tag, accepted_members in MEMBERS.items():
        reread[archive(packages,tag).resolve()] = (set(accepted_members),set())
    def training_only(z, member, *args, **kwargs):
        archive_path = Path(z.filename).resolve() if isinstance(z.filename,(Path,str)) else None
        if archive_path in reread:
            allowed,seen = reread[archive_path]
            name = member.filename if isinstance(member,zipfile.ZipInfo) else member
            require(name in allowed,"Attempted to read a nontraining archive member")
            seen.add(name)
        return original(z, member, *args, **kwargs)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture = root/"fixture"
        prepare(fixture)
        y = read(fixture/"audit_truth.json")
        for label in ("before","after"):
            arm = PUBLIC/"goal"/label
            result = read(arm/"result.json")
            frozen = read(arm/"frozen.json")
            a,b = [arm/f"prediction_{n}"/"predictions.csv" for n in (0,1)]
            require(frozen==result["frozen"] and frozen["train_package"]==sha(fixture/"train.zip")
                    and frozen["features"]==sha(fixture/"features.zip")
                    and sha(a)==sha(b)==frozen["prediction"],"Anonymous package or frozen prediction changed")
            with a.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            require([int(r["row_id"]) for r in rows]==list(range(len(y))),"Anonymous rows differ")
            predicted = np.array([int(r["prediction"]) for r in rows])
            require(int((predicted==y).sum())==result["audit_correct"] and
                    np.isclose(f1_score(y,predicted,average="macro"),result["audit_macro_f1"]),"Anonymous macro-F1 differs")
            ledger = read(arm/"classifier_fits.json")
            require(result["fits"]==ledger["attempted_classifier_fits"]==ledger["completed_classifier_fits"]==
                    sum(len(w["calls"]) for w in ledger["workers"])<=10 and result["api"]==0,"Anonymous fit or API budget differs")
            require({t["primary_metric"] for t in read(arm/"trials.json") if t["status"]=="completed"}=={result["actual_metric"]},
                    "A trial optimized a different goal")
        with patch.object(zipfile.ZipFile,"open",training_only):
            cover = reconstruct(packages,read(PUBLIC/"confirmation_trace.json"))
            require(cover==read(PUBLIC/"confirmation_coverage.json"),"Evaluation overlap differs")
            out = root/"decisions"
            freeze(archive(packages,"titles"),out)
            score(archive(packages,"titles"),out)
        for path in out.rglob("*"):
            if path.is_file():
                require(sha(path)==sha(PUBLIC/"selection"/path.relative_to(out)),"Recomputed selection evidence differs")
        initial, checked = read(PUBLIC/"selection_first_seal.json"), read(out/"sealed.json")
        require({k:v for k,v in initial.items() if k!="runner"}=={k:v for k,v in checked.items() if k!="runner"},
                "Prediction or policy changed after initial freeze")
    require(all(allowed==seen for allowed,seen in reread.values()),"Unexpected course data access")
    rows = read(PUBLIC/"assets_after.json")
    require(all(rows[k]["passed"] and len(rows[k]["candidates"])==count for k,count in CASES.items())
            and all(n in rows["optional_incomplete"]["candidates"] for n in EXTRA[1:])
            and not read(PUBLIC/"assets_before.json")["optional_incomplete"]["passed"],"Asset fault evidence differs")
    if weights is not None:
        require(probe(ROOT, weights)==rows,"Live eligibility checks differ from the saved results")
    print("Verified both anonymous paths, 19 recorded fits, two historical split traces and ten locked decisions.")
    print("Course members read:", {p.name:sorted(seen) for p,(_,seen) in reread.items()})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    cmds=p.add_subparsers(dest="stage",required=True)
    pub=cmds.add_parser("publish")
    pub.add_argument("run",type=Path)
    check=cmds.add_parser("verify")
    check.add_argument("packages",type=Path)
    check.add_argument("--weights",type=Path)
    args=p.parse_args()
    if args.stage=="publish":
        publish(args.run)
    else:
        verify(args.packages,args.weights)
