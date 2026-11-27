"""Relocate and replay a sealed follow-up using its own frozen runtime and resource inventory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import read,require,save,sha
from superagent.ab_sentence import ALTERNATIVES,DEEP_ENV,DEEP_MODEL,MODEL


def make(args):
    path={"legacy_long":"long-cold/original","frontier":"frontier-raw/frontier",
          "frontier_scale":"frontier-raw/frontier_scale","auto_portfolio":"auto-raw/auto_portfolio"}[args.arm]
    run=args.root/path
    plan=read(run/"protocol.json")
    sealed=read(run/"sealed.json")
    evidence=args.root/args.evaluation
    lock=read(evidence/"evaluation_lock.json")
    scored=read(evidence/"results.json")
    require(sha(evidence/"evaluation_lock.json")==scored["lock_sha256"] and
            lock["seals"][args.arm]==sha(run/"sealed.json"),"Score is not bound to this run")
    require(source_identity(args.source)==plan["runtime_sha256"] and sha(args.features)==plan["features_sha256"] and
            sha(run/"delivery/predictions.csv")==sealed["prediction_sha256"] and sealed["independent_reload"],
            "Runtime, features or predictions differ")
    measure=scored["completed"][args.arm]
    env=dict(os.environ)
    models={s["env"]:s["model"] for s in ALTERNATIVES.values()}
    models.update({DEEP_ENV:DEEP_MODEL,"SUPERAGENT_MINILM_WEIGHTS":MODEL})
    for key,model in models.items():
        where=args.weights/model.split("/")[-1]
        if where.is_dir():
            env[key]=str(where)
    args.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output) as stage:
        stage=Path(stage)
        titles=stage/"titles"
        titles.mkdir()
        for name in ("delivery","physical.jsonl","run"):
            (titles/name).symlink_to(run/name)
        (titles/"features.zip").symlink_to(args.features)
        save(stage/"protocol.json",dict(original_run=plan,score_lock=lock,packages=dict(titles=dict(features=sha(args.features)))))
        save(stage/"validation.json",dict(rows=measure["rows"],correct=measure["correct"],accuracy=measure["accuracy"],
             runtime_sha256=plan["runtime_sha256"],prediction_sha256=sealed["prediction_sha256"],model_sha256=sealed["model_sha256"],
             independent_reload=True,source=args.arm,fits=sealed["fits"],official_test_accessed=False))
        subprocess.run([sys.executable,"-m","tools.package_verified_delivery","--run",str(stage),"--notices",str(args.notices),
                        "--output",str(args.output)],stdin=subprocess.DEVNULL,cwd=args.source,env=env,check=True,timeout=900)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("legacy_long","frontier","frontier_scale","auto_portfolio"),required=True)
    p.add_argument("--evaluation",choices=("followup_scores","portfolio_scores"),default="portfolio_scores")
    for name in ("root","features","weights","notices","source","output"):
        p.add_argument("--"+name,type=lambda x:Path(x).resolve(),required=True)
    make(p.parse_args())
