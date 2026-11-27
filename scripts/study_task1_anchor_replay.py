"""Reproduce the teammate's ordinary entry using the sealed, unscored prediction identity."""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from scripts.unified_released_study import env,make_zip,read,require,sha,save
from scripts.run_planner_self_review import source_identity


def run(source, archive, out, cap=10, table_weight=None, optional_runtime=None, feedback_mode=None):
    out.mkdir(parents=True,exist_ok=False)
    entry=source/"research/task1_context_agent_01"
    previous=read(entry/"sealed.json")
    expected=previous["predictions"]["agent"]
    with zipfile.ZipFile(archive) as f:
        prefix,=(n[:-len("C_train.npy")] for n in f.namelist() if n.endswith("/C_train.npy") and "__MACOSX" not in n)
        query=f.read(prefix+"C_val.npy")
    features=out/"features.zip"
    make_zip(features,[("C_test.npy",query)])
    log=out/"fits.jsonl"
    opened=out/"training_members.jsonl"
    settings=env(out/"unused",log)
    for k in list(settings):
        if k.startswith("SUPERAGENT_"):
            settings.pop(k)
    allowed=[prefix+s for s in ("C_train.npy","y_train.npy","Readme.md","info.json")]
    search_path=[source/"scripts/unified_acceptance",source]
    if optional_runtime:
        search_path.append(optional_runtime)
    if table_weight:
        from superagent.table_foundation import SHA256, WEIGHT_ENV
        require(sha(table_weight)==SHA256,"Unverified table model")
        settings[WEIGHT_ENV]=str(table_weight)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,search_path)),
                    UNIFIED_RAW_INPUT=str(archive),UNIFIED_RAW_LOG=str(opened),
                    UNIFIED_RAW_MEMBERS=json.dumps(allowed),SUPERAGENT_OFFLINE="1")
    save(out/"inputs.json",dict(source=source_identity(source),reference=sha(entry/"sealed.json"),
                                archive=sha(archive),features=sha(features),fit_cap=cap,
                                optional_asset=sha(table_weight) if table_weight else None,
                                selection_policy=feedback_mode,method_hint=None))
    train=out/"run"
    flags=["--feedback-reflection",feedback_mode] if feedback_mode else []
    if cap!=10:
        flags.extend(["--time-limit","1800"])
    for action,args in [("solve",["solve",archive,"--prediction-input",features,"--fit-limit",cap,*flags]),("replay",["predict",features])]:
        with (out/(action+".log")).open("x") as stream:
            subprocess.run([sys.executable,"-m","unified_agent",*map(str,args),"--run-dir",str(train),"--output",str(out/(action+".csv"))],
                           cwd=source,env=settings,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=2000)
    outcome=read(train/"agent/run_outcome.json")
    final=read(train/"agent/final_selection.json")
    receipt=sha(out/"solve.csv")
    times=len(log.read_text().splitlines())
    access=sorted({json.loads(s) for s in opened.read_text().splitlines()})
    require(times==outcome["attempted_classifier_fits"]<=cap and outcome["status"]=="completed"
            and sha(out/"replay.csv")==receipt and not set(access)-set(allowed) and outcome["new_api_requests"]==0,
            "Entry, cost, training inputs or independent reload mismatch")
    score=read(entry/"completion.json")["scores"]["agent"] if expected==receipt else None
    result=dict(inputs=sha(out/"inputs.json"),members=outcome["members"],historical_prediction_sha256=expected,
         local_prediction_sha256=receipt,historical_bytes_match=(expected==receipt),verified_designated_score=score,
         model=final["model_sha256"],actual_fits=times,replayed=True,opened=access,
         model_api_requests=0,designated_labels_opened=False,formal_test_opened=False)
    save(out/"result.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("source","archive","out"):
        p.add_argument("--"+n,type=lambda s:Path(s).resolve(),required=True)
    p.add_argument("--cap",type=int,choices=[10,32,48,64],default=10)
    p.add_argument("--table-weight",type=lambda s:Path(s).resolve())
    p.add_argument("--optional-runtime",type=lambda s:Path(s).resolve())
    p.add_argument("--feedback-mode",choices=["fusion-holdout","near-full-confirmation","remaining-folds","seed-replication"])
    args=p.parse_args()
    run(args.source,args.archive,args.out,args.cap,args.table_weight,args.optional_runtime,args.feedback_mode)
