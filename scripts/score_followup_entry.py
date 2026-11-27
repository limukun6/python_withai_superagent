"""Score sealed replications together, reusing prior scores for identical predictions."""
import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.summarize_encoder_experiments import trace
from scripts.unified_released_study import archive, read, require, save, sha, title_rows


def prediction_file(folder):
    receipt = read(folder / "sealed.json")
    path = folder / "delivery/predictions.csv"
    require(receipt["independent_reload"] and receipt["prediction_sha256"] == sha(path),
            "Prediction file differs from its independent reload")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([r["row_id"] for r in rows] == list(map(str, range(940))), "Incomplete or reordered predictions")
    predictions = np.array([int(r["prediction"]) for r in rows])
    require(np.isin(predictions,[0,1]).all(),"Invalid binary predictions")
    return receipt, predictions


def score(root, course, reference, *, include_portfolio=False, frontier_pair=False, include_resource_default=False):
    paths = {"legacy_long": root/"long-cold/original", "frontier": root/"frontier-raw/frontier",
             "frontier_scale": root/"frontier-raw/frontier_scale"}
    if include_portfolio or include_resource_default:
        paths["auto_portfolio"]=root/"auto-raw/auto_portfolio"
    if include_resource_default:
        paths["resource_default"]=root/"resource-raw/resource_default"
        launch=read(paths["resource_default"]/"launch.json")
        require(launch==dict(individual_model_variables=[],model_hint=None,review_flag=[],limits=[],
                             resource_directory_argument=True),"The ordinary entry received extra instructions")
    if frontier_pair:
        del paths["legacy_long"]
    raw = root/"raw"
    known_results = read(raw/"validation.json")
    previous = read(raw/"evaluation_lock.json")
    old = {"delivered_10_fits":reference, **{name:raw/name for name in previous["arms"]}}
    require(sha(reference/"sealed.json")==previous["baseline"] and
            all(sha(raw/name/"sealed.json")==digest for name,digest in previous["seals"].items()),
            "A previously scored run changed")
    known = {}
    outputs = {}
    for name, folder in old.items():
        seal, outputs[name] = prediction_file(folder)
        known[seal["prediction_sha256"]] = name
    previous_pair = root/"frontier_scores"
    carried = {}
    if not frontier_pair and (previous_pair/"results.json").is_file():
        pair_lock=read(previous_pair/"evaluation_lock.json")
        pair_results=read(previous_pair/"results.json")
        require(pair_results["lock_sha256"]==sha(previous_pair/"evaluation_lock.json") and
                pair_lock["reference_scores"]==sha(raw/"validation.json"),"Pair scoring history differs")
        for name,data in pair_results["completed"].items():
            require(pair_lock["seals"][name]==sha(paths[name]/"sealed.json"),"Frozen pair prediction changed")
            prior_tag="previous_batch_"+name
            seal,outputs[prior_tag]=prediction_file(paths[name])
            known.setdefault(seal["prediction_sha256"],prior_tag)
            carried[prior_tag]=data
    original_protocol = read(raw/"default/protocol.json")
    for name, folder in paths.items():
        plan = read(folder/"protocol.json")
        require(plan["features_sha256"] == original_protocol["features_sha256"] and
                plan["raw_sha256"] == original_protocol["raw_sha256"] and plan["candidate_hint"] is None,
                "Replication changed the input or supplied a model hint")
    records = {name:prediction_file(folder) for name,folder in paths.items()}
    require(all(r["api"]==0 and r["rows"]==8301 and r["fits"]<=read(paths[n]/"protocol.json")["fit_limit"]
                for n,(r,_) in records.items()),
            "Unexpected resources or partial training")
    target=root/("frontier_scores" if frontier_pair else "delivery_scores" if include_resource_default
                 else "portfolio_scores" if include_portfolio else "followup_scores")
    target.mkdir(exist_ok=True)
    lock = dict(seals={name:sha(folder/"sealed.json") for name,folder in paths.items()},
                reference_lock=sha(raw/"evaluation_lock.json"), reference_scores=sha(raw/"validation.json"))
    if carried:
        lock["prior_frontier_scores"]=sha(previous_pair/"results.json")
    save(target/"evaluation_lock.json",lock)
    if (target/"results.json").exists():
        saved=read(target/"results.json")
        require(saved["lock_sha256"]==sha(target/"evaluation_lock.json"),"Different prior scoring lock")
        return saved
    needs_labels = any(r["prediction_sha256"] not in known for r, _ in records.values())
    truth = None
    if needs_labels:
        task = archive(course,"titles")
        require(sha(task)==original_protocol["raw_sha256"],"Unexpected reference archive")
        truth = np.asarray([r[2] for r in title_rows(task,"val",labels=True)])
        require(len(truth)==940,"Wrong validation size")
    baseline = outputs["default"]
    compared = {}
    for name,(seal, values) in records.items():
        previous_name = known.get(seal["prediction_sha256"])
        if previous_name is not None:
            require(np.array_equal(values,outputs[previous_name]),"Sealed prediction identity differs")
        if previous_name in carried:
            matched=carried[previous_name]
            correct,gained,lost=(matched[k] for k in ("correct","gained_vs_default","lost_vs_default"))
        elif previous_name is not None:
            matched = known_results[previous_name]
            correct = matched["correct"]
            if np.array_equal(values,baseline):
                gained=lost=0
            else:
                pair=next(p for p in known_results["paired_comparisons"] if {p["reference"],p["candidate"]}=={"default",previous_name})
                gained,lost=(pair["fixed"],pair["lost"]) if pair["reference"]=="default" else (pair["lost"],pair["fixed"])
        else:
            current=values==truth
            incumbent=baseline==truth
            correct=int(current.sum())
            gained=int(sum(current & ~incumbent))
            lost=int(sum(~current & incumbent))
        if previous_name is not None and previous_name not in carried:
            require(known_results[previous_name]["correct"]==correct,"Reused score differs")
        compared[name]=dict(correct=correct,rows=len(values),accuracy=correct/len(values),gained_vs_default=gained,
                            lost_vs_default=lost,identical_to=previous_name,trace=trace(paths[name]))
    report=dict(lock_sha256=sha(target/"evaluation_lock.json"),new_designated_validation_read=needs_labels,
                completed=compared,official_test_opened=False,H2_opened=False,new_model_requests=0)
    save(target/"results.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("root","course","reference"):
        parser.add_argument("--"+key,type=lambda x:Path(x).resolve(),required=True)
    batch=parser.add_mutually_exclusive_group()
    batch.add_argument("--include-portfolio",action="store_true")
    batch.add_argument("--frontier-pair",action="store_true")
    batch.add_argument("--include-resource-default",action="store_true")
    args=parser.parse_args()
    result=score(args.root,args.course,args.reference,include_portfolio=args.include_portfolio,frontier_pair=args.frontier_pair,
                 include_resource_default=args.include_resource_default)
    for name,receipt in result["completed"].items():
        print(name,receipt["correct"],receipt["rows"],receipt["trace"]["members"],receipt["identical_to"])
