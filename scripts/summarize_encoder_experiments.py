"""Summarize actual autonomous decisions and separate cold, reused, and research-only fits."""
import argparse
from pathlib import Path

from scripts.unified_released_study import read,require,save,sha


def trace(folder):
    agent=folder/"run/agent"
    seal=read(folder/"sealed.json")
    outcomes=read(agent/"run_outcome.json")
    history=read(agent/"history.json")
    decisions=read(agent/"decisions.json")
    for i,row in enumerate(history):
        step=decisions[i]
        require(step["observed_trials"]==[r["trial"] for r in history[:i]] and step["choice"]["members"]==[row["candidate"]]
                and step["choice"]["split"]==row["split_id"],"History and prospective decision do not match")
    completed=outcomes["status"]=="completed"
    if completed:
        require(decisions[-1]["choice"]["action"]=="stop" and decisions[-1]["choice"]["members"]==seal["members"],
                "The delivered model differs from the agent's final action")
    evidence=dict(status=outcomes["status"],members=seal["members"],physical_fits=seal["fits"],training_rows=seal["rows"],
                  model_requests=seal["api"],training_members=seal["opened"],independent_uncached_reload=seal["independent_reload"],
                  raw_predictions_sha256=seal["prediction_sha256"],raw_model_sha256=seal["model_sha256"],
                  first_menu_size=len(decisions[0]["alternatives"]),
                  generated_methods_actually_trained=sorted({r["candidate"] for r in history if r["candidate"].startswith("open_")}),
                  choices=[dict(step=row["trial"],action=decisions[i]["choice"]["action"],member=row["candidate"],
                                fold=row["split_id"],status=row["status"],score=row.get("search_score")) for i,row in enumerate(history)],
                  full_refit=read(agent/"final_selection.json")["refit"])
    if (agent/"finalist_lock.json").exists():
        evidence.update(finalists_frozen=read(agent/"finalist_lock.json"),matched_result=read(agent/"finalist_result.json"))
    return evidence


def summarize(root):
    raw=root/"raw"
    scores=read(raw/"validation.json")
    cases={n:trace(raw/n) for n in ("original","expanded","race","default")}
    studies={}
    for n in ("frozen","context","token","cross","multiview","strength","low-rank","scale"):
        location=root/(n+"-screen")/"results.json"
        if location.is_file():
            d=read(location)
            studies[n]=dict(fits=d.get("actual_fits",d.get("fits")),
                            advanced=d.get("advance_to_raw_solve",d.get("passed")),result_sha256=sha(location))
    image=read(root/"anonymous-image/results.json")
    summary=dict(validation=scores,cold_runs=cases,training_only_studies=studies,image_development=dict(
                 fits=image["actual_fits"],members=image["members"],exported_rows=image["predicted_rows"],
                 training_rows=image["training_rows"],external_labels_opened=False),
                 new_model_api_calls=0,official_test_opened=False,H2_opened=False,
                 total_cold_task2_physical_fits=sum(c["physical_fits"] for c in cases.values()),
                 total_research_screen_physical_fits=sum(s["fits"] for s in studies.values()),
                 image_demonstration_physical_fits=image["actual_fits"])
    if (raw/"original_compute_matched/sealed.json").exists():
        summary["compute_matched_control"]=read(raw/"original_compute_matched/sealed.json")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    save(args.output,summarize(args.root))
