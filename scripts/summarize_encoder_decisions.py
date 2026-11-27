"""Export the ordinary menu decisions without task rows, raw text or host paths."""
import argparse
from pathlib import Path

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import read, require, sha


def summarize(run, output, candidate=None):
    agent=run/"run/agent"
    result=read(agent/"run_outcome.json")
    review=read(agent/"analogy_review.json")
    require(result["status"]=="completed" and result["training_only"],"Expected a completed training-only election")
    steps=[]
    for d in read(agent/"decisions.json"):
        opts=d["alternatives"]
        choice=d["choice"]
        row=dict(step=d["step"],source=d["source"],remaining=d["remaining_fits"],observed=d["observed_trials"],
                 alternatives=len(opts),action=choice["action"],members=choice["members"],split=choice["split"])
        if candidate:
            wanted=[(i+1,a) for i,a in enumerate(opts) if a["action"]=="explore" and a["members"]==[candidate]]
            if wanted:
                rank,a=wanted[0]
                ties=[v["members"][0] for v in opts if v["action"]=="explore" and v["utility"]==a["utility"]]
                row["candidate"]=dict(menu_position=rank,utility=a["utility"],equal_utility_explorers=ties,
                                      exactly_tied_with_chosen=choice["action"]=="explore" and choice["utility"]==a["utility"])
        steps.append(row)
    summary=dict(sealed=sha(run/"sealed.json"),decisions=sha(agent/"decisions.json"),history=sha(agent/"history.json"),
        review=sha(agent/"analogy_review.json"),tracked_candidate=candidate,steps=steps,
        measurements=[{k:t.get(k) for k in ("trial","candidate","status","split_id","search_score")}
                      for t in read(agent/"history.json")],member_source=review["source_member"],
        derived_encoder=review["derived_configuration"]["encoder"],derived_status=review["status"],
        nominated_share=review["locked_weight"],paired_confirmations=review["allocations"],
        members=result["members"],final_weights=read(agent/"final_selection.json").get("weights"),
        fits=result["attempted_classifier_fits"],api=result["new_api_requests"])
    lock(output,summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--candidate")
    args=p.parse_args()
    summarize(args.run,args.output,args.candidate)
