"""Test the unchanged controller on the next disjoint panel of shared-title groups."""
import argparse
from pathlib import Path

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.study_portfolio_recruitment import prepare,score,solve
from scripts.unified_released_study import ROOT, read, require, sha


PLAN=ROOT/"research/paired_encoder_search_1127/recruitment_replication_protocol.json"


def run(root,action):
    original=root/"portfolio-recruitment"
    out=root/"recruitment-replication"
    first=original/"comparison/results.json"
    template=read(original/"inputs.json")
    require(template["challenger"]==source_identity(ROOT) and read(first)["paired_accuracy"]["protected"]["net"]>0
            and read(original/"table/results.json")["received_score_after_sealing"],"Earlier replication conditions changed")
    if action=="prepare":
        prepare(root,ROOT,out=out,fold_index=1,comparison_only=True,plan=PLAN,trigger=first)
        return
    mounted=read(out/"inputs.json")
    require(mounted["resources"]==template["resources"] and mounted["plan"]==sha(PLAN) and mounted["trigger"]==sha(first),
            "Registered resources or protocol changed")
    if action=="solve":
        solve(root,ROOT,"comparison","challenger",out=out)
        return
    score(root,"comparison",out=out,own_incumbent_only=True)
    next_panel=out/"comparison/results.json"
    panels=[]
    for result in (first,next_panel):
        r=read(result)
        panels.append(dict(rows=r["scored"]["challenger"]["rows"],**r["paired_accuracy"]["protected"]))
    net=sum(p["net"] for p in panels)
    rows=sum(p["rows"] for p in panels)
    lock(out/"replication.json",dict(protocol=sha(PLAN),panel_receipts=[sha(first),sha(next_panel)],
         panels=panels,disjoint_rows=rows,net=net,pooled_gain=net/rows,
         repeated_positive_gain=all(p["net"]>0 for p in panels) and net/rows>=.002,
         held_group_overlap=0,model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(panels,net,rows,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","solve","score"))
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    a=p.parse_args()
    run(a.root,a.action)
