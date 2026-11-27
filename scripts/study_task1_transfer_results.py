"""Summarize encoder transfer, out-of-fold teaching and selection replay without row-level files."""
import argparse
from pathlib import Path

from scripts.unified_released_study import ROOT, read, require, save, sha

DIR=ROOT/"research/task1_feedback_extension_1127"
SOURCES={"semantic":("e5_tree","e5_linear","bge_tree","bge_linear"),
         "binding":("global","pooled"), "distill":("constant","agreement")}


def export(out, destination):
    prior=read(DIR/"followup_results.json")
    bank={}
    counts={}
    dev=out/"development"
    for family, candidates in SOURCES.items():
        path=dev/(family+"_election")
        election=read(path/"results.json")
        require(election["lock"]==sha(path/"locked.json"),"Changed election")
        locked=read(path/"locked.json")
        training={}
        for name in candidates:
            seal=dev/(family+"_"+name)/"sealed.json"
            record=read(seal)
            require(record["protocol"]==locked["protocol"] and not record["api_requests"],"Changed method study")
            training[name]=dict(seal=sha(seal),**record)
        bank[family]=dict(candidates=training,nomination=locked["alternatives"],reference=locked["reference"],
                          lock=election["lock"],score_sha256=sha(path/"results.json"),
                          winner=election["winner"],checks=election["checks"],eligible=election["eligible"])
        metric={"semantic":"supervised_fits","binding":"additional_classifier_fits","distill":"student_fits"}[family]
        counts[family]=sum(d[metric] for d in training.values())
    teacher=dev/"distill_teachers/sealed.json"
    counts["distill_shared_teachers"]=read(teacher)["fits"]
    diagnostics={}
    for tag,path in (("replica",out/"replica_five_result.json"),("allocation",out/"evidence_allocation/result.json")):
        diagnostics[tag]=dict(sha256=sha(path),**read(path))
    actual=sum(counts.values())
    result=dict(prior_sha256=sha(DIR/"followup_results.json"),prior_completed_fits=prior["accumulated_completed_fits"],
                new_fits=counts,new_fits_total=actual,accumulated_completed_fits=prior["accumulated_completed_fits"]+actual,
                synthetic_engineering_probe_fits=1,shared_teacher_record=dict(sha256=sha(teacher),**read(teacher)),
                studies=bank,diagnostics=diagnostics,api_requests=0,campus_input_tokens=0,campus_output_tokens=0,
                course_api_cost_rmb=0,designated_validation_opened=False,official_test_opened=False,h2_opened=False)
    save(destination,result)
    print(counts,result["accumulated_completed_fits"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("out","destination"):
        parser.add_argument("--"+arg,required=True,type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    export(args.out,args.destination)
