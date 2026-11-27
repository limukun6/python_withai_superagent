"""Collect receipts and count actual base, mixture and feedback fits in the 48-fit batch."""
import argparse
from pathlib import Path

from scripts.study_task1_remaining_folds import ARMS,FOLDS
from scripts.unified_released_study import ROOT,read,require,save,sha

REGISTRY=ROOT/"research/task1_feedback_extension_1127"


def export(out,destination):
    previous_path=REGISTRY/"method_transfer_results.json"
    prior=read(previous_path)
    dev=out/"development"
    nine=out/"remaining_folds"
    match=read(nine/"result.json")
    paths={"ordinary":nine/"result.json","measurement_nominee":out/"measurement_nominee_result.json",
           "latent_large":out/"latent_diagnostic_result.json","centered_development":out/"latent_centering_check/result.json",
           "direct_original_comparison":out/"incumbent_comparison/result.json","selection_variability":out/"selection_noise_result.json",
           "portfolio":out/"portfolio_stacking/result.json","typed_table":dev/"typedicl_election/results.json",
           "typed_table_nomination":dev/"typedicl_election/locked.json","group_offsets":dev/"groupoffset_election/results.json",
           "group_offsets_vs_original":dev/"groupoffset_election/vs_original.json",
           "latent_dev":dev/"latent_election/results.json","latent_dev_vs_original":dev/"latent_election/vs_original.json"}
    result={k:read(p) for k,p in paths.items()}
    receipts={k:sha(p) for k,p in paths.items()}
    typed=sorted((dev/"typed_icl_components").glob("*/sealed.json"))
    require(len(typed)==7,"Unverified number of added transformer contexts")
    counts=dict(ordinary_task_fits=0,typed_table_context_fits=sum(read(p)["context_fits"] for p in typed),
                residual_corrections=sum(read(dev/("groupoffset_"+n)/"sealed.json")["correction_fits"] for n in ("main","interaction")),
                portfolio_weight_fits=result["portfolio"]["router_fits"],
                latent_development=sum(read(dev/("latent_"+n)/"sealed.json")["mixture_fits"] for n in ("two","four")),
                centered_development=result["centered_development"]["model_fits"],
                frozen_measurement_recipe=result["measurement_nominee"]["classifier_fits"],
                larger_latent_mixtures=0,larger_latent_new_teachers=0)
    internal=dict(typed_additional_preprocessing_views=sum(read(p)["views"] for p in typed),
                  latent_group_initializations=3,latent_joint_m_steps=42,larger_latent_verified_teachers_reused=0)
    seals={}
    for i in FOLDS:
        root=nine/str(i)
        by_arm={}
        for arm in ARMS:
            side=root/arm
            d=read(side/"sealed.json")
            launch=read(side/"launch.json")
            verified=d["independent_reload"] and d["launch"]==sha(side/"launch.json") and launch["cap"]==48
            require(verified and not d["model_api_requests"] and d["fits"]<=48,"Unverified ordinary fit accounting")
            counts["ordinary_task_fits"]+=d["fits"]
            by_arm[arm]=dict(launch_sha256=sha(side/"launch.json"),sealed_sha256=sha(side/"sealed.json"),
                actual_fits=d["fits"],seconds=round(d["seconds"],2),source=launch["source"],members=d["members"])
        d=read(root/"latent_diagnostic/sealed.json")
        counts["larger_latent_mixtures"]+=d["latent_model_fits"]
        counts["larger_latent_new_teachers"]+=d["new_teacher_fits"]
        internal["latent_group_initializations"]+=d["initializations"]
        internal["latent_joint_m_steps"]+=d["joint_m_steps"]
        internal["larger_latent_verified_teachers_reused"]+=d["reused_teacher_fits"]
        by_arm["latent_component_seal"]=sha(root/"latent_diagnostic/sealed.json")
        by_arm["measurement_component_seal"]=sha(root/"measurement_nominee/sealed.json")
        seals[str(i)]=by_arm
    if match["advance"]:
        require((out/"designated_result.json").is_file(),"A qualified release still needs its raw designated grade")
        paths["designated"]=out/"designated_result.json"
        result["designated"]=read(paths["designated"])
        receipts["designated"]=sha(paths["designated"])
        counts["raw_qualified_entry"]=read(out/"release/result.json")["actual_fits"]
    value=dict(previous_report_sha256=sha(previous_path),previous_total=prior["accumulated_completed_fits"],
               receipts=receipts,ordinary_entry_receipts=seals,results=result,new_fits=counts,
               new_total=sum(counts.values()),accumulated_task_and_router_fits=prior["accumulated_completed_fits"]+sum(counts.values()),
               internal_operations_separately=internal,ordinary_frozen_commit="5705396ed92fab364c5fcdb08a5452f1d5d47f2b",
               partner_historical_designated=dict(correct=1549,rows=2030),
               designated_validation_opened=bool(match["advance"]),official_test_opened=False,h2_opened=False,
               language_model_api_requests=0,campus_input_tokens=0,campus_output_tokens=0,course_api_cost_rmb=0)
    save(destination,value)
    print(counts,"increment",value["new_total"],"cumulative",value["accumulated_task_and_router_fits"],flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    for name in ("out","destination"):
        arg.add_argument("--"+name,required=True,type=lambda p:Path(p).resolve())
    a=arg.parse_args()
    export(a.out,a.destination)
