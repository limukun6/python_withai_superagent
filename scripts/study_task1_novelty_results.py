"""Publish aggregate Task1 evidence without exporting models, input rows or predictions."""
import argparse
import json
from pathlib import Path

from scripts.study_task1_accuracy_veto import STAGE as VETO
from scripts.study_task1_catalogue_recovery import STAGE as MATCHED
from scripts.study_task1_conditional_opportunity import STUDY as CONDITIONAL
from scripts.study_task1_disagreement_reading import STAGE as HEADS
from scripts.study_task1_population_expert import STAGE as POPULATION
from scripts.study_task1_unseen_profiles import FOLDS, STUDY as STRESS, completed_arm
from scripts.unified_released_study import ROOT, read, require, save, sha

REGISTRY = ROOT / "research/task1_feedback_extension_1127"


def export(out):
    full = out / MATCHED
    paths = {k:out / d / "result.json" for k,d in dict(conditional=CONDITIONAL, disagreement=HEADS,
                                                       impaired_stress=STRESS, veto=VETO).items()}
    paths.update(stress=full / STRESS / "result.json", population=full / POPULATION / "result_0-3.json",
                 impaired_population=out / POPULATION / "result_0-3.json")
    results = {k:read(p) for k,p in paths.items()}
    for evidence in results.values():
        require(evidence["model_api_requests"] == 0 and not any(evidence[k] for k in
                ("official_test_opened", "designated_validation_opened", "h2_opened")), "Study boundary changed")
    c, h, s, pop, veto = (results[k] for k in ("conditional", "disagreement", "stress", "population", "veto"))
    require(c["rows"] == h["rows"] == veto["rows"] and
            c["correct"]["control"] == h["correct"]["retain"] == veto["correct"]["control"], "Regular controls differ")
    impaired, alt = results["impaired_stress"], results["impaired_population"]
    require(impaired["rows"] == alt["rows"] == s["rows"] == pop["rows"] and
            alt["correct"]["control"] == impaired["correct"]["frozen"] and pop["correct"]["control"] == s["correct"]["frozen"] and
            s["profile_overlap"] == 0, "Novel-profile controls differ")
    physical, last_known = [], []
    for i in FOLDS:
        fold = out / STRESS / str(i)
        for label, directory in (("full", full / STRESS / str(i)), ("impaired", fold)):
            arm = completed_arm(directory)
            rc = read(arm / "sealed.json")
            launch = read(arm / "launch.json")
            observed = (arm / "physical.jsonl").read_text().splitlines()
            count = len(observed)
            require(rc["fits"] == count and rc["launch"] == sha(arm / "launch.json")
                    and rc["independent_reload"] and rc["prediction"] == sha(arm / "solve.csv") == sha(arm / "reload.csv")
                    and not launch["method_hint"] and launch["cap"] == 64, "Incomplete ordinary comparison")
            physical.append(dict(fold=i, resource=label, successful_solve=sha(arm / "sealed.json"), fits=count,
                 model=rc["model"], predictions=rc["prediction"], seconds=round(rc["seconds"],2), members=rc["members"],
                 frozen_context_calls=sum(json.loads(line)["classifier"]=="TabICLClassifier" for line in observed)))
        interrupted = fold / "interruption.json"
        if interrupted.is_file():
            prev = read(interrupted)
            original = fold / "replicated"
            latest = max(p.stat().st_mtime for p in original.rglob("*") if p.is_file())
            observed_seconds = latest - (original / "launch.json").stat().st_mtime
            last_known.append(dict(fold=i, attempted=prev["physical_attempts"], rows_in_partial=prev["partial_rows"],
                 interruption=sha(interrupted), last_recorded_activity_seconds=round(observed_seconds,1),
                 exact_wall_seconds=None))
    condition_cost = {}
    for method, field in (("control","additional_control_fits"), ("posterior","classifier_fits"), ("crossed","classifier_fits")):
        condition_cost[method] = sum(read(out / CONDITIONAL / str(i) / method / "sealed.json")[field] for i in range(5))
    require(sum(condition_cost.values()) == c["actual_new_fits"] and
            sum(z["attempted"] for z in last_known) == impaired["interrupted_fits"], "Training accounting differs")
    counts = dict(rebuilt_regular_reference=condition_cost["control"], posterior_trees=condition_cost["posterior"],
                  crossed_logistic=condition_cost["crossed"], small_feedback_optimizers=h["new_optimizer_fits"],
                  completed_full_resource_solves=s["fits"], interrupted_stress_solves=impaired["interrupted_fits"],
                  impaired_resource_solves=impaired["fits"], impaired_population_comparison=alt["new_fits"],
                  matched_population_comparison=pop["new_fits"], decision_replay=veto["new_fits"])
    prior = REGISTRY / "foundation_completion_results.json"
    frozen = REGISTRY / "designated_release.json"
    reference = read(frozen)
    designated = reference["records"]["designated"]
    output = dict(source_commit=reference["verified_solver_git_commit"], frozen_receipt=sha(frozen),
         previous_designated=dict(correct=designated["release_correct"], rows=designated["rows"], scored_again=False),
         previous_cumulative=read(prior)["cumulative"], previous_receipt=sha(prior),
         results=results, source_evidence={k:sha(v) for k,v in paths.items()}, cold_runs=physical,
         hardware_interruptions=last_known, same_evaluation_rows=read(full / "same_rows.json"),
         full_resource_validation=read(full / "resource_outcomes.json"),
         full_resource_probe=read(full / "resource_probe.json"), training=counts, new_fit_units=sum(counts.values()),
         auxiliary_feature_estimation=dict(conditional_response_maps=c["supervised_response_maps"],
                                           population_crossfit_route_constructions=pop["route_constructions"]+alt["route_constructions"]),
         cumulative=read(prior)["cumulative"]+sum(counts.values()), model_api_requests=0, new_api_tokens=0,
         new_model_api_currency_rmb=0, designated_validation_opened=False, official_test_opened=False, h2_opened=False)
    save(REGISTRY / "novelty_and_feedback_results.json", output)
    print(counts, output["new_fit_units"], output["cumulative"], flush=True)


def export_followup(out):
    from scripts.study_task1_novel_seed_feedback import STAGE as SEEDS
    from scripts.study_task1_profile_runnerup import STAGE as RUNNERUP
    previous = REGISTRY / "novelty_and_feedback_results.json"
    primary = read(previous)
    target = out / MATCHED
    seed, backup = (read(target / stage / "result.json") for stage in (SEEDS,RUNNERUP))
    for value in (seed,backup):
        require(value["rows"] == primary["results"]["stress"]["rows"]
                and value["correct"]["ordinary"] == primary["results"]["stress"]["correct"]["frozen"]
                and value["model_api_requests"] == 0 and not any(value[k] for k in
                    ("h2_opened","official_test_opened","designated_validation_opened")), "Follow-up control differs")
    fits = seed["new_fits"] + backup["new_fits"]
    value = dict(previous=sha(previous), seed_override=seed, runnerup=backup, new_fit_units=fits,
                 current_round_fit_units=primary["new_fit_units"]+fits, cumulative=primary["cumulative"]+fits,
                 extra_route_constructions=backup["route_constructions"], model_api_requests=0, new_api_tokens=0,
                 new_model_api_currency_rmb=0, designated_validation_opened=False, official_test_opened=False, h2_opened=False)
    save(REGISTRY / "novel_profile_decision_followup_results.json", value)
    print("follow-up fits", fits, "round", value["current_round_fit_units"], "cumulative", value["cumulative"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    parser.add_argument("--followup",action="store_true")
    args=parser.parse_args()
    (export_followup if args.followup else export)(args.out)
