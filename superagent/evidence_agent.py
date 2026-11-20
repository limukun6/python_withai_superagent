"""Measured facts and reversible route memory for bounded experiment planning.

All three policies share fact computation, parameter definitions and proposal
validation. Only evidence visibility and the explicit route board differ.
The controller validates references/reading, not arbitrary prose or causality.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .hypothesis_agent import build_state, parse_proposal
from .metrics import score_of,completed,metric_of


EVIDENCE_STRATEGIES = ("evidence_blind", "evidence_feedback", "evidence_routes", "budget_routes")
ROUTE_STRATEGIES = ("evidence_routes", "budget_routes")
BASE_FIELDS = {"candidate", "reason", "hypothesis", "review", "evidence_trials", "expected_change"}
PARAMETERS = {
    "C": "Inverse regularization strength: smaller C means STRONGER regularization; increasing C weakens it. Scores alone do not identify underfitting or overfitting.",
    "min_samples_leaf": "Larger minimum leaf size constrains tree complexity more strongly.",
    "depth": "Larger tree depth permits more complexity; a measured gain is not guaranteed.",
    "l2_leaf_reg": "Larger values strengthen the leaf-weight L2 penalty.",
    "representation": "word uses word TF-IDF; char uses character TF-IDF; word_char combines both; word_char_stats adds generic shape/count features. Pair-text routes use left-minus-right differences; single-text routes classify one document. Features are fitted only on the training partition.",
    'supervised_feature_weighting':'nb_word fits smoothed preferred/nonpreferred word log-count ratios using training labels only; ratios are frozen for evaluation.',
    'classifier':'logistic optimizes log loss; linear_svm uses a squared hinge objective and has no probability output. Research candidates are unproven alternatives, not preferred solutions.',
    'style_weight':'Fixed probability weight on the statistic-tree component; the linear component gets 1-weight. More tree weight does not guarantee improvement.',
    'classifier_fits_per_evaluation':'A fixed blend or adaptive iteration selector plus full-partition refit trains two classifiers per candidate evaluation; count both, including final refits, when comparing compute.',
    'symmetric_probability':'Average forward and reversed-complement probabilities. Pair swap gives complementary probabilities; exact ties still use a fixed class tie-break.',
    'max_iterations':'Maximum selector length. Adaptive CatBoost chooses its final iteration count using a separate grouped subset inside the current training partition, then refits the whole partition.',
    'early_stopping_rounds':'Patience after the best training-internal loss. This selector must never use the outer search or provided validation labels.',
    'dimension':'Size of a randomly initialized trainable word representation, fitted on current training rows only. Larger is not necessarily better.',
    'maximum_classifier_fits_per_evaluation':'Conditional grouped stacking can fit up to17 classifiers including all OOF, epoch-selection, component-refit and fusion-head fits; report actual fits.',
}


def route_of(candidate):
    if candidate in ('native_histogram_categories','feedback_histogram_categories'):return 'histogram_categories'
    if candidate in ('feedback_boosting_iterations','boosting_smaller_steps'):return 'boosting_training_curve'
    if candidate in ('category_conditional_distributions','permuted_category_distributions','category_distributions_with_direct_splits'):return 'category_distributions'
    if candidate=='category_marginal_frequency':return 'category_frequency'
    if candidate in ('typed_category_features','typed_features_with_direct_splits'):return 'typed_category_features'
    if candidate in ('direct_category_splits','direct_splits_with_category_context'):return 'direct_category_splits'
    if candidate in ('scratch_neural_classifier','training_category_cooccurrence','minilm_category_levels'):return candidate
    if candidate in ('pretrained_qwen3_levels','ablation_permuted_qwen3_levels'):return 'pretrained_category_levels'
    if candidate == "pretrained_qwen3_context_C0.3":return "pretrained_qwen3_context"
    if candidate.startswith('generated_'):return candidate.rsplit('_',1)[0]
    if candidate.startswith("recipe_"):return candidate.rsplit("_",1)[0]
    for name in ('edit_pair','category_cross','style_tree','style_blend','symmetric_stats','adaptive_catboost','neural_pair','learned_stack'):
        if candidate.startswith(name+'_'):return name
    for name in ('word_char_stats_svm','nb_word'):
        if candidate.startswith(name+'_'):return name
    for prefix in ('numeric_tabular_','mixed_tabular_','text_tabular_','image_'):
        if candidate.startswith(prefix):
            suffix=candidate[len(prefix):]
            for name in ('logistic','extra_trees','rbf_svm'):
                if suffix.startswith(name+'_'):return prefix+name
    for name in ('single_word_char','single_word','single_char'):
        if candidate.startswith(name+'_'):return name
    if candidate.startswith("word_char_stats_"):
        return "word_char_stats"
    if candidate.startswith("word_char_"):
        return "word_char"
    for name in ("word", "char", "logistic", "extra_trees", "catboost"):
        if candidate.startswith(name + "_"):
            return name
    raise ValueError("Unknown catalog candidate route")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def measured_facts(history):
    """Only completed, finite measurements support numerical comparisons."""
    facts = []
    previous_best = None
    for row in history:
        trial = row["trial"]
        status = row["status"]
        facts.append({"id": f"t{trial}.status", "trial": trial, "metric": "status", "value": status})
        if status != "completed":
            continue
        for metric in ("train_score","search_score","score_gap","train_accuracy", "search_accuracy", "generalization_gap", "training_seconds",'selected_iterations','classifier_fits'):
            if finite(row.get(metric)):
                facts.append({"id": f"t{trial}.{metric}", "trial": trial, "metric": metric, "value": row[metric]})
        paired=row.get('paired_error_comparison',{})
        if paired.get('status')=='completed':
            for metric in ('evaluated_rows','newly_correct','newly_wrong','net_correct_change'):
                facts.append({'id':f't{trial}.{metric}','trial':trial,'metric':metric,'value':paired[metric],
                    'reference_trial':paired['reference_trial'],'scope':'same ordered internal search rows; descriptive only'})
        score = score_of(row)
        if not finite(score):
            continue
        if previous_best is not None:
            facts.append({"id": f"t{trial}.delta_vs_prior_best", "trial": trial,
                          "metric": "delta_vs_prior_best", "reference_trial": previous_best["trial"],
                          "value": score - score_of(previous_best),'primary_metric':metric_of(history)})
        if previous_best is None or score > score_of(previous_best):
            previous_best = row
    return facts


def latest_reading(history, visible=True):
    if not visible:
        return "unavailable"
    if not history:
        return "no_comparison"
    latest = history[-1]
    if latest["status"] != "completed":
        return "execution_failure"
    if score_of(latest) is None:
        return "not_evaluable"
    earlier = [score_of(row) for row in history[:-1] if completed(row)]
    if not earlier:
        return "no_comparison"
    delta = score_of(latest) - max(earlier)
    return "improved" if delta > 1e-12 else "lower" if delta < -1e-12 else "unchanged"


def latest_result(history,primary_metric=None):
    """Deterministic arithmetic is supplied as evidence, not asked of the LLM."""
    name=metric_of(history,primary_metric or 'accuracy')
    if primary_metric is not None and name!=primary_metric:raise ValueError('Task metric disagrees with history')
    latest = history[-1] if history else None
    earlier = [r for r in history[:-1] if completed(r)]
    best = max(earlier, key=lambda r: (score_of(r), -r["trial"])) if earlier else None
    score = score_of(latest) if latest and latest['status']=='completed' else None
    prior_score = score_of(best) if best else None
    return {"reading": latest_reading(history), "source": "executor",
            "latest_trial": latest["trial"] if latest else None,
            "best_prior_trial": best["trial"] if best else None,
            'primary_metric':name,'latest_search_score':score,'best_prior_search_score':prior_score,
            "latest_search_accuracy": latest.get('search_accuracy') if latest and latest['status']=='completed' else None,
            "best_prior_search_accuracy": best.get('search_accuracy') if best else None,
            "delta_vs_prior_best": score - prior_score if finite(score) and best else None,
            "comparison_rule": "latest completed result versus best earlier completed result; not significance"}


def route_board(history, available,primary_metric=None):
    """Rebuild from authoritative trials. A paused route is never removed."""
    name=metric_of(history,primary_metric or 'accuracy')
    if primary_metric is not None and name!=primary_metric:raise ValueError('Task metric disagrees with history')
    all_ids = [row["candidate"] for row in history] + [item["id"] for item in available]
    routes = sorted({route_of(value) for value in all_ids})
    last_route = route_of(history[-1]["candidate"]) if history else None
    notes = {}
    for row in history:
        proposal = row.get("proposal") or {}
        update = proposal.get("route_update")
        if update:
            notes.setdefault(update["selected_route"], {})["next_check"] = update["next_check"]
            for paused in update["paused_routes"]:
                notes.setdefault(paused["route"], {})["resume_if"] = paused["resume_if"]
    board = []
    for route in routes:
        trials = [row for row in history if route_of(row["candidate"]) == route]
        eligible = [row for row in trials if completed(row)]
        best = max(eligible, key=lambda row: (score_of(row), -row["trial"])) if eligible else None
        remaining = [item["id"] for item in available if route_of(item["id"]) == route]
        status = "exhausted" if not remaining else "untried" if not trials else "active" if route == last_route else "paused"
        board.append({"route": route, "status": status, "attempted_trials": [r["trial"] for r in trials],
                      "best_trial": best["trial"] if best else None,
                      "best_search_accuracy": best["search_accuracy"] if best else None,
                      'primary_metric':name,'best_search_score':score_of(best) if best else None,
                      "remaining_configurations": len(remaining), **notes.get(route, {})})
    return board


def state_for(profile, history, available, limit, strategy):
    visible = strategy != "evidence_blind"
    state = build_state(profile, history, available, limit, visible)
    # Keep each arm's own notes, but avoid repeated prose duplicating facts.
    for plan in state["plans"]:
        proposal = plan.get("proposal")
        if proposal:
            plan["proposal"] = {key: proposal[key] for key in ("hypothesis", "expected_change")}
    state["available_candidates"] = [{**item, "route": route_of(item["id"])} for item in available]
    used = {key for item in available for key in item["params"]}
    state["parameter_definitions"] = {key: value for key, value in PARAMETERS.items() if key in used}
    state["facts"] = measured_facts(history) if visible else []
    if visible:
        from .configuration_facts import latest_configuration_change
        from .prediction_evidence import visible_comparison
        state['latest_configuration_comparison']=latest_configuration_change(history)
        state['latest_error_comparison']=visible_comparison(history)
        if history and history[-1].get('error_slices'):
            state['latest_error_slices']=history[-1]['error_slices']
        from .hypothesis_memory import build_memory
        state['hypothesis_memory']=build_memory(history,available,profile.get('primary_metric'))
        state["latest_execution_status"] = history[-1]["status"] if history else None
        if history and history[-1].get('explanation_audit'):
            from .explanation_audit import visible_warning_context
            state['previous_explanation_review']=visible_warning_context(history[-1]['explanation_audit'])
    state["has_route_memory"] = strategy in ROUTE_STRATEGIES
    if state["has_route_memory"]:
        state["routes"] = route_board(history, available,profile.get('primary_metric'))
    if strategy == "budget_routes":
        state["latest_result"] = latest_result(history,profile.get('primary_metric'))
        remaining = max(0, limit - len(history))
        state["horizon"] = {
            "attempts_including_this": remaining,
            "followups_after_this": max(0, remaining - 1),
            "final_attempt": remaining == 1,
            "untried_routes": [r["route"] for r in state["routes"] if r["status"] == "untried"],
            "candidate_order_is_ranking": False,
        }
    return state


def text_field(value, name, maximum=300):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text up to {maximum} characters")
    return value.strip()


def parse_evidence(answer, history, available, strategy):
    routes = strategy in ROUTE_STRATEGIES
    visible = strategy != "evidence_blind"
    required = BASE_FIELDS | {"assessment"} | ({"route_plan"} if routes else set())
    if not isinstance(answer, dict) or set(answer) != required:
        raise ValueError("Incorrect evidence proposal fields")
    base = parse_proposal({key: answer[key] for key in BASE_FIELDS}, history, available, visible).record()
    assessment = answer["assessment"]
    executor_reading = strategy == "budget_routes"
    assessment_fields = {"fact_ids", "interpretation", "uncertainty"} | (set() if executor_reading else {"reading"})
    if not isinstance(assessment, dict) or set(assessment) != assessment_fields:
        raise ValueError("Incorrect assessment fields")
    reading = latest_reading(history, visible)
    if not executor_reading and assessment["reading"] != reading:
        raise ValueError("Reading contradicts the measured latest result")
    fact_ids = assessment["fact_ids"]
    facts = {fact["id"]: fact for fact in measured_facts(history)} if visible else {}
    if (not isinstance(fact_ids, list) or any(not isinstance(value, str) for value in fact_ids)
        or len(fact_ids) > 4 or len(set(fact_ids)) != len(fact_ids) or not set(fact_ids) <= set(facts)):
        raise ValueError("Invalid or fabricated fact reference")
    if visible and history and (not fact_ids or not any(facts[key]["trial"] == history[-1]["trial"] for key in fact_ids)):
        raise ValueError("Assessment must cite a fact from the latest trial")
    if visible and not {facts[key]["trial"] for key in fact_ids} <= set(base["evidence_trials"]):
        raise ValueError("Evidence trials must include every cited fact's trial")
    base["assessment"] = {"reading": reading, "fact_ids": fact_ids,
                          "interpretation": text_field(assessment["interpretation"], "interpretation"),
                          "uncertainty": text_field(assessment["uncertainty"], "uncertainty")}
    if executor_reading:
        base["assessment"]["reading_source"] = "executor"
    # Store facts from the executor, never accept numerical restatements as truth.
    base["cited_facts"] = [deepcopy(facts[key]) for key in fact_ids]
    if routes:
        update = answer["route_plan"]
        if not isinstance(update, dict) or set(update) != {"next_check", "revisit_previous_if"}:
            raise ValueError("Incorrect route plan fields")
        selected = route_of(base["candidate"])
        prior_routes = {route_of(row["candidate"]) for row in history}
        last = route_of(history[-1]["candidate"]) if history else None
        action = "continue" if selected == last else "revisit" if selected in prior_routes else "start"
        next_check = text_field(update["next_check"], "next_check", 200)
        revisit = text_field(update["revisit_previous_if"], "revisit_previous_if", 200)
        base["route_plan"] = {"next_check": next_check, "revisit_previous_if": revisit}
        # Route identity/status are deterministic facts, not another LLM task.
        paused = [{"route": last, "resume_if": revisit}] if last is not None and last != selected else []
        base["route_update"] = {"selected_route": selected, "action": action, "next_check": next_check, "paused_routes": paused}
    return base


INSTRUCTION = """Select one available experiment. Return concise JSON only, with exactly:
candidate, reason, hypothesis, review, evidence_trials, expected_change, assessment;
add route_plan ONLY when has_route_memory=true.
candidate is an untried ID. reason/review/hypothesis are short experiment notes.
Facts are measured by the executor. Treat interpretation as an uncertain explanation,
not another measurement. One configuration's result cannot reject its entire route.
latest_error_comparison counts newly correct and newly wrong predictions against the
best earlier result on the SAME reused internal rows. A net gain can contain many
opposing changes; these counts establish neither significance nor ensemble benefit.
Read parameter_definitions carefully (increasing C WEAKENS regularization).
expected_change is null or {metric, reference_trial, direction, min_delta}; metric is
search_score/score_gap/search_accuracy/generalization_gap/training_seconds; direction increase/decrease.
Accuracy changes are fractions, not percentage points. Use only prior trial IDs.
assessment has exactly {reading, fact_ids, interpretation, uncertainty}. Strings <=300 chars.
First check latest_execution_status. If it is failed or timeout, reading MUST be
execution_failure, even though no numeric comparison can be made. Only if the
latest run completed, compare it with the BEST EARLIER completed primary search_score:
improved/lower/unchanged (tolerance 1e-12), no_comparison if no earlier score,
not_evaluable for a completed run with missing score,
unavailable if feedback_visible=false. A numerical difference does not establish
statistical significance. A failed execution is not measured zero accuracy.
fact_ids cites 1..4 provided fact IDs including the latest trial when visible;
evidence_trials contains their trial IDs. When blind both lists are empty, review
acknowledges unknown results, and no result can be asserted. interpretation gives
one possible explanation; uncertainty states what these experiments cannot establish.
With route memory: route_plan={next_check, revisit_previous_if}, both short strings
<=200 chars. next_check states a useful follow-up question. When leaving the most
recent route, revisit_previous_if states a concrete reason to return to that route.
When continuing it, state what evidence would cause you to reconsider the current
direction. The controller derives route IDs, continue/start/revisit, and paused
status from the chosen candidate and history; do not output these metadata fields.
Pause preserves every untried configuration.
Choose continued investigation, a new route, or a revisit based on evidence, prior
plans and remaining attempts. No route gets a mandatory turn and no fixed explore ratio.
Keep the hypothesis testable, distinguish its scope from the whole model family,
and consider what would change your decision. Best-model retention is automatic.
Do not ask for holdout/test scores, external data, arbitrary code, or more trials."""

BUDGET_INSTRUCTION = """
Use horizon to plan for a finite experiment budget. Candidate order is not a ranking.
Before choosing, weigh the strongest available alternative to another small step
on the current route. Several adjacent parameter checks can use up the budget
without testing a promising different representation or model family. Early
experiments can leave time to refine a promising route; an untried route also has
uncertain value and is never mandatory. State the tradeoff briefly in reason.
On the final adaptive attempt, choose for expected final model quality under
profile.evaluation_protocol.planned_final_selection when supplied. Fixed multi-split
confirmation may follow, but no further adaptive proposal is available. Do not choose
solely to plan another proposal that cannot occur. Keep a route or return to an
earlier one when its evidence warrants it; one disappointing
configuration is insufficient to discard a family. The best prior model is kept
automatically even if the next experiment fails. Do not claim statistical proof
from small score changes. Do not change the required JSON fields."""


EXECUTOR_FACT_INSTRUCTION = """Select one available classification experiment.
Return concise JSON only, with exactly these fields:
candidate, reason, hypothesis, review, evidence_trials, expected_change, assessment, route_plan.
candidate is an untried ID. reason/review/hypothesis are short testable experiment notes.
Read parameter_definitions: smaller C means stronger regularization.
Use search_score for the declared primary metric. Accuracy and paired correct counts remain diagnostics; they need not improve when macro_f1 improves.
Hypothesis memory checks only numerical expectations, not causal explanations.
Measured facts and latest_result are authoritative executor output. latest_result
already states whether the latest result improved on the best earlier score,
fell, tied, failed to execute, or lacks a comparison. Do not calculate or emit
a redundant reading label. A timeout is not measured zero accuracy.
assessment has exactly {fact_ids, interpretation, uncertainty}. No reading field.
fact_ids cites 1..4 provided fact IDs, including the latest trial when history exists.
evidence_trials contains all their prior trial IDs; neither list may invent references.
interpretation gives a possible explanation; uncertainty states what remains unknown.
Both strings must be nonempty and <=300 characters. A small score change does not
establish significance, and one bad configuration cannot reject an entire route.
expected_change is null or exactly {metric, reference_trial, direction, min_delta}.
metric: search_score/score_gap/search_accuracy/generalization_gap/training_seconds; direction: increase/decrease;
reference_trial is a prior trial ID; min_delta is nonnegative, using fractions for accuracy.
route_plan is exactly {next_check, revisit_previous_if}, nonempty strings <=200 chars.
next_check is a useful follow-up question. When switching routes, revisit_previous_if
gives a reason to return; when continuing, give a reason to reconsider the direction.
The controller derives route identity and start/continue/revisit from your candidate,
preserves paused routes and automatically keeps the best previous model.
Use latest_result for exact arithmetic. Do not call a measured decline an exact tie,
or assert that a difference is statistically significant/non-significant without a
provided test. Free-form explanations are hypotheses, not additional measurements.
If previous_explanation_review is present, use its executor facts and warning codes;
do not repeat a flagged earlier explanation as established evidence.
Choose a continuation, new route or revisit independently from the evidence and horizon.
Do not ask for holdout/test scores, external data, arbitrary code, or extra trials."""


class EvidenceAgent:
    def __init__(self, client, logger, limit, strategy):
        self.client, self.logger, self.limit, self.strategy = client, logger, limit, strategy
        self.last_proposal = None

    def choose(self, profile, history, available):
        self.last_proposal = None
        state = state_for(profile, history, available, self.limit, self.strategy)
        instruction = (EXECUTOR_FACT_INSTRUCTION + BUDGET_INSTRUCTION
                       if self.strategy == "budget_routes" else INSTRUCTION)
        answer = self.client.ask({"instruction": instruction, **state}, self.logger, "evidence_planner")
        self.last_proposal = parse_evidence(answer, history, available, self.strategy)
        return self.last_proposal["candidate"], self.last_proposal["reason"]
